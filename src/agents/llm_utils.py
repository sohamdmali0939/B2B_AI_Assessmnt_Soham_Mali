import os
import sys
from dotenv import load_dotenv

# 1. Add current dir to path & load .env
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()


os.environ["ACTIVE_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = "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"

ACTIVE_PROVIDER = os.getenv("ACTIVE_PROVIDER", "groq")

PROVIDERS = {
    "groq": {
        "base_url":    "https://api.groq.com/openai/v1",
        "model":       "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "signup":      "https://console.groq.com",
        "limit":       "30 RPM · 14,400 RPD — free forever",
    },
    
}


def _smart_fallback(prompt: str) -> str:
    p = prompt.lower()
    if "crop or traffic" in p or "answer only one word" in p:
        if "video_path" in p or "traffic" in p:
            return "traffic"
        return "crop"
    if "quality control" in p or "status:" in p or "avg_confidence" in p:
        return "STATUS: OK\nREASON: Fallback — LLM unavailable, defaulting to ACCEPT\nACTION: ACCEPT"
    if "report" in p or "summary" in p or "batch" in p:
        return "Batch processed successfully. LLM reporting unavailable — check API key."
    return "OK"

# =========================
# 🔹 MAIN call_llm FUNCTION
# =========================
def call_llm(prompt: str, model: str = None) -> str:
    cfg = PROVIDERS.get(ACTIVE_PROVIDER)
    if not cfg:
        return _smart_fallback(prompt)

    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        print(f"⚠️ API key not set for {ACTIVE_PROVIDER}. Using fallback.")
        return _smart_fallback(prompt)

    active_model = model or cfg["model"]

    
    if cfg.get("sdk") == "cohere":
        try:
            import cohere
            co = cohere.ClientV2(api_key=api_key)
            res = co.chat(
                model=active_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return res.message.content[0].text.strip()
        except Exception as e:
            print(f"⚠️ Cohere fail: {e}")
            return _smart_fallback(prompt)

  
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        response = client.chat.completions.create(
            model=active_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ {ACTIVE_PROVIDER.upper()} fail: {e}")
        return _smart_fallback(prompt)