# from src.agents.orchestrator import OrchestratorAgent


# conf_values = [0.8, 0.7, 0.6, 0.5, 0.4]

# def crop_tool(input_data):
#     return {
#         "type": "crop",
#         "count": 10,
#         "avg_confidence": conf_values.pop(0)
#     }

# def traffic_tool(input_data):
#     return {
#         "type": "traffic",
#         "count": 25,
#         "avg_confidence": 0.75
#     }

# agent = OrchestratorAgent(crop_tool, traffic_tool)

# if __name__ == "__main__":
#     input_data = {
#         "type": "traffic"       # change to "traffic" to test
#     }

#     agent.run(input_data)



# import os
# from dotenv import load_dotenv

# os.environ["OPENAI_API_KEY"] = "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"

# # 1. Load the .env file FIRST before any other imports
# load_dotenv()

# # 2. Now import your project modules
# from src.agents.orchestrator import OrchestratorAgent
# from src.crop_pipeline.crop_tool import crop_tool
# from src.traffic_pipeline.traffic_tool import traffic_tool

# def main():
#     # Initialize the agent with the tools
#     agent = OrchestratorAgent(crop_tool, traffic_tool)

#     # Define the data batches
#     batches = [
#         {"type": "traffic", "video_path": "data/raw/sample.mp4"},
#         {"type": "crop", "image_path": "data/raw/sample.png"},
#         {"type": "traffic", "video_path": "data/raw/sample.mp4"}
#     ]

#     # Process each batch
#     for i, batch in enumerate(batches):
#         print(f"\n--- Batch {i+1} ---")
#         try:
#             agent.run(batch)
#         except Exception as e:
#             print(f"❌ Error processing batch {i+1}: {e}")

# if __name__ == "__main__":
#     main()



import os
import sys
from dotenv import load_dotenv

# Force the environment to use the key from your screenshot
os.environ["GROQ_API_KEY"] = "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"
os.environ["ACTIVE_PROVIDER"] = "groq"

load_dotenv() # This will load other vars, but the ones above are now forced

# DEBUG: This will print in your terminal so you can verify the key is active
print(f"--- DEBUG: ACTIVE_PROVIDER is {os.environ.get('ACTIVE_PROVIDER')} ---")

# =========================
# 🔹 SET YOUR FREE API KEY
# =========================
# Uncomment the provider you want to use and paste your key.
# Only ONE should be active at a time.
# Get free keys (no credit card needed):

# GROQ        → https://console.groq.com           (30 RPM, free forever)
os.environ.setdefault("GROQ_API_KEY", "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")
os.environ.setdefault("ACTIVE_PROVIDER", "groq")

# CEREBRAS    → https://cloud.cerebras.ai          (ultra fast, free)
# os.environ.setdefault("CEREBRAS_API_KEY", "your_cerebras_key_here")
# os.environ.setdefault("ACTIVE_PROVIDER", "cerebras")

# MISTRAL     → https://console.mistral.ai         (1B tokens/month free)
# os.environ.setdefault("MISTRAL_API_KEY", "your_mistral_key_here")
# os.environ.setdefault("ACTIVE_PROVIDER", "mistral")

# COHERE      → https://dashboard.cohere.com       (1K calls/month free)
# os.environ.setdefault("COHERE_API_KEY", "your_cohere_key_here")
# os.environ.setdefault("ACTIVE_PROVIDER", "cohere")

# HUGGINGFACE → https://huggingface.co/settings/tokens  (free serverless)
# os.environ.setdefault("HF_TOKEN", "hf_your_token_here")
# os.environ.setdefault("ACTIVE_PROVIDER", "huggingface")

# OPENROUTER  → https://openrouter.ai/keys         (50 RPD, 24+ free models)
# os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-your_key_here")
# os.environ.setdefault("ACTIVE_PROVIDER", "openrouter")

# ── Now import project modules ──────────────────────────────────────────
from src.agents.orchestrator import OrchestratorAgent
from src.crop_pipeline.crop_tool import crop_tool
from src.traffic_pipeline.traffic_tool import traffic_tool


def main():
    # Initialize the orchestrator with both CV tools
    agent = OrchestratorAgent(crop_tool, traffic_tool)

    # Define your input batches
    # Add or remove batches as needed
    batches = [
        {"type": "traffic", "video_path": "data/raw/traffic.mp4"},
        {"type": "crop",    "image_path": "data/raw/sample.png"},   # replace with actual crop image
        {"type": "traffic", "video_path": "data/raw/sample.mp4"},
    ]

    print(f"\n{'='*50}")
    print(f"  B2B AI Assessment — Task 3 Pipeline")
    print(f"  Total batches: {len(batches)}")
    print(f"{'='*50}")

    results = []
    for i, batch in enumerate(batches):
        print(f"\n{'─'*40}")
        print(f"  Batch {i+1}/{len(batches)}  |  type={batch.get('type')}")
        print(f"{'─'*40}")
        try:
            result = agent.run(batch)
            results.append({"batch": i + 1, "status": "success", "result": result})
        except Exception as e:
            print(f"❌ Error processing batch {i+1}: {e}")
            results.append({"batch": i + 1, "status": "error", "error": str(e)})

    # Final summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*50}")
    print(f"  Pipeline complete: {success}/{len(batches)} batches succeeded")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()