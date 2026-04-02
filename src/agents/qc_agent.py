from src.agents.llm_utils import call_llm


class QCAgent:
    """
    Quality Control Agent.
    Evaluates inference results and decides ACCEPT or REPROCESS.
    Uses the free LLM provider configured in llm_utils.py.
    """

    def evaluate(self, result: dict) -> dict:
        prompt = f"""
You are a Quality Control Agent for a Computer Vision pipeline.

Analyze the inference output below and evaluate its reliability.

--- Output ---
Task Type        : {result.get('type', 'unknown')}
Objects Detected : {result.get('objects_detected', 0)}
Avg Confidence   : {result.get('avg_confidence', 0.0):.3f}
Image Condition  : {result.get('condition', 'unknown')}
Extra Info       : {result.get('extra', 'none')}

--- Your Tasks ---
1. Determine if the output is reliable (confidence >= 0.45 is reliable)
2. Identify possible issues (blur, low-light, occlusion, fog, off-angle)
3. Decide if reprocessing is needed

Respond STRICTLY in this format (no extra text):
STATUS: OK or LOW
REASON: <one sentence>
ACTION: REPROCESS or ACCEPT
"""

        response = call_llm(prompt)

        print("\n[QC AGENT RESPONSE]")
        print(response)
        print()

        response_upper = response.upper()

        if "LOW" in response_upper or "REPROCESS" in response_upper:
            return {
                "status":   "LOW_CONFIDENCE",
                "response": response,
            }
        else:
            return {
                "status":   "OK",
                "response": response,
            }