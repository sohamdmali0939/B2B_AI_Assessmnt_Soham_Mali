import os
import json
from datetime import datetime
from src.agents.llm_utils import call_llm, ACTIVE_PROVIDER


class ReportingAgent:
    """
    Reporting Agent.
    Generates natural language batch summaries and anomaly alerts
    using the free LLM provider configured in llm_utils.py.
    Saves reports to outputs/reports/.
    """

    def __init__(self):
        self.report_dir = os.path.join("outputs", "reports")
        os.makedirs(self.report_dir, exist_ok=True)

    def generate(self, result: dict, memory) -> str:
        """
        Generates a natural language summary and saves it to disk.
        Returns the summary string.
        """
        history     = memory.get_all()
        total       = len(history)
        confidences = [r.get("avg_confidence", 0) for r in history if r.get("avg_confidence") is not None]
        avg_conf    = sum(confidences) / len(confidences) if confidences else 0.0
        flagged     = sum(1 for r in history if r.get("escalated", False))

        prompt = f"""
You are a Computer Vision Pipeline Reporting Agent.

Write a concise batch report for a data scientist based on the information below.

--- Batch Result ---
Task Type         : {result.get('type', 'unknown')}
Objects Detected  : {result.get('objects_detected', 0)}
Avg Confidence    : {result.get('avg_confidence', 0.0):.3f}
Re-inference Runs : {result.get('reinference_count', 0)}
Escalated         : {result.get('escalated', False)}
Drift Detected    : {result.get('drift_detected', False)}
Image Condition   : {result.get('condition', 'unknown')}
LLM Provider Used : {ACTIVE_PROVIDER.upper()}

--- Historical Context (last {total} batches) ---
Total Batches     : {total}
Avg Confidence    : {avg_conf:.3f}
Escalated Batches : {flagged}

--- Your Output Format ---
SUMMARY: <2-3 sentence batch summary>
ANOMALIES: <any anomalies detected, or "None">
RECOMMENDATION: <PASS | ESCALATE | RETRAIN> — <one sentence reason>
AUGMENTATION: <if drift detected, list 2-3 specific augmentation strategies; else "N/A">
"""

        response = call_llm(prompt)

        print("\n" + "=" * 50)
        print("[REPORTING AGENT] 📋 Batch Report")
        print("=" * 50)
        print(response)
        print("=" * 50)

        # Saving teh report to disk
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_type  = result.get("type", "unknown")
        filename   = f"report_{task_type}_{timestamp}.txt"
        filepath   = os.path.join(self.report_dir, filename)

        report_content = (
            f"Timestamp : {datetime.now().isoformat()}\n"
            f"Provider  : {ACTIVE_PROVIDER.upper()}\n"
            f"Task Type : {task_type}\n"
            f"{'='*50}\n"
            f"{response}\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"[Reporting] 💾 Report saved → {filepath}")
        return response