import os
from dotenv import load_dotenv

load_dotenv()

from src.agents.qc_agent import QCAgent
from src.agents.reporting_agent import ReportingAgent
from src.agents.memory import Memory
from src.mlops.drift import DriftDetector
from src.agents.llm_utils import call_llm, ACTIVE_PROVIDER, PROVIDERS
from src.mlops.metrics_logger import MetricsLogger


# =========================
# 🔹 ORCHESTRATOR AGENT
# =========================
class OrchestratorAgent:
    def __init__(self, crop_tool, traffic_tool):
        self.crop_tool    = crop_tool
        self.traffic_tool = traffic_tool

        self.qc       = QCAgent()
        self.reporter = ReportingAgent()
        self.memory   = Memory()
        self.drift    = DriftDetector()
        self.logger   = MetricsLogger()   # ✅ ADDED

        cfg = PROVIDERS.get(ACTIVE_PROVIDER, {})
        print(f"[Orchestrator] 🤖 LLM Provider : {ACTIVE_PROVIDER.upper()}")
        print(f"[Orchestrator] 🧠 Model        : {cfg.get('model', 'unknown')}")
        print(f"[Orchestrator] 📊 Limits       : {cfg.get('limit', 'unknown')}")

    # =========================
    # 🔀 ROUTING (LLM + FALLBACK)
    # =========================
    def route(self, input_data):
        prompt = f"""
You are an AI orchestrator for a computer vision pipeline.

Decide which pipeline to use based on the input data below.

Input:
{input_data}

Rules:
- If the input contains a video file or traffic-related data, answer: traffic
- If the input contains an image or crop/agriculture data, answer: crop

Answer ONLY one word: crop OR traffic
"""
        try:
            decision = call_llm(prompt).strip().lower()
            print(f"[Routing] LLM decision: '{decision}'")

            if "traffic" in decision:
                return "traffic"
            elif "crop" in decision:
                return "crop"
            else:
                raise ValueError(f"Unclear LLM response: '{decision}'")

        except Exception as e:
            print(f"⚠️ Routing LLM failed, using key-based fallback: {e}")

            if "video_path" in input_data or input_data.get("type") == "traffic":
                return "traffic"
            return "crop"

    # =========================
    # 🧠 MAIN EXECUTION LOOP
    # =========================
    def run(self, input_data):
        print("\n==============================")
        print("[Orchestrator] 🚀 New Batch")
        print("==============================")

        # 🔀 Step 1: Routing
        route = self.route(input_data)

        # 🔧 Step 2: Select correct tool
        tool = self.crop_tool if route == "crop" else self.traffic_tool
        print(f"[Agent] ✅ Selected pipeline: {route}")

        # ▶️ Step 3: Run inference
        print(f"[Agent] ▶️ Running inference...")
        result = tool(input_data)

        # 📊 Step 4: QC Evaluation
        print(f"[Agent] 📊 Running QC evaluation...")
        qc_result = self.qc.evaluate(result)

        # 🔁 Step 5: Self-Optimization Loop
        reinference_count = 0
        MAX_REINFERENCE = 2

        while qc_result.get("status") == "LOW_CONFIDENCE" and reinference_count < MAX_REINFERENCE:
            reinference_count += 1
            print(f"[Agent] 🔁 Re-running inference (attempt {reinference_count}/{MAX_REINFERENCE})")
            result = tool(input_data)
            qc_result = self.qc.evaluate(result)

        if reinference_count > 0:
            if qc_result.get("status") == "LOW_CONFIDENCE":
                print(f"[Agent] ⚠️ Still low confidence after {reinference_count} retries → Escalating")
                result["escalated"] = True
            else:
                print(f"[Agent] ✅ Confidence improved after retry")

        result["reinference_count"] = reinference_count

        # 💾 Step 6: Store in memory
        print(f"[Agent] 💾 Storing result in memory...")
        self.memory.store(result)

        # 📊 Step 6.5: LOG METRICS (NEW)
        print(f"[Agent] 📊 Logging metrics...")
        self.logger.log(result)

        # 📉 Step 7: Drift Detection
        history = self.memory.get_all()
        drift_detected = self.drift.detect(history)

        if drift_detected:
            print("\n🚨 [Drift] ⚠️ Performance drift detected!")
            print("[Drift] 🔁 Retraining recommended")

        result["drift_detected"] = drift_detected

        # 📊 Step 8: Reporting
        print(f"[Agent] 📝 Generating report...")
        self.reporter.generate(result, self.memory)

        # ✅ Step 9: Final Output
        print("\n[Agent] ✅ Batch completed successfully")

        return result