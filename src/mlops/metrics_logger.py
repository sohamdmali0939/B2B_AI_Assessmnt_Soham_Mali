import json
import os
from datetime import datetime


class MetricsLogger:
    def __init__(self, log_path="logs/metrics.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, data):
        """
        Log batch metrics with timestamp
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": data.get("type"),
            "objects_detected": data.get("objects_detected"),
            "avg_confidence": float(data.get("avg_confidence", 0))
        }

        # Append to JSON file
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(entry)

        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=4)

        print(f"[MetricsLogger] Logged batch at {entry['timestamp']}")