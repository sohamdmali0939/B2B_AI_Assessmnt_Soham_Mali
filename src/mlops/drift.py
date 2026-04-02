class DriftDetector:
    def detect(self, history):
        if len(history) < 3:
            return False

        recent = [h["confidence"] for h in history[-3:]]
        avg_recent = sum(recent) / len(recent)

        overall = [h["confidence"] for h in history]
        avg_overall = sum(overall) / len(overall)

        if avg_recent < 0.7 * avg_overall:
            return True

        return False