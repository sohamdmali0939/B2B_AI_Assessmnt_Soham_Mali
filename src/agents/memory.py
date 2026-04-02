import time

class Memory:
    def __init__(self):
        self.history = []

    def store(self, result):
        entry = {
            "timestamp": time.time(),
            "type": result["type"],
            "confidence": result["avg_confidence"],
            "objects": result["objects_detected"]
        }
        self.history.append(entry)

    def get_recent(self, n=5):
        return self.history[-n:]

    def get_all(self):
        return self.history