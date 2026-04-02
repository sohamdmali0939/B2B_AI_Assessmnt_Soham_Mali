class SimpleTracker:
    def __init__(self):
        self.id_count = 0
        self.tracks = {}

    def update(self, detections):
        """
        Assign IDs to detections (simple tracker)

        Args:
            detections: list of dicts from detector

        Returns:
            list of tracked objects
        """

        updated_tracks = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_id"]
            conf = det["confidence"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.id_count += 1

            updated_tracks.append({
                "id": self.id_count,
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "class_id": cls,
                "confidence": conf
            })

        return updated_tracks