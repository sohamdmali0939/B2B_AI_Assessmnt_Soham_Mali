from ultralytics import YOLO

# Loading teh  model once
model = YOLO("yolov8n.pt")


def detect(frame):
    """
    Runs YOLO detection and returns structured detections
    """

    results = model(frame)

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls = int(boxes.cls[i])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls
            })

    return detections