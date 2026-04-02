from src.crop_pipeline.inference import run_inference
import cv2


def crop_tool(input_data):
    """
    Crop pipeline tool for agent
    """

    image_path = input_data.get("image_path", "data/raw/sample.png")

    results = run_inference(image_path)

    confidences = []

    for r in results:
        confidences.append(r["confidence"])

    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return {
        "type": "crop",
        "objects_detected": len(results),
        "avg_confidence": avg_conf
    }