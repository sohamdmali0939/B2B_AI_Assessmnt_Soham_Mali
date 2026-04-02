import os
import cv2

from src.traffic_pipeline.detector import detect
from src.traffic_pipeline.tracker import SimpleTracker
from src.traffic_pipeline.density import compute_density
from src.traffic_pipeline.heatmap import generate_heatmap


def fallback_traffic_logic(detections):
    """
    Fallback logic if Gemini API fails (optional placeholder)
    Simply returns basic stats instead of calling API.
    """
    return {
        "type": "traffic",
        "frames_processed": len(detections),
        "objects_detected": sum(len(frame) for frame in detections),
        "avg_confidence": (
            sum(det.get("confidence", 0) for frame in detections for det in frame) /
            max(1, sum(len(frame) for frame in detections))
        )
    }


def traffic_tool(input_data):
    """
    Traffic pipeline tool for agent
    """

    video_path = input_data.get("video_path", "data/raw/traffic.mp4")

    # --- Step 1: Safe video check ---
    if not os.path.isfile(video_path):
        raise ValueError(f"❌ Cannot open video: {video_path}. File does not exist!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {video_path}. Check format or path.")

    tracker = SimpleTracker()

    total_conf = []
    total_detections = 0
    frame_count = 0
    all_detections = []  # For fallback logic if API fails

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔹 DETECTION
        detections = detect(frame)
        all_detections.append(detections)  # store for fallback

        # ✅ DEBUG
        print(f"[Frame {frame_count}] Detections: {len(detections)}")

        # 🔹 TRACKING
        tracks = tracker.update(detections)

        # 🔹 COLLECT CONFIDENCE
        for det in detections:
            conf = det.get("confidence", None)
            if conf is not None:
                total_conf.append(float(conf))

        total_detections += len(detections)

        # 🔹 DENSITY + HEATMAP
        density_map = compute_density(tracks, frame.shape)
        heatmap = generate_heatmap(density_map)

        # Save first heatmap frame for proof
        if frame_count == 1:
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite("outputs/traffic_heatmap.png", heatmap)

    cap.release()

    
    if len(total_conf) > 0:
        avg_conf = sum(total_conf) / len(total_conf)
    else:
        avg_conf = 0.0

    result = {
        "type": "traffic",
        "frames_processed": frame_count,
        "objects_detected": total_detections,
        "avg_confidence": avg_conf
    }

    
    # try:
    #     # response = call_gemini_api(result)  
    #     pass
    # except Exception as e:
    #     if "RESOURCE_EXHAUSTED" in str(e):
    #         print("⚠️ Gemini quota exceeded. Using fallback logic.")
    #         result = fallback_traffic_logic(all_detections)
    #     else:
    #         raise e

    return result