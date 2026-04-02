import cv2
import os
from src.traffic_pipeline.detector import detect
from src.traffic_pipeline.tracker import SimpleTracker
from src.traffic_pipeline.density import compute_density
from src.traffic_pipeline.heatmap import generate_heatmap

def run(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = SimpleTracker()

    all_density = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        detections = detect(frame)
        tracks = tracker.update(detections)

        density = compute_density(tracks, frame.shape)

        if all_density is None:
            all_density = density
        else:
            all_density += density

    cap.release()

    if all_density is None:
        print("❌ No frames processed")
        return

    print("✅ Generating heatmap...")

    heatmap = generate_heatmap(all_density)

    # 🔥 IMPORTANT: create folder
    output_path = "outputs/heatmaps"
    os.makedirs(output_path, exist_ok=True)

    save_path = os.path.join(output_path, "traffic_heatmap.png")

    success = cv2.imwrite(save_path, heatmap)

    if success:
        print(f"✅ Heatmap saved at: {save_path}")
    else:
        print("❌ Failed to save heatmap")


if __name__ == "__main__":
    run("data/raw/traffic.mp4")