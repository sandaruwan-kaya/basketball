# video_frame_utils.py

import cv2
import os

# -----------------------------------------------------
# Convert timestamp formats to seconds
# "3.2"  → 3.2
# "00:03.2" → 3.2
# -----------------------------------------------------
def timestamp_to_seconds(ts: str) -> float:
    ts = ts.strip()

    if ":" not in ts:
        return float(ts)

    parts = ts.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)

    return 0.0


# -----------------------------------------------------
# Extract a single frame from video at a timestamp
# -----------------------------------------------------
def extract_frame(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return False

    frame_num = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    success, frame = cap.read()
    if not success:
        cap.release()
        return False

    cv2.imwrite(output_path, frame)
    cap.release()
    return True


# -----------------------------------------------------
# Extract multiple frames from event list
# events = [{ "timestamp": "03.2" }, ...]
# prefix = "g25_attempt"
# -----------------------------------------------------
def extract_images_from_events(video_path: str, events: list, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, ev in enumerate(events):
        ts = ev["timestamp"]
        sec = timestamp_to_seconds(ts)
        out_path = os.path.join(output_dir, f"{prefix}_{idx+1}.jpg")

        if extract_frame(video_path, sec, out_path):
            results.append((ts, out_path))

    return results
