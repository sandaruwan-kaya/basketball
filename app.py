import os
import time
from gemini_engine import gemini_analyse
# from shot_detector_yolo import detect_made_shots # Uncomment if you use YOLO

# --- Configuration ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

VIDEO_PATH = "samples/10.MOV"

# 🌟 NEW CONFIGURATION VARIABLE
NUM_RUNS = 5  # Increase for better consistency (optional)

# 🔥 IMPORTANT PROMPT UPDATE (requires timestamps)
PROMPT = (
    "You are an expert basketball video analyst. "
    "Analyze the provided video segment and identify:\n"
    "Analyze the provided video segment and accurately count the total number of "
    "basketball shots attempted and the total number of shots made. "
    "Only count fully visible shots where the outcome (make or miss) can be determined. "
    "Do not include free throws unless explicitly visible. "
    "1. Total shots attempted\n"
    "2. Total shots made\n"
    "3. Timestamps OF EACH shot attempt\n"
    "4. Timestamps OF EACH made shot\n\n"
    "Return timestamps RELATIVE TO THIS CHUNK ONLY.\n\n"
    "STRICT JSON SCHEMA:\n"
    "{\n"
    '  "shots_attempted": 0,\n'
    '  "shots_made": 0,\n'
    '  "shot_attempt_events": ["0.0", "1.2"],\n'
    '  "shot_made_events": ["1.2"]\n'
    "}\n"
)

with open(VIDEO_PATH, "rb") as f:
    video_bytes = f.read()

print("PROMPT: ", PROMPT)
print(f"Executing Gemini Analysis for {NUM_RUNS} run(s)...")

merged, session_dir = gemini_analyse(
    prompt=PROMPT,
    video_bytes=video_bytes,
    log_dir=LOG_DIR,
    num_runs=NUM_RUNS
)

print("\nGemini processing done. Logs at:", session_dir)
print("\nFinal Consistency Array Output:")
print(merged)
