import os
import time
from gemini_engine import gemini_analyse
# from shot_detector_yolo import detect_made_shots # Uncomment if you use YOLO

# --- Configuration ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

VIDEO_PATH = "samples/10.MOV"

# 🌟 NEW CONFIGURATION VARIABLE
NUM_RUNS = 1 
# Set this to any integer (e.g., 1 for a quick test, 10 for detailed consistency)

PROMPT = (
    "Analyze the provided video segment and accurately count the total number of "
    "basketball shots attempted and the total number of shots made. "
    "Only count fully visible shots where the outcome (make or miss) can be determined. "
    "Do not include free throws unless explicitly visible. "
    "Return the counts in the required JSON schema."
)

with open(VIDEO_PATH, "rb") as f:
    video_bytes = f.read()

# -----------------------------------------------------
# Run Gemini
# -----------------------------------------------------
print("PROMPT: ", PROMPT)
print(f"Executing Gemini Analysis for {NUM_RUNS} run(s)...")

merged, session_dir = gemini_analyse(
    prompt=PROMPT,
    video_bytes=video_bytes,
    log_dir=LOG_DIR,
    num_runs=NUM_RUNS  # 🌟 PASS THE FLAG HERE
)

print("\nGemini processing done. Logs at:", session_dir)
print("\nFinal Consistency Array Output:")


# -----------------------------------------------------
# Run YOLO made-shot detector (Optional)
# -----------------------------------------------------
# made_log_dir = os.path.join(session_dir, "yolo")
# os.makedirs(made_log_dir, exist_ok=True)

# made_shots = detect_made_shots(VIDEO_PATH, made_log_dir)

# print("\nMade shots detected at timestamps (sec):")
# print(made_shots)