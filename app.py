# app.py
import streamlit as st
import os
import json
import datetime
from gemini_engine import compare_models

# -------------------------
# Hard-coded prompt here
# -------------------------
PROMPT = """
You are an expert basketball shooting coach AND a precise video analyst.
Your job is to (1) COUNT SHOTS.

You MUST follow the instructions below exactly and ONLY return the JSON object in the specified format.

--------------------------------
TASK
--------------------------------
From the provided basketball shooting video:

1. Identify every DISTINCT shot attempt.
2. Identify which of those attempts are MADE shots.

--------------------------------
DEFINITIONS
--------------------------------
- "Shot attempt":
  - A deliberate shooting motion TOWARD the basket where:
    - The player gathers the ball,
    - Moves through a shooting motion, and
    - RELEASES the ball from their hands TOWARD the hoop.
  - EXCLUDE passes, lobs, fakes, dribbles, unclear releases.

- "Made shot":
  - Only if the ball CLEARLY goes through the hoop.
  - If unclear → count as attempt only.

--------------------------------
STRICT OUTPUT FORMAT
--------------------------------
{
  "shots_attempted": { "total": 0 },
  "shots_made": { "total": 0 },
  "shot_attempt_events": [{ "timestamp": "" }],
  "shot_made_events": [{ "timestamp": "" }],
}

--------------------------------
RULES
--------------------------------
- No guessing.
- shots_made.total <= shots_attempted.total
- If unclear → do not count.
"""

# - "Made shot":
#   - Only if the ball CLEARLY goes through the hoop.
#   - This includes clean swishes, rim-ins, bank shots, and rattled shots.
#   - If unclear → count as attempt only.

  # "shot_attempt_events": [{ "timestamp": "" }],
  # "shot_made_events": [{ "timestamp": "" }],

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

st.title("⚡ Gemini Video Test (Minimal Mode)")
st.write("Upload a video and compare Gemini 2.5 Pro vs Gemini 3 Preview outputs.")

video = st.file_uploader("Upload a training video:", type=["mp4", "mov", "avi"])

if st.button("Run Analysis"):

    if video is None:
        st.error("Please upload a video first.")
        st.stop()

    video_bytes = video.read()

    st.info("Running models… please wait.")

    # Run both Gemini models
    results = compare_models(PROMPT, video_bytes)

    # st.subheader("Live Logs")
    # st.text("\n".join(results["gemini_2_5_pro"]["log"]))
    # st.text("\n".join(results["gemini_3_pro_preview"]["log"]))


    # -------------------------
    # Create log folder
    # -------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(LOG_DIR, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # Save raw outputs
    with open(os.path.join(session_dir, "raw_g25.txt"), "w", encoding="utf-8") as f:
        f.write(results["gemini_2_5_pro"]["raw"])

    with open(os.path.join(session_dir, "raw_g30.txt"), "w", encoding="utf-8") as f:
        f.write(results["gemini_3_pro_preview"]["raw"])

    with open(os.path.join(session_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

      # Save metadata
    with open(os.path.join(session_dir, "meta.txt"), "w") as f:
        # f.write(f"Tester: {tester_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Original Filename: {video.name}\n")
        # f.write(f"Video File: {video_path}\n")

    st.success("Completed! Results logged.")

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------

    st.subheader("🐢 Gemini 2.5 Pro Output")
    r = results["gemini_2_5_pro"]
    st.write(f"Latency: **{r['latency']} sec**")
    st.write(f"Tokens (in/out/total): {r['input_tokens']}/{r['output_tokens']}/{r['total_tokens']}")
    st.text_area("Raw Output", r["raw"], height=300, key="raw_output_g25")


    st.subheader("🚀 Gemini 3 Pro Preview Output")
    r = results["gemini_3_pro_preview"]
    st.write(f"Latency: **{r['latency']} sec**")
    st.write(f"Tokens (in/out/total): {r['input_tokens']}/{r['output_tokens']}/{r['total_tokens']}")
    st.text_area("Raw Output", r["raw"], height=300, key="raw_output_g30")


    st.code(f"Logs saved at: {session_dir}")
