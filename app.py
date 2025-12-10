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
{
  "role": "EXPERT_BASKETBALL_ANALYST",
  "task": "PRECISION_SHOOTING_METRICS_BY_INTERVAL",
  "objective": "Analyze the video in 5-second distinct intervals. Count shot attempts and made shots for each interval, then aggregate for the final total.",
 
  "PROCESSING_RULES": {
    "segmentation_strategy": "Divide the video into rigid 5-second chunks (00:00-00:05, 00:05-00:10, etc.).",
    "attribution_rule": "A shot belongs to the interval where the BALL RELEASE occurs. If a player releases at 00:04 but the ball goes in at 00:06, count both the attempt and the make in the 00:00-00:05 interval.",
    "uncertainty_rule": "If visibility is obstructed or if the ball release is not perfectly clear, DO NOT COUNT."
  },
 
  "DEFINITIONS": {
    "SHOT_ATTEMPT": {
      "definition": "A deliberate shooting motion where the ball is RELEASED.",
      "action": "Count as 1 attempt in the current time interval."
    },
    "MADE_SHOT": {
      "definition": "A shot attempt that successfully goes through the hoop.",
      "visual_cues": ["Ball clears the rim and passes through the net."],
      "action": "Count as 1 make in the current time interval."
    }
  },
 
  "OUTPUT_FORMAT": {
    "type": "STRICT_JSON",
    "schema": {
      "intervals": [
        {
          "time_range": "00:00 - 00:05",
          "attempts": 0,
          "makes": 0
        },
        {
          "time_range": "00:05 - 00:10",
          "attempts": 0,
          "makes": 0
        }
        // Continue for the duration of the video
      ],
      "final_stats": {
        "total_attempts": 0,
        "total_makes": 0
      }
    }
  },
 
  "VALIDATION_RULES": [
    "1. 'final_stats.total_attempts' MUST be the exact sum of 'attempts' in all intervals.",
    "2. 'final_stats.total_makes' MUST be the exact sum of 'makes' in all intervals.",
    "3. Ensure every second of the video is covered by an interval.",
    "4. Do NOT include conversational text, markdown, or explanations. ONLY return the JSON."
  ]
}
"""

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

st.title("⚡ Gemini Video Test (Minimal Mode)")
st.write("Upload a video and compare Gemini 2.5 Pro vs Gemini 3 Preview outputs.")

video = st.file_uploader("Upload MP4 video", type=["mp4"])

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

    st.success("Completed! Results logged.")

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------

    st.subheader("🐢 Gemini 2.5 Pro Output")
    r = results["gemini_2_5_pro"]
    st.write(f"Latency: **{r['latency']} sec**")
    st.write(f"Tokens (in/out/total): {r['input_tokens']}/{r['output_tokens']}/{r['total_tokens']}")
    st.text_area("Raw Output", r["raw"], height=300)

    st.subheader("🚀 Gemini 3 Pro Preview Output")
    r = results["gemini_3_pro_preview"]
    st.write(f"Latency: **{r['latency']} sec**")
    st.write(f"Tokens (in/out/total): {r['input_tokens']}/{r['output_tokens']}/{r['total_tokens']}")
    st.text_area("Raw Output", r["raw"], height=300)

    st.code(f"Logs saved at: {session_dir}")
