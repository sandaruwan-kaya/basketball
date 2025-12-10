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
You are an expert Basketball Video Analyst and Shooting Coach. Your task is to analyze the provided video to extract precise shot metrics and provide technical coaching feedback.
 
### INSTRUCTIONS FOR VIDEO PROCESSING:
1.  **Scan the entire video** to understand the camera angle and lighting.
2.  **Identify "Shot Events":** A shot event begins when the player enters the shooting motion and concludes when the result (make/miss) is confirmed.
3.  **Ignore Non-Shots:** Do not count dribbling, passing, or pump fakes. Only count if the ball actually leaves the shooter's hands with an arc towards the rim.
4.  **Verify "Makes":** A "Made Shot" requires visual confirmation of the ball passing through the hoop (e.g., net movement, ball falling below the rim). If the net is not visible, infer based on the ball's trajectory and reaction of the player/crowd, but mark it as "inferred".
 
### OUTPUT FORMAT:
You must output VALID JSON only. Do not provide introductory text or markdown formatting (like ```json).
 
### JSON SCHEMA:
{
  "shot_log": [
    {
      "id": 1,
      "timestamp_release": "mm:ss",
      "timestamp_result": "mm:ss",
      "outcome": "MADE" | "MISSED" | "UNCERTAIN",
      "visual_evidence": "Brief description of visual proof (e.g., 'Ball swished net', 'Hit back rim and out')"
    }
  ],
  "statistics": {
    "total_attempts": 0,
    "total_made": 0,
    "shooting_percentage": "0%"
  },
  "technical_analysis": {
    "stance_and_alignment": "Analysis of foot placement and body alignment.",
    "shot_mechanics": "Analysis of the set point, release point, and guide hand.",
    "arc_and_trajectory": "Observations on the loop of the ball.",
    "lower_body_power": "Usage of legs and rhythm.",
    "consistency_notes": "Is the form repeatable?"
  },
  "coaching_verdict": {
    "primary_strength": "",
    "primary_weakness": "",
    "actionable_correction": "One specific drill or focus point."
  },
  "video_limitations": "Note any occlusions, frame drops, or camera angle issues affecting analysis."
}
 
### CRITICAL RULES:
- **Accuracy over Quantity:** If a shot is completely obscured, list it in 'video_limitations' rather than guessing.
- **Consistency:** 'statistics.total_attempts' must exactly match the number of objects in 'shot_log'.
- **Timestamps:** 'timestamp_release' is when the ball leaves the fingertips.
- **Visual Evidence:** You MUST populate the 'visual_evidence' field. This is required to validate your count.
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
