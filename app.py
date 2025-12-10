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
  - Includes jump shots, set shots, free throws, and layups IF clearly attempted at the basket.
  - EXCLUDE:
    - Passes, lobs, or heaves that are clearly not meant to score.
    - Dribbling, fakes, and pump fakes that do NOT end with the ball leaving the hand toward the rim.
    - Motions where the ball leaves the frame and it is unclear that it was a shot.
 
- "Made shot":
  - A shot attempt where it is CLEARLY visible that the ball goes completely through the hoop.
  - If the ball’s path or hoop is obscured and you cannot CONFIRM that the ball went through:
    - DO NOT count it as made.
    - Treat it as "attempt only" IF the attempt itself is clear.
 
--------------------------------
OUTPUT JSON FORMAT (STRICT)
--------------------------------
You MUST return ONLY this JSON structure and nothing else:
 
{
  "shots_attempted": {
    "total": 0
  },
  "shots_made": {
    "total": 0
  }
}

--------------------------------
COUNTING & VALIDATION RULES
--------------------------------
You MUST obey ALL of these:
 
1. NO GUESSING:
   - Only count a shot attempt if:
     - The shooting motion is clearly visible, AND
     - The ball release toward the basket is clearly visible.
   - Only count a made shot if:
     - The ball CLEARLY goes through the hoop.
   - If you cannot clearly see the full motion or the hoop outcome:
     - Do NOT count a made shot.
     - You may still count it as an attempt ONLY if the attempt itself is clearly visible.
     - If even the attempt is unclear, do not count it at all.
 
2. CONSISTENCY CHECKS:
   - `shots_attempted.total` MUST equal `shot_attempt_events.length`.
   - `shots_made.total` MUST equal `shot_made_events.length`.
   - `shots_made.total` MUST NOT exceed `shots_attempted.total`.
 
3. VIDEO LIMITATIONS:
   - If any potential attempts or makes cannot be judged because of:
     - Poor camera angle
     - Obstructions
     - Blurry frames
     - Hoop or ball going out of frame
   - DO NOT count those as attempts or makes.
 
If the video is too short, too dark, or does not contain any valid shot attempts:
- Set totals to 0.
- Leave `shot_attempt_events` and `shot_made_events` as empty arrays.
 
--------------------------------
FINAL REQUIREMENT
--------------------------------
Return ONLY the JSON object described above.
Do NOT include explanations, notes, or any text outside the JSON.
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
