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
  "shots_made": { "total": 0 }
}

--------------------------------
RULES
--------------------------------
- No guessing.
- shots_made.total <= shots_attempted.total
- If unclear → do not count.
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

    # --------------------------------------
    # RUN BOTH MODELS (now returns list per model)
    # --------------------------------------
    results = compare_models(PROMPT, video_bytes)

    # --------------------------------------
    # COMBINE CHUNK OUTPUTS FOR EACH MODEL
    # --------------------------------------
    def combine_raw(model_chunks):
        """Turn list of chunk outputs into a single readable string."""
        parts = []
        for idx, chunk in enumerate(model_chunks):
            raw = chunk.get("raw", "")
            parts.append(f"----- CHUNK {idx+1} -----\n{raw}")
        return "\n\n".join(parts)

    combined_25 = combine_raw(results["gemini_2_5_pro"])
    combined_30 = combine_raw(results["gemini_3_pro_preview"])

    # --------------------------------------
    # MAKE SESSION FOLDER
    # --------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(LOG_DIR, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # --------------------------------------
    # SAVE RAW COMBINED OUTPUTS
    # --------------------------------------
    with open(os.path.join(session_dir, "raw_g25.txt"), "w", encoding="utf-8") as f:
        f.write(combined_25)

    with open(os.path.join(session_dir, "raw_g30.txt"), "w", encoding="utf-8") as f:
        f.write(combined_30)

    with open(os.path.join(session_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    st.success("Completed! Results logged.")

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    st.subheader("🐢 Gemini 2.5 Pro Output (All Chunks Combined)")
    # Latency is now sum of chunk latencies? Or last chunk?
    # For now, show the final chunk latency:
    last_chunk_25 = results["gemini_2_5_pro"][-1]
    st.write(f"Final Chunk Latency: **{last_chunk_25.get('latency', 'N/A')} sec**")

    st.text_area("Raw Output (Combined)", combined_25, height=300)

    st.subheader("🚀 Gemini 3 Pro Preview Output (All Chunks Combined)")
    last_chunk_30 = results["gemini_3_pro_preview"][-1]
    st.write(f"Final Chunk Latency: **{last_chunk_30.get('latency', 'N/A')} sec**")

    st.text_area("Raw Output (Combined)", combined_30, height=300)

    st.code(f"Logs saved at: {session_dir}")
