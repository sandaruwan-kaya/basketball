import streamlit as st
import os

from gemini_engine import analyze_frames_parallel as analyze_gemini_parallel
from gpt_engine import analyze_frames_parallel as analyze_gpt_parallel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_FRAMES_DIR = "frames"
MAX_FRAMES = 30

st.set_page_config(page_title="Shot Result Comparison", layout="centered")
st.title("🏀 Shot Result Comparison")
st.write("Gemini (new SDK) vs GPT Vision ×5 (same frames)")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
video_no = st.text_input("Video number", value="10")
attempt_no = st.text_input("Attempt number", value="1")

FRAMES_DIR = os.path.join(BASE_FRAMES_DIR, video_no, attempt_no)

if not os.path.exists(FRAMES_DIR):
    st.error(f"Frames folder not found: {FRAMES_DIR}")
    st.stop()

frame_files = sorted([
    os.path.join(FRAMES_DIR, f)
    for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Use last frames only (decision-heavy frames)
frame_files = frame_files[-MAX_FRAMES:]

if len(frame_files) == 0:
    st.error("No frames found in directory.")
    st.stop()

# --------------------------------------------------
# TERMINAL LOGS
# --------------------------------------------------
print("=" * 60)
print(f"📂 Frames directory: {FRAMES_DIR}")
print(f"🧮 Frames used: {len(frame_files)}")
for f in frame_files:
    print("   -", f)
print("=" * 60)

# --------------------------------------------------
# UI – PREVIEW
# --------------------------------------------------
with st.expander("Preview frames"):
    for f in frame_files:
        st.image(f, width=220)

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------
if st.button("Analyze Shot (Gemini ×5 + GPT ×5)"):

    with st.spinner("Running Gemini (5 parallel runs)..."):
        gemini_results = analyze_gemini_parallel(frame_files, num_runs=5)

    # with st.spinner("Running GPT Vision (5 parallel runs)..."):
    #     gpt_results = analyze_gpt_parallel(frame_files, num_runs=5)

    st.subheader("Results")

    col1, col2 = st.columns(2)

    # -------- Gemini --------
    with col1:
        st.markdown("## 🟦 Gemini ×5")
        for r in gemini_results:
            st.markdown(f"**Run {r['run']}**")
            st.text(r["raw"])

    # -------- GPT --------
    # with col2:
    #     st.markdown("## 🟩 GPT Vision ×5")
    #     for r in gpt_results:
    #         st.markdown(f"**Run {r['run']}**")
    #         st.text(r["raw"])
