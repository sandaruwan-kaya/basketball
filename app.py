import streamlit as st
import os

from gemini_engine import analyze_frames as analyze_gemini
from gpt_engine import analyze_frames as analyze_gpt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_FRAMES_DIR = "frames"
MAX_FRAMES = 30

st.set_page_config(page_title="Shot Result Comparison", layout="centered")
st.title("🏀 Shot Result Comparison")
st.write("Gemini vs GPT Vision (same frames)")

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

# Use last frames only
frame_files = frame_files[-MAX_FRAMES:]

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
# UI
# --------------------------------------------------
with st.expander("Preview frames"):
    for f in frame_files:
        st.image(f, width=220)

if st.button("Analyze Shot"):

    with st.spinner("Running Gemini analysis..."):
        try:
            gemini_result = analyze_gemini(frame_files)
        except Exception as e:
            gemini_result = f"ERROR: {e}"

    with st.spinner("Running GPT Vision analysis..."):
        try:
            gpt_result = analyze_gpt(frame_files)
        except Exception as e:
            gpt_result = f"ERROR: {e}"

    # Normalize Gemini empty output
    if not gemini_result:
        gemini_result = "UNCLEAR"

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟦 Gemini")
        st.info(gemini_result)

    with col2:
        st.markdown("### 🟩 GPT Vision")
        st.success(gpt_result)
