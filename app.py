import streamlit as st
import os
from gemini_engine import analyze_frames

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_FRAMES_DIR = "frames"
MAX_FRAMES = 30  # decision frames only (important)

st.set_page_config(page_title="Gemini Shot Test", layout="centered")
st.title("🏀 Gemini Shot Result Test")
st.write("Frames only • One shot attempt • MADE / MISSED / UNCLEAR")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
video_no = st.text_input("Video number", value="10")
attempt_no = st.text_input("Attempt number", value="1")

FRAMES_DIR = os.path.join(BASE_FRAMES_DIR, video_no, attempt_no)

# --------------------------------------------------
# LOAD FRAMES
# --------------------------------------------------
if not os.path.exists(FRAMES_DIR):
    st.error(f"Frames folder not found: {FRAMES_DIR}")
    st.stop()

frame_files = sorted([
    os.path.join(FRAMES_DIR, f)
    for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Take LAST frames only (most signal)
if len(frame_files) > MAX_FRAMES:
    frame_files = frame_files[-MAX_FRAMES:]

# --------------------------------------------------
# TERMINAL LOGS
# --------------------------------------------------
print("=" * 60)
print(f"📂 Using frames directory: {FRAMES_DIR}")
print(f"🧮 Frames used: {len(frame_files)}")
for f in frame_files:
    print("   -", f)
print("=" * 60)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.write(f"Frames selected: **{len(frame_files)}**")

with st.expander("Preview frames"):
    for f in frame_files:
        st.image(f, width=220)

if st.button("Analyze Shot"):
    with st.spinner("Sending frames to Gemini..."):
        result = analyze_frames(frame_files)

    st.subheader("Result")

    if result == "MADE":
        st.success("🏀 MADE")
    elif result == "MISSED":
        st.error("❌ MISSED")
    else:
        st.warning("⚠️ UNCLEAR")
