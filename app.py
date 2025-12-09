import streamlit as st
import os
import json
import datetime
from gemini_engine import gemini_analyse

st.set_page_config(page_title="Basketball AI Coach", layout="wide")

st.title("🏀 Basketball Training Video Analysis (POC)")

st.write("Upload a video and compare Gemini 2.5 Pro vs Gemini 3 Preview performance.")

# ---------------------------
# Create logs folder
# ---------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------
# Tester Name Input
# ---------------------------
tester_name = st.text_input("Tester Name")

# ---------------------------
# File Upload
# ---------------------------
video = st.file_uploader("Upload a training video:", type=["mp4", "mov", "avi"])

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):

    # Validate inputs
    if tester_name.strip() == "":
        st.error("Please enter tester name.")
        st.stop()

    if video is None:
        st.error("Please upload a video.")
        st.stop()

    video_bytes = video.read()

    with st.spinner("Analyzing using both models…"):
        results = gemini_analyse(video_bytes)

    # ---------------------------
    # Save Video + Results
    # ---------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(LOG_DIR, f"{tester_name}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    # Save video file
    video_path = os.path.join(session_folder, "video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # Save model outputs
    results_path = os.path.join(session_folder, "analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save metadata
    meta_path = os.path.join(session_folder, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"Tester: {tester_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Video File: {video_path}\n")
        f.write(f"Analysis File: {results_path}\n")

    # ---------------------------
    # UI Display (Your Layout)
    # ---------------------------
    st.success("Analysis completed and session saved!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Gemini 2.5 Pro")
        st.text(results["gemini_2_5_pro"])

    with col2:
        st.subheader("🚀 Gemini 3 Pro Preview")
        st.text(results["gemini_3_pro_preview"])

    # Show folder name
    st.write("📁 Saved to:")
    st.code(session_folder)
