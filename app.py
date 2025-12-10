import streamlit as st
import os
import json
import datetime
import base64
from gemini_engine import gemini_analyse
from gpt_engine import gpt_analyse

st.set_page_config(page_title="Basketball AI Coach", layout="wide")

st.title("🏀 Basketball Training Video Analysis (POC)")
st.write("Upload a video and compare Gemini Vision vs GPT Vision performance.")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

tester_name = st.text_input("Tester Name")

engine = st.selectbox(
    "Choose AI Engine",
    ("Gemini Vision", "GPT Vision")
)

video = st.file_uploader("Upload a training video:", type=["mp4", "mov", "avi"])

# -----------------------------------------------------
# Analyze Button
# -----------------------------------------------------
if st.button("Analyze"):
    if tester_name.strip() == "":
        st.error("Please enter tester name.")
        st.stop()

    if video is None:
        st.error("Please upload a video.")
        st.stop()

    video_bytes = video.read()

    # -----------------------------------------------------
    # Create ONE session folder for everything
    # -----------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(LOG_DIR, f"{tester_name}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    # -----------------------------------------------------
    # Run Engine
    # -----------------------------------------------------
    if engine == "Gemini Vision":
        with st.spinner("Analyzing using Gemini…"):
            results = gemini_analyse(video_bytes)
    else:
        with st.spinner("Analyzing using GPT Vision…"):
            results = gpt_analyse(video_bytes)

    # -----------------------------------------------------
    # Save uploaded video
    # -----------------------------------------------------
    video_path = os.path.join(session_folder, "video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # -----------------------------------------------------
    # Save extracted frames (GPT only)
    # -----------------------------------------------------
    if engine != "Gemini Vision" and "frames" in results:
        frames_folder = os.path.join(session_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)

        for i, fdata in enumerate(results["frames"]):
            frame_bytes = base64.b64decode(fdata["b64"])
            frame_filename = f"frame_{i+1:02d}_{fdata['timestamp'].replace(':','-')}.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)

            with open(frame_path, "wb") as img_file:
                img_file.write(frame_bytes)

    # -----------------------------------------------------
    # Save raw outputs
    # -----------------------------------------------------
    with open(os.path.join(session_folder, "raw_g25.txt"), "w", encoding="utf-8") as f:
        f.write(results["gemini_2_5_pro"]["raw"])

    with open(os.path.join(session_folder, "raw_g30.txt"), "w", encoding="utf-8") as f:
        f.write(results["gemini_3_pro_preview"]["raw"])

    # -----------------------------------------------------
    # Save parsed JSON
    # -----------------------------------------------------
    with open(os.path.join(session_folder, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------
    with open(os.path.join(session_folder, "meta.txt"), "w") as f:
        f.write(f"Tester: {tester_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Video File: {video_path}\n")
        if engine != "Gemini Vision":
            f.write(f"Frames Saved: {len(results['frames'])}\n")

    # -----------------------------------------------------
    # Extract parsed results
    # -----------------------------------------------------
    g25 = results["gemini_2_5_pro"]["parsed"]
    g30 = results["gemini_3_pro_preview"]["parsed"]

    # -----------------------------------------------------
    # UI Display
    # -----------------------------------------------------
    st.success(f"Analysis completed and saved to: {session_folder}")

    col1, col2 = st.columns(2)

    # ---------------------------
    # LEFT PANEL — Gemini 2.5 Pro
    # ---------------------------
    with col1:
        st.subheader("🤖 Gemini 2.5 Pro")

        if g25 is None:
            st.error("Model returned unreadable output.")
        else:
            st.metric("Shots Attempted", g25["shots_attempted"]["total"])
            st.metric("Shots Made", g25["shots_made"]["total"])

            st.write("### Shot Attempt Timestamps")
            for ev in g25["shot_attempt_events"]:
                st.write(f"• {ev['timestamp']}")

            st.write("### Made Shot Timestamps")
            for ev in g25["shot_made_events"]:
                st.write(f"• {ev['timestamp']}")

            st.write("### 📝 Coaching Feedback")

            cf = g25["coaching_feedback"]

            st.markdown("#### 🟩 Summary")
            st.write(cf["summary"])

            st.markdown("#### 🟦 Stance & Balance")
            st.write(cf["stance_and_balance"])

            st.markdown("#### 🟧 Footwork")
            st.write(cf["footwork"])

            st.markdown("#### 🟨 Ball Gather & Set Point")
            st.write(cf["ball_gather_and_set_point"])

            st.markdown("#### 🟥 Release & Follow-through")
            st.write(cf["release_and_follow_through"])

            st.markdown("#### 🟪 Timing & Rhythm")
            st.write(cf["timing_and_rhythm"])

            st.markdown("#### 🟫 Shot Arc & Power")
            st.write(cf["shot_arc_and_power"])

            st.markdown("#### 🟩 Consistency")
            st.write(cf["consistency"])

            st.markdown("#### 🟦 Shot Selection")
            st.write(cf["shot_selection"])

            st.markdown("#### ⭐ Areas to Improve")
            for item in cf["areas_to_improve"]:
                st.write(f"• {item}")

            if cf["limitations_in_video"]:
                st.markdown("#### ⚠️ Limitations in Video")
                for item in cf["limitations_in_video"]:
                    st.write(f"• {item}")

    # ---------------------------
    # RIGHT PANEL — Gemini 3 Pro Preview
    # ---------------------------
    with col2:
        st.subheader("🚀 Gemini 3 Pro Preview")

        if g30 is None:
            st.error("Model returned unreadable output.")
        else:
            st.metric("Shots Attempted", g30["shots_attempted"]["total"])
            st.metric("Shots Made", g30["shots_made"]["total"])

            st.write("### Shot Attempt Timestamps")
            for ev in g30["shot_attempt_events"]:
                st.write(f"• {ev['timestamp']}")

            st.write("### Made Shot Timestamps")
            for ev in g30["shot_made_events"]:
                st.write(f"• {ev['timestamp']}")

            st.write("### 📝 Coaching Feedback")

            cf = g30["coaching_feedback"]

            st.markdown("#### 🟩 Summary")
            st.write(cf["summary"])

            st.markdown("#### 🟦 Stance & Balance")
            st.write(cf["stance_and_balance"])

            st.markdown("#### 🟧 Footwork")
            st.write(cf["footwork"])

            st.markdown("#### 🟨 Ball Gather & Set Point")
            st.write(cf["ball_gather_and_set_point"])

            st.markdown("#### 🟥 Release & Follow-through")
            st.write(cf["release_and_follow_through"])

            st.markdown("#### 🟪 Timing & Rhythm")
            st.write(cf["timing_and_rhythm"])

            st.markdown("#### 🟫 Shot Arc & Power")
            st.write(cf["shot_arc_and_power"])

            st.markdown("#### 🟩 Consistency")
            st.write(cf["consistency"])

            st.markdown("#### 🟦 Shot Selection")
            st.write(cf["shot_selection"])

            st.markdown("#### ⭐ Areas to Improve")
            for item in cf["areas_to_improve"]:
                st.write(f"• {item}")

            if cf["limitations_in_video"]:
                st.markdown("#### ⚠️ Limitations in Video")
                for item in cf["limitations_in_video"]:
                    st.write(f"• {item}")

    st.write("📁 Saved to:")
    st.code(session_folder)
