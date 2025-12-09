import streamlit as st
import os
import json
import datetime
from gemini_engine import gemini_analyse
from video_frame_utils import (
    extract_images_from_events,
    timestamp_to_seconds,
    extract_frame
)


st.set_page_config(page_title="Basketball AI Coach", layout="wide")

st.title("🏀 Basketball Training Video Analysis (POC)")
st.write("Upload a video and compare Gemini 2.5 Pro vs Gemini 3 Preview performance.")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

tester_name = st.text_input("Tester Name")
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

    with st.spinner("Analyzing using both models…"):
        results = gemini_analyse(video_bytes)

    # -----------------------------------------------------
    # Extract parsed results
    # -----------------------------------------------------
    g25 = results["gemini_2_5_pro"]["parsed"]
    g30 = results["gemini_3_pro_preview"]["parsed"]

    # ---------------------------
    # Create session folder
    # ---------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(LOG_DIR, f"{tester_name}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(session_folder, "video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # -----------------------------------------------------
    # Extract image snapshots from shot events
    # -----------------------------------------------------
    frame_dir = os.path.join(session_folder, "frames")

    g25_attempt_frames = extract_images_from_events(
        video_path,
        g25["shot_attempt_events"],
        frame_dir,
        "g25_attempt"
    )

    g25_made_frames = extract_images_from_events(
        video_path,
        g25["shot_made_events"],
        frame_dir,
        "g25_made"
    )

    g30_attempt_frames = extract_images_from_events(
        video_path,
        g30["shot_attempt_events"],
        frame_dir,
        "g30_attempt"
    )

    g30_made_frames = extract_images_from_events(
        video_path,
        g30["shot_made_events"],
        frame_dir,
        "g30_made"
    )


    # Save raw outputs
    with open(os.path.join(session_folder, "raw_g25.txt"), "w", encoding="utf-8") as f:
        f.write(results["gemini_2_5_pro"]["raw"])

    with open(os.path.join(session_folder, "raw_g30.txt"), "w", encoding="utf-8") as f:
        f.write(results["gemini_3_pro_preview"]["raw"])

    # Save parsed JSON
    with open(os.path.join(session_folder, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Save metadata
    with open(os.path.join(session_folder, "meta.txt"), "w") as f:
        f.write(f"Tester: {tester_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Video File: {video_path}\n")

    # # -----------------------------------------------------
    # # Extract parsed results
    # # -----------------------------------------------------
    # g25 = results["gemini_2_5_pro"]["parsed"]
    # g30 = results["gemini_3_pro_preview"]["parsed"]

    # ---------------------------
    # UI Display – POC Clean Mode
    # ---------------------------
    st.success("Analysis completed and session saved!")

    col1, col2 = st.columns(2)

    # ---------------------------
    # LEFT PANEL — Gemini 2.5 Pro
    # ---------------------------
    with col1:
        st.subheader("🤖 Gemini 2.5 Pro")

        metrics25 = results["gemini_2_5_pro"]

        st.markdown("### ⚡ Performance Metrics")
        st.write(f"**Latency:** {metrics25['latency_sec']} sec")
        st.write(f"**Input Tokens:** {metrics25['tokens']['input']}")
        st.write(f"**Output Tokens:** {metrics25['tokens']['output']}")
        st.write(f"**Total Tokens:** {metrics25['tokens']['total']}")

        if g25 is None:
            st.error("Model returned unreadable output.")
        else:
            attempted = g25["shots_attempted"]["total"]
            made = g25["shots_made"]["total"]

            st.metric("Shots Attempted", attempted)
            st.metric("Shots Made", made)

            # st.write("### Shot Attempt Timestamps")
            # for ev in g25["shot_attempt_events"]:
            #     st.write(f"• {ev['timestamp']}")

            # st.write("### Made Shot Timestamps")
            # for ev in g25["shot_made_events"]:
            #     st.write(f"• {ev['timestamp']}")

            st.write("### 📸 Attempt Snapshots")
            for ts, path in g25_attempt_frames:
                st.image(path, caption=f"Attempt @ {ts}", use_column_width=True)

            st.write("### 🏀 Made Shot Snapshots")
            for ts, path in g25_made_frames:
                st.image(path, caption=f"Made Shot @ {ts}", use_column_width=True)


            st.write("### 📝 Coaching Feedback")

            cf = g25["coaching_feedback"]   # for Gemini 2.5 Pro
            # cf = g30["coaching_feedback"] # for Gemini 3 Pro Preview

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

            # Optional: Show video limitations
            if cf["limitations_in_video"]:
                st.markdown("#### ⚠️ Limitations in Video")
                for item in cf["limitations_in_video"]:
                    st.write(f"• {item}")


    # ---------------------------
    # RIGHT PANEL — Gemini 3 Pro Preview
    # ---------------------------
    with col2:
        st.subheader("🚀 Gemini 3 Pro Preview")

        metrics30 = results["gemini_3_pro_preview"]

        st.markdown("### ⚡ Performance Metrics")
        st.write(f"**Latency:** {metrics30['latency_sec']} sec")
        st.write(f"**Input Tokens:** {metrics30['tokens']['input']}")
        st.write(f"**Output Tokens:** {metrics30['tokens']['output']}")
        st.write(f"**Total Tokens:** {metrics30['tokens']['total']}")

        if g30 is None:
            st.error("Model returned unreadable output.")
        else:
            attempted = g30["shots_attempted"]["total"]
            made = g30["shots_made"]["total"]

            st.metric("Shots Attempted", attempted)
            st.metric("Shots Made", made)

            # st.write("### Shot Attempt Timestamps")
            # for ev in g30["shot_attempt_events"]:
            #     st.write(f"• {ev['timestamp']}")

            # st.write("### Made Shot Timestamps")
            # for ev in g30["shot_made_events"]:
            #     st.write(f"• {ev['timestamp']}")

            st.write("### 📸 Attempt Snapshots")
            for ts, path in g30_attempt_frames:
                st.image(path, caption=f"Attempt @ {ts}", use_column_width=True)

            st.write("### 🏀 Made Shot Snapshots")
            for ts, path in g30_made_frames:
                st.image(path, caption=f"Made Shot @ {ts}", use_column_width=True)


            st.write("### 📝 Coaching Feedback")

            # cf = g25["coaching_feedback"]   # for Gemini 2.5 Pro
            cf = g30["coaching_feedback"] # for Gemini 3 Pro Preview

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

            # Optional: Show video limitations
            if cf["limitations_in_video"]:
                st.markdown("#### ⚠️ Limitations in Video")
                for item in cf["limitations_in_video"]:
                    st.write(f"• {item}")

    # ---------------------------
    # Folder name
    # ---------------------------
    st.write("📁 Saved to:")
    st.code(session_folder)
