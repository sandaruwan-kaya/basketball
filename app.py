import streamlit as st
from gemini_engine import gemini_analyse
from gpt_engine import gpt5_analyse

st.set_page_config(page_title="Basketball AI Coach", layout="wide")

st.title("🏀 Basketball Training Video Analysis (POC)")

st.write("Upload a video and choose which AI engine should analyze your movement.")

engine = st.selectbox(
    "Choose AI Engine",
    ("Gemini Vision",)
    # ("Gemini Vision", "GPT-5 Vision")
)

video = st.file_uploader("Upload a training video:", type=["mp4", "mov", "avi"])

if st.button("Analyze"):
    if video is None:
        st.error("Please upload a video.")
    else:
        video_bytes = video.read()

        with st.spinner("Analyzing…"):
            if engine == "Gemini Vision":
                result = gemini_analyse(video_bytes)
            # else:
            #     result = gpt5_analyse(video_bytes)

        st.subheader("🏀 Coaching Feedback")
        st.write(result)
