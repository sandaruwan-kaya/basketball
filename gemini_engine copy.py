# import google.generativeai as genai

# genai.configure(api_key="AIzaSyDlz3_1VehQmr-N6dut9gxxw3c0ly5LSWs")

import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_analyse(video_bytes):
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    COURSE_JSON_DATA = open("courses.json").read()

    prompt = f"""
    You are a professional basketball skills trainer.

    Below is the library of courses and drills we teach. Use this knowledge to classify the user's video and give course-specific feedback.

    COURSE LIBRARY:
    {COURSE_JSON_DATA}

    YOUR TASKS:

    1. **Drill Identification**
    - Identify which course and drill the video most closely matches.
    - If the drill cannot be confidently identified, say:
        "Drill Match: Not enough visual information to determine the drill."

    2. **Uncertainty Detection (IMPORTANT)**
    - If any part of the player's body is missing from the frame (feet, shooting arm, hips, ball, etc.), clearly state:
        "Unable to evaluate ____ because it is not visible in the video."
    - If the angle, lighting, video quality, or motion blur prevents clear analysis, explicitly state:
        "Video quality or angle prevents accurate evaluation of ____."

    3. **Technique Evaluation**
    - Evaluate the player's technique ONLY when the relevant part is visible.
    - If something is partially visible, add:
        "Assessment is limited due to partial visibility."

    4. **Scoring**
    Give a score from 0–100 for:
    - Drill Accuracy
    - Technical Execution
    - Footwork & Control
    - Overall Skill Level
    If scoring is not possible, say:
        "Scoring not possible due to insufficient visual information."

    5. **Feedback Format**
    Provide the final output in this structure:

    How many shots did the person try 
    How many shots did the person actually made
    Course Match:
    Drill Match:
    Missing Information:
    Score Breakdown:
    Strengths:
    Corrections:
    Final Feedback:
    """

    model = genai.GenerativeModel("gemini-2.5-pro")

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "video/mp4",
                "data": video_base64
            }
        ]
    )

    return response.text
