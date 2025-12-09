import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------------------------------
# Helper: Run shot counter prompt for a specific model
# -----------------------------------------------------
def run_shot_analysis(model_name, video_bytes):
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    prompt = """
You are a professional basketball shooting analyst.

ONLY report the following:

How many shots did the person try (count per 10-second intervals + total)
How many shots did the person actually made (count per 10-second intervals + total)

Rules:
- Count every clear shooting motion as an attempt.
- Count as 'made' only when the ball clearly goes inside the basket.
- If visibility is bad, write: "Not enough visual information to count accurately."

Format EXACTLY like this:

Shots Attempted:
- Per 10 sec: X, X, X ...
- Total: X

Shots Made:
- Per 10 sec: X, X, X ...
- Total: X
"""

    model = genai.GenerativeModel(model_name)

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


# -----------------------------------------------------
# Main comparison function (used in app.py)
# -----------------------------------------------------
def gemini_analyse(video_bytes):
    # Run both models
    output_25 = run_shot_analysis("gemini-2.5-pro", video_bytes)
    output_30 = run_shot_analysis("gemini-3-pro-preview", video_bytes)

    # Return side-by-side results to Streamlit
    return {
        "gemini_2_5_pro": output_25,
        "gemini_3_pro_preview": output_30
    }
