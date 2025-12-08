import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------
# Helper: Run model with prompt
# -------------------------------
def run_shot_counter(model_name, video_bytes):
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    prompt = """
You are an expert basketball video analyst.

ONLY OUTPUT THE FOLLOWING FIELDS:

1. How many shots did the person try (count per 10-second intervals + total)
2. How many shots did the person actually make (count per 10-second intervals + total)

RULES:
- Count each shot attempt (any shooting motion toward the basket).
- Count made shots only when the ball clearly goes into the hoop.
- If the video quality or angle prevents determining a shot attempt or made shot,
  say: "Not enough visual information for accurate counting."

FORMAT STRICTLY AS JSON ONLY:

{
  "shots_attempted": {
    "per_10_seconds": [],
    "total": 0
  },
  "shots_made": {
    "per_10_seconds": [],
    "total": 0
  }
}
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


# -----------------------------------------------
# MAIN: Analyse video using both Gemini models
# -----------------------------------------------
def gemini_analyse(video_bytes):

    result_25 = run_shot_counter("gemini-2.5-pro", video_bytes)
    result_30 = run_shot_counter("gemini-3-pro-preview", video_bytes)

    # Combined side-by-side return
    return {
        "gemini_2_5_pro": result_25,
        "gemini_3_pro_preview": result_30
    }
