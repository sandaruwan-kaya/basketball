import google.generativeai as genai
import base64
import os
import json
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-pro"

# -----------------------------
# VIDEO ANALYSIS PROMPT
# -----------------------------
PROMPT = """
You are an expert basketball shooting coach AND a precise video analyst.
Your job is to (1) COUNT SHOTS.

You MUST follow the instructions below exactly and ONLY return the JSON object in the specified format.

--------------------------------
TASK
--------------------------------
From the provided basketball shooting video:

1. Identify every DISTINCT shot attempt.
2. Identify which of those attempts are MADE shots.

--------------------------------
STRICT OUTPUT FORMAT
--------------------------------
{
  "shots_attempted": { "total": 0 },
  "shots_made": { "total": 0 },
  "shot_attempt_events": [{ "timestamp": "" }],
  "shot_made_events": [{ "timestamp": "" }],
  "coaching_feedback": {
    "summary": "",
    "stance_and_balance": "",
    "footwork": "",
    "ball_gather_and_set_point": "",
    "release_and_follow_through": "",
    "timing_and_rhythm": "",
    "shot_arc_and_power": "",
    "consistency": "",
    "shot_selection": "",
    "areas_to_improve": [],
    "limitations_in_video": []
  }
}

--------------------------------
RULES
--------------------------------
- No guessing.
- shots_made.total <= shots_attempted.total
"""

# -----------------------------
# JSON PARSER
# -----------------------------
def try_parse_json(raw_text: str):
    raw = raw_text.strip()
    try:
        return json.loads(raw)
    except:
        pass

    if "```" in raw:
        try:
            cleaned = raw.split("```")[1]
            cleaned = cleaned.replace("json", "").strip()
            return json.loads(cleaned)
        except:
            pass

    return None


# -----------------------------
# VIDEO ANALYSIS
# -----------------------------
def analyze_video_raw(video_bytes: bytes):
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(
        [
            PROMPT,
            {"mime_type": "video/mp4", "data": video_base64}
        ]
    )

    raw = response.text or ""
    parsed = try_parse_json(raw)

    if parsed is None:
        raise ValueError("Failed to parse Gemini video analysis")

    return parsed, raw


# -----------------------------
# CLIP RECOMMENDATION (TEXT ONLY)
# -----------------------------
def recommend_clips_with_gemini(analysis_json: dict, courses_json: list):
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = f"""
You are a professional basketball skills coach.
 
Your task is to recommend relevant training clips for a player based specifically on their "Areas to Improve".
 
You will be given:
1) PLAYER_ANALYSIS (JSON) - Pay special attention to "coaching_feedback" -> "areas_to_improve".
2) COURSE_CATALOG (JSON) - A list of available courses and their clips.
 
MANDATORY RULES:
- You MUST select clips based ONLY on the context of "areas_to_improve" found in the PLAYER_ANALYSIS.
- For each recommendation, analyze the combination of Course Title, Course Description, Clip Name, and Clip Description from the COURSE_CATALOG to find the most relevant matches for the player's specific improvement needs.
- Recommend ONLY the clips that are highly compatible and convenient for the identified improvement areas.
- You can recommend a MAXIMUM of 5 clips. Do not exceed 5.
- You do NOT need to provide 5 clips if fewer (or none) are truly relevant. Only provide those that genuinely match the context.
- DO NOT invent clips or modify names/links.
- Output MUST be valid JSON only.
 
OUTPUT FORMAT:
{{
  "recommended_clips": [
    {{
      "course_title": "",
      "clip_name": "",
      "clip_link": "",
      "reason": "Explain how this specific clip addresses one or more items in the 'areas_to_improve' context."
    }}
  ]
}}
 
PLAYER_ANALYSIS:
{json.dumps(analysis_json, indent=2)}
 
COURSE_CATALOG:
{json.dumps(courses_json, indent=2)}
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()
    parsed = try_parse_json(raw)

    if parsed is None:
        raise ValueError("Failed to parse Gemini clip recommendations")

    return parsed
