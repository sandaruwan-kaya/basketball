import google.generativeai as genai
import base64
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------------------------------
# Structured basketball analysis prompt
# -----------------------------------------------------
prompt = """
You are an expert basketball shooting coach. Analyze the player's shooting technique AND count shots.

### OUTPUT MUST BE IN THIS JSON FORMAT:

{
  "shot_attempt_events": [
    { "timestamp": "" }
  ],
  "shot_made_events": [
    { "timestamp": "" }
  ],
  "shots_attempted": {
    "total": 0
  },
  "shots_made": {
    "total": 0
  },
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

### RULES:
- You MUST NOT guess. Only count a shot if the ball is clearly visible and the motion is fully observable in the video.
- If uncertain, DO NOT count the shot.
- Your output MUST be deterministic. Identical input videos must always produce the same counts.
- Review and analyze every visible shot frame-by-frame logically before counting.
- For each shot attempt, return the exact timestamp (mm:ss.s or ss.s format).
- For each made shot, return the timestamp when the ball enters the hoop.
- timestamps MUST be inside the shot_attempt_events and shot_made_events arrays.
- shots_attempted.total MUST equal shot_attempt_events.length.
- shots_made.total MUST equal shot_made_events.length.
- If timestamp cannot be identified due to video quality or angle, omit that event and explain in limitations_in_video.
- Coaching feedback must be concise and structured.
- DO NOT include extra text outside JSON.
"""



# -----------------------------------------------------
# Helper: Try to parse model output JSON
# -----------------------------------------------------
def try_parse_json(raw_text: str):
    raw = raw_text.strip()

    # Direct parse attempt
    try:
        return json.loads(raw)
    except:
        pass

    # If Gemini wrapped JSON in markdown ```json
    if "```" in raw:
        try:
            cleaned = raw.split("```")[1]
            cleaned = cleaned.replace("json", "").strip()
            return json.loads(cleaned)
        except:
            pass

    # Still failed → return None
    return None


# -----------------------------------------------------
# Main runner for each model (returns raw + parsed + metrics)
# -----------------------------------------------------
def run_shot_counter(model_name, video_bytes):
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    model = genai.GenerativeModel(model_name)

    start_time = time.time()

    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "video/mp4",
                "data": video_base64
            }
        ]
    )

    end_time = time.time()
    latency_sec = round(end_time - start_time, 3)

    # -------------------------
    # SAFE TEXT EXTRACTION
    # -------------------------
    raw = ""

    try:
        raw = response.text or ""
    except Exception:
        try:
            if response.candidates:
                parts = response.candidates[0].content.parts
                if parts:
                    raw = parts[0].text
        except:
            raw = ""

    parsed = try_parse_json(raw)

    # -------------------------
    # TOKEN USAGE (Gemini API)
    # -------------------------
    input_tokens = 0
    output_tokens = 0

    try:
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
    except:
        pass

    return {
        "raw": raw,
        "parsed": parsed,
        "latency_sec": latency_sec,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }
    }


# -----------------------------------------------------
# Compare both models with metrics
# -----------------------------------------------------
def gemini_analyse(video_bytes):

    result_25 = run_shot_counter("gemini-2.5-pro", video_bytes)
    result_30 = run_shot_counter("gemini-3-pro-preview", video_bytes)

    return {
        "gemini_2_5_pro": result_25,
        "gemini_3_pro_preview": result_30
    }