import cv2
import base64
import json
import tempfile
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------
# Structured basketball analysis prompt
# -----------------------------------------------------
PROMPT = """
You are an expert basketball shooting coach. You will be given a SEQUENCE OF IMAGES (frames extracted from a video). 
Analyze the frames and produce the SAME JSON STRUCTURE used by the Gemini engine.

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

RULES:
- STRICT JSON OUTPUT ONLY.
- NO guessing. Only count a shot if the ball and motion are clearly visible.
- If uncertain, DO NOT count the shot.
- Output MUST be deterministic.
- Use timestamps provided inside each frame payload.
"""

# -----------------------------------------------------
# Extract frames safely on Windows
# -----------------------------------------------------
def extract_frames(video_bytes, fps=2, max_frames=15):
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps)) if video_fps else 1

    frames = []
    frame_count = 0
    frame_index = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Resize to reduce request size
            frame = cv2.resize(frame, (640, 360))

            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buffer).decode("utf-8")

            frames.append({
                "timestamp": f"{timestamp:.2f}s",
                "b64": b64
            })

            frame_count += 1

        frame_index += 1

    cap.release()

    try:
        os.remove(temp_path)
    except:
        pass

    return frames


# -----------------------------------------------------
# Try to parse JSON safely
# -----------------------------------------------------
def try_parse_json(raw):
    raw = raw.strip()

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
            return None

    return None


# -----------------------------------------------------
# Main GPT video analysis function
# -----------------------------------------------------
def gpt_analyse(video_bytes):
    frames = extract_frames(video_bytes, fps=2, max_frames=15)

    image_payloads = []
    for f in frames:
        image_payloads.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{f['b64']}"
            }
        })

        # Add timestamp as a separate text message to help the model
        image_payloads.append({
            "type": "text",
            "text": f"Frame timestamp: {f['timestamp']}"
        })

    messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here are the extracted frames (max 15). Analyze carefully."},
                *image_payloads
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",      # Safest and most powerful for vision
        messages=messages,
        temperature=0
    )

    raw = response.choices[0].message.content
    parsed = try_parse_json(raw)

    return {
        "frames": frames,   # <-- NEW: return extracted frames to Streamlit
        "gemini_2_5_pro": {
            "raw": raw,
            "parsed": parsed
        },
        "gemini_3_pro_preview": {
            "raw": raw,
            "parsed": parsed
        }
    }
