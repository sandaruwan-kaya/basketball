import os
import base64
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """
You are a basketball video analyst.

You are given multiple consecutive frames from ONE single shot attempt.

Task:
- Decide whether the shot is MADE or MISSED.

Rules:
- Only say MADE if the ball clearly passes through the hoop.
- If unclear, say MISSED.
- Do NOT guess.

Return ONLY one word:
MADE
or
MISSED
"""

def analyze_frames(frame_paths):
    inputs = [
        {
            "type": "input_text",
            "text": PROMPT
        }
    ]

    for path in frame_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            inputs.append({
                "type": "input_image",
                "image_base64": b64
            })

    response = client.responses.create(
        model="gpt-5-vision",
        input=inputs,
        max_output_tokens=10
    )

    return response.output_text.strip()
