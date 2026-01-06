import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """
You are a basketball video analyst.

You are given multiple consecutive frames from ONE SINGLE shot attempt.

Task:
1. Decide whether the shot is MADE or MISSED.
2. Provide a short overall description of the shot.

Rules:
- Only say MADE if the ball clearly passes through the hoop.
- If unclear, say MISSED.
- Do NOT guess.

Return the output in exactly this format:

RESULT: <MADE | MISSED>
DESCRIPTION: <one or two sentences>
"""

def analyze_frames(frame_paths):
    print("▶️ [Gemini NEW] Preparing request...")

    contents = [PROMPT]

    for i, frame_path in enumerate(frame_paths, start=1):
        print(f"   └─ Frame {i}: {frame_path}")
        with open(frame_path, "rb") as f:
            img_bytes = f.read()

        contents.append(
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg"
            )
        )

    print("🚀 [Gemini NEW] Sending request...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    print("✅ [Gemini NEW] Response received")
    return response.text.strip()
