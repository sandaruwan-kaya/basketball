import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """
You are a basketball video analyst.

You are given multiple consecutive frames from ONE SINGLE shot attempt.

Task:
- Decide whether the shot is MADE or MISSED.

Rules:
- Say MADE only if the ball clearly passes through the hoop.
- Say MISSED if the ball clearly does not go through the hoop.
- If the result is unclear, return UNCLEAR.
- Do NOT guess.

Return ONLY ONE WORD:
MADE
MISSED
or
UNCLEAR
"""

def analyze_frames(frame_paths):
    print("▶️ [Gemini] Initializing model...")
    model = genai.GenerativeModel("gemini-2.5-pro")  # ✅ VISION MODEL

    inputs = [PROMPT]

    print(f"🖼️ [Gemini] Adding {len(frame_paths)} frames")
    for i, frame_path in enumerate(frame_paths, start=1):
        print(f"   └─ Frame {i}: {frame_path}")
        img = Image.open(frame_path).convert("RGB")
        inputs.append(img)

    print("🚀 [Gemini] Sending request to Gemini...")
    start = time.time()

    response = model.generate_content(
        inputs,
        generation_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 5
        }
    )

    elapsed = round(time.time() - start, 2)
    print(f"✅ [Gemini] Response received in {elapsed}s")

    candidate = response.candidates[0]
    print(f"ℹ️ [Gemini] finish_reason = {candidate.finish_reason}")

    if candidate.finish_reason != 0:
        print("❌ [Gemini] Model returned no output")
        return None

    result = candidate.content.parts[0].text.strip()
    print(f"🎯 [Gemini] Final decision: {result}")
    return result
