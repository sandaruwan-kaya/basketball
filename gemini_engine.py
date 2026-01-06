import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# --------------------------------------------------
# Single Gemini run
# --------------------------------------------------
def _run_single(frame_paths):
    contents = [PROMPT]

    for path in frame_paths:
        with open(path, "rb") as f:
            contents.append(
                types.Part.from_bytes(
                    data=f.read(),
                    mime_type="image/jpeg"
                )
            )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents
    )

    return response.text.strip()


# --------------------------------------------------
# Parallel Gemini runs
# --------------------------------------------------
def analyze_frames_parallel(frame_paths, num_runs=5):
    results = []

    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = [
            executor.submit(_run_single, frame_paths)
            for _ in range(num_runs)
        ]

        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                results.append({
                    "run": idx,
                    "raw": future.result()
                })
            except Exception as e:
                results.append({
                    "run": idx,
                    "raw": f"ERROR: {e}"
                })

    return results
