import os
import base64
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """
You are a basketball video analyst.

You are given multiple consecutive frames from ONE single shot attempt.

Task:
1. Decide whether the shot is MADE or MISSED.
2. Provide a short overall description of the shot outcome.

Rules:
- Only say MADE if the ball clearly passes through the hoop.
- If unclear, say MISSED.
- Do NOT guess.

Return the output in exactly this format:

RESULT: <MADE | MISSED>
DESCRIPTION: <one or two sentences>
"""

# --------------------------------------------------
# Single GPT Vision run
# --------------------------------------------------
def _run_single(frame_paths):
    content = [{"type": "text", "text": PROMPT}]

    for path in frame_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0
    )

    text = resp.choices[0].message.content.strip()
    return text


# --------------------------------------------------
# Parallel GPT Vision runs
# --------------------------------------------------
def analyze_frames_parallel(frame_paths, num_runs=5):
    """
    Runs GPT Vision multiple times in parallel on the same frames.
    Returns a list of raw outputs.
    """
    results = []

    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = [
            executor.submit(_run_single, frame_paths)
            for _ in range(num_runs)
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                output = future.result()
                results.append({
                    "run": i,
                    "raw": output
                })
            except Exception as e:
                results.append({
                    "run": i,
                    "raw": f"ERROR: {e}"
                })

    return results
