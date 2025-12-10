import google.generativeai as genai
import base64
import os
import time
import json
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def run_gemini(model_name: str, prompt: str, video_bytes: bytes):
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    model = genai.GenerativeModel(model_name)

    start = time.time()

    response = model.generate_content(
        [
            prompt,
            {"mime_type": "video/mp4", "data": video_base64}
        ]
    )

    latency = round(time.time() - start, 3)

    raw = ""
    try:
        raw = response.text
    except:
        try:
            raw = response.candidates[0].content.parts[0].text
        except:
            raw = ""

    usage = getattr(response, "usage_metadata", None)
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    return {
        "model": model_name,
        "latency": latency,
        "raw": raw,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def compare_models(prompt: str, video_bytes: bytes):
    return {
        "gemini_2_5_pro": run_gemini("gemini-2.5-pro", prompt, video_bytes),
        "gemini_3_pro_preview": run_gemini("gemini-3-pro-preview", prompt, video_bytes)
    }
