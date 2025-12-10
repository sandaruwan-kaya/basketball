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

    # -------------------------------
    # 🔥 Generation configuration here
    # -------------------------------
    generation_config_25 = {
        "temperature": 0.2,          # lower = more deterministic
        "top_p": 0.4,
        "top_k": 10,
        "max_output_tokens": 8192,
        "candidate_count": 1,
        # "thinking_budget": type.ThinkingConfig(thinking_budget=1024),
        # "response_mime_type": "application/json",  # force JSON
        # "stop_sequences": ["END"],
    }

    generation_config_30 = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 65535,
        "candidate_count": 1,
        # "thinking_budget": type.ThinkingConfig(thinking_budget=1024),
        # "response_mime_type": "application/json",  # force JSON
        # "stop_sequences": ["END"],
    }

    start = time.time()

    response_25 = model.generate_content(
        [
            prompt,
            {"mime_type": "video/mp4", "data": video_base64}
        ],
        generation_config=generation_config_25
    )

    response_30 = model.generate_content(
        [
            prompt,
            {"mime_type": "video/mp4", "data": video_base64}
        ],
        generation_config=generation_config_30
    )

    latency = round(time.time() - start, 3)

    if model_name == "gemini-2.5-pro":
        response = response_25
    else:
        response = response_30

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
