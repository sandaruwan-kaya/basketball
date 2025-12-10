import google.generativeai as genai
import base64
import os
import time
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------------------
# SIMPLE LOGGER
# -------------------------------------------
def log(msg, buffer):
    timestamp = time.strftime("[%H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)                # terminal / streamlit backend logs
    buffer.append(line)        # send to UI
    return buffer


def run_gemini(model_name: str, prompt: str, video_bytes: bytes, ui_log_buffer):
    ui_log_buffer = log(f"Starting model: {model_name}", ui_log_buffer)

    # ---------------------------------------------------------
    # Base64 encode
    # ---------------------------------------------------------
    ui_log_buffer = log(f"{model_name}: Encoding video to base64…", ui_log_buffer)
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    ui_log_buffer = log(f"{model_name}: Initializing model…", ui_log_buffer)
    model = genai.GenerativeModel(model_name)

    # ---------------------------------------------------------
    # Generation config
    # ---------------------------------------------------------
    if model_name == "gemini-2.5-pro":
        generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 8192,
        }
    else:
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }

    ui_log_buffer = log(f"{model_name}: Calling Gemini API… this may take a while.", ui_log_buffer)
    start = time.time()

    response = model.generate_content(
        [
            prompt,
            {"mime_type": "video/mp4", "data": video_base64}
        ],
        generation_config=generation_config
    )

    elapsed = round(time.time() - start, 2)
    ui_log_buffer = log(f"{model_name}: Model finished in {elapsed} sec", ui_log_buffer)

    # ---------------------------------------------------------
    # Extract raw text
    # ---------------------------------------------------------
    ui_log_buffer = log(f"{model_name}: Extracting response text…", ui_log_buffer)
    raw = ""
    try:
        raw = response.text
    except:
        try:
            raw = response.candidates[0].content.parts[0].text
        except:
            raw = ""

    # ---------------------------------------------------------
    # Token usage
    # ---------------------------------------------------------
    usage = getattr(response, "usage_metadata", None)
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    ui_log_buffer = log(f"{model_name}: Completed parsing output.", ui_log_buffer)

    return {
        "model": model_name,
        "latency": elapsed,
        "raw": raw,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "log": ui_log_buffer,
    }


# def compare_models(prompt: str, video_bytes: bytes):
#     # shared log buffer for UI
#     ui_log_buffer = []

#     r25 = run_gemini("gemini-2.5-pro", prompt, video_bytes, ui_log_buffer.copy())
#     r30 = run_gemini("gemini-3-pro-preview", prompt, video_bytes, ui_log_buffer.copy())

#     return {
#         "gemini_2_5_pro": r25,
#         "gemini_3_pro_preview": r30
#     }

def compare_models(prompt: str, video_bytes: bytes):

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_25 = executor.submit(run_gemini, "gemini-2.5-pro", prompt, video_bytes, [])
        future_30 = executor.submit(run_gemini, "gemini-3-pro-preview", prompt, video_bytes, [])

        r25 = future_25.result()
        r30 = future_30.result()

    return {
        "gemini_2_5_pro": r25,
        "gemini_3_pro_preview": r30
    }
