import google.generativeai as genai
import base64
import os
import time
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import vertexai
from vertexai.generative_models import GenerativeModel, Part


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

def slog(msg):
    timestamp = time.strftime("[%H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)                # terminal / streamlit backend logs


vertexai.init(
    project="inner-doodad-481015-a5",
    location="us-central1"
)

TUNED_ENDPOINT = (
    "projects/inner-doodad-481015-a5/"
    "locations/us-central1/"
    "endpoints/5678607022044479488"
)

def run_tuned_gemini(prompt: str, video_bytes: bytes):
    slog(f"Starting model: gemini-2.5-pro-tuned")
    model = GenerativeModel(TUNED_ENDPOINT)

    start = time.time()
    elapsed = round(time.time() - start, 2)
    video_part = Part.from_data(
        mime_type="video/mp4",
        data=video_bytes
    )

    response = model.generate_content(
        [video_part, prompt],
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "top_k": 1
        }
    )
    slog(f"gemini-2.5-pro-tuned: Model finished in {elapsed} sec")
    return response.text

def run_gemini(model_name: str, prompt: str, video_bytes: bytes, ui_log_buffer):
    ui_log_buffer = log(f"Starting model: {model_name}", ui_log_buffer)

    # ---------------------------------------------------------
    # Base64 encode
    # ---------------------------------------------------------
    ui_log_buffer = log(f"{model_name}: Encoding video to base64…", ui_log_buffer)
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    ui_log_buffer = log(f"{model_name}: Initializing model…", ui_log_buffer)
    if model_name == "gemini-2.5-pro-tuned":
        model = genai.GenerativeModel(TUNED_ENDPOINT)
    else:
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
            "top_p": 1,
            "top_k": 1,
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

def compare_models(prompt, video_bytes):
    with ThreadPoolExecutor() as executor:
        f_base_25 = executor.submit(
            run_gemini,
            "gemini-2.5-pro",
            prompt,
            video_bytes,
            []
        )

        f_tuned = executor.submit(
            run_tuned_gemini,
            prompt,
            video_bytes
        )

        f_gemini_3 = executor.submit(
            run_gemini,
            "gemini-3-pro-preview",
            prompt,
            video_bytes,
            []
        )

    r25 = f_base_25.result()       # already a full dict
    r30 = f_gemini_3.result()     # already a full dict

    tuned_raw = f_tuned.result()  # STRING → normalize

    return {
        "gemini_2_5_pro": r25,
        "gemini_2_5_pro_tuned": {
            "model": "gemini-2.5-pro-tuned",
            "latency": None,
            "raw": tuned_raw,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "log": []
        },
        "gemini_3_pro_preview": r30
    }

def compare_models_multi_run(prompt, video_bytes, num_runs=5):
    """
    Run gemini-2.5-pro-tuned multiple times in parallel
    and return all outputs.
    """

    results = {}

    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = {}

        for i in range(num_runs):
            futures[i] = executor.submit(
                run_tuned_gemini,
                prompt,
                video_bytes
            )

        for i, future in futures.items():
            results[f"run_{i+1}"] = {
                "model": "gemini-2.5-pro-tuned",
                "raw": future.result(),
                "latency": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "log": []
            }

    return results
