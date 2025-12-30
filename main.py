from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from gemini_engine import analyze_video_raw, recommend_clips_with_gemini

import os
import json
import datetime
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
import re
import traceback


def safe_fs_name(value: str, max_len: int = 50) -> str:
    """
    Make a string safe for Windows/Linux file systems.
    """
    value = value.strip()
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)   # remove illegal chars
    value = re.sub(r"\s+", "_", value)             # spaces -> _
    value = re.sub(r"_+", "_", value)               # collapse ___
    value = value.strip("._")                       # trim bad edges
    return value[:max_len] or "unknown"


# -----------------------------
# CONFIG
# -----------------------------
LOG_DIR = "logs"
COURSES_FILE = "courses.json"

NUM_GEMINI_RUNS = 5
GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=NUM_GEMINI_RUNS)

os.makedirs(LOG_DIR, exist_ok=True)

with open(COURSES_FILE, "r", encoding="utf-8") as f:
    COURSES = json.load(f)

# -----------------------------
# LOGGING CONFIG (UPDATED – FILE + CONSOLE + ROTATION)
# -----------------------------
LOG_FILE = os.path.join(LOG_DIR, "api.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=10,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Basketball Video Analysis API")

app.mount(
    "/.well-known",
    StaticFiles(directory=".well-known"),
    name="well-known",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/analyze-video")
async def analyze_video_endpoint(
    tester_name: str = Form(...),
    video: UploadFile = File(...)
):
    request_start = time.time()

    logger.info(f"▶️ API START | tester={tester_name} | video={video.filename}")

    if tester_name.strip() == "":
        logger.error("❌ Missing tester_name")
        raise HTTPException(status_code=400, detail="tester_name is required")

    if video.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        logger.error(
            f"❌ Unsupported video format | tester={tester_name} | type={video.content_type}"
        )
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # -----------------------------
    # READ VIDEO
    # -----------------------------
    video_bytes = await video.read()
    loop = asyncio.get_running_loop()

    # -----------------------------
    # CREATE SESSION + SAVE INPUTS EARLY (NEW)
    # -----------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_tester = safe_fs_name(tester_name)
    session_folder = os.path.join(LOG_DIR, f"{safe_tester}_{timestamp}")

    try:
        os.makedirs(session_folder, exist_ok=True)
    except Exception as e:
        logger.error(f"❌ Failed to create session folder: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid tester name. Please avoid special characters."
        )

    with open(os.path.join(session_folder, video.filename), "wb") as f:
        f.write(video_bytes)

    with open(os.path.join(session_folder, "meta.txt"), "w") as f:
        f.write(f"Tester: {tester_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Video Name: {video.filename}\n")
        f.write("Model: gemini-2.5-pro\n")

    with open(os.path.join(session_folder, "status.txt"), "w") as f:
        f.write("RECEIVED\n")

    # -----------------------------
    # GEMINI VIDEO ANALYSIS
    # -----------------------------
    gemini_analysis_start = time.time()
    logger.info("🤖 Gemini ANALYSIS started")

    try:
        analysis, raw = await loop.run_in_executor(
            GEMINI_EXECUTOR,
            analyze_video_raw,
            video_bytes
        )
    except Exception as e:
        logger.error(
            f"❌ Gemini ANALYSIS failed | tester={tester_name} | error={e}"
        )

        with open(os.path.join(session_folder, "status.txt"), "w") as f:
            f.write("ANALYSIS_FAILED\n")

        with open(os.path.join(session_folder, "error.txt"), "w", encoding="utf-8") as f:
            f.write("Exception type:\n")
            f.write(f"{type(e)}\n\n")

            f.write("Exception message:\n")
            f.write(f"{str(e)}\n\n")

            f.write("Full traceback:\n")
            f.write(traceback.format_exc())

        raise HTTPException(status_code=500, detail="Gemini analysis failed")

    with open(os.path.join(session_folder, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=4)

    with open(os.path.join(session_folder, "raw_gemini.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    logger.info(
        f"🤖 Gemini ANALYSIS finished | latency={round(time.time() - gemini_analysis_start, 2)}s"
    )

    # -----------------------------
    # GEMINI RECOMMENDATION
    # -----------------------------
    gemini_reco_start = time.time()
    logger.info("🤖 Gemini RECOMMENDATION started")

    try:
        recommendations = await loop.run_in_executor(
            GEMINI_EXECUTOR,
            recommend_clips_with_gemini,
            analysis,
            COURSES
        )
    except Exception as e:
        logger.error(f"❌ Gemini RECOMMENDATION failed | tester={tester_name} | error={e}")

        with open(os.path.join(session_folder, "status.txt"), "w") as f:
            f.write("RECOMMENDATION_FAILED\n")

        with open(os.path.join(session_folder, "error.txt"), "w") as f:
            f.write(str(e))

        raise HTTPException(status_code=500, detail="Clip recommendation failed")

    with open(os.path.join(session_folder, "recommended_clips.json"), "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4)

    with open(os.path.join(session_folder, "status.txt"), "w") as f:
        f.write("SUCCESS\n")

    # -----------------------------
    # TOTAL LATENCY
    # -----------------------------
    total_latency = round(time.time() - request_start, 2)
    logger.info(
        f"✅ API DONE | tester={tester_name} | total_latency={total_latency}s"
    )

    return {
        "status": "success",
        "model": "gemini-2.5-pro",
        "tester_name": tester_name,
        "analysis": analysis,
        "recommended_clips": recommendations["recommended_clips"]
    }
