from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gemini_engine import analyze_video_raw, recommend_clips_with_gemini

import os
import json
import datetime
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# CONFIG
# -----------------------------
LOG_DIR = "logs"
COURSES_FILE = "courses.json"

NUM_GEMINI_RUNS = 5   # <<< this is your num_runs
GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=NUM_GEMINI_RUNS)

os.makedirs(LOG_DIR, exist_ok=True)

with open(COURSES_FILE, "r", encoding="utf-8") as f:
    COURSES = json.load(f)

# -----------------------------
# LOGGING CONFIG
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Basketball Video Analysis API")

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

    logging.info(
        f"▶️ API START | tester={tester_name} | video={video.filename}"
    )

    if tester_name.strip() == "":
        logging.error("❌ Missing tester_name")
        raise HTTPException(status_code=400, detail="tester_name is required")

    if video.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        logging.error(
            f"❌ Unsupported video format | tester={tester_name} | type={video.content_type}"
        )
        raise HTTPException(status_code=400, detail="Unsupported video format")

    video_bytes = await video.read()
    loop = asyncio.get_running_loop()

    # -----------------------------
    # GEMINI VIDEO ANALYSIS (THREADED)
    # -----------------------------
    gemini_analysis_start = time.time()
    logging.info("🤖 Gemini ANALYSIS started")

    try:
        analysis, raw = await loop.run_in_executor(
            GEMINI_EXECUTOR,
            analyze_video_raw,
            video_bytes
        )
    except Exception as e:
        logging.error(
            f"❌ Gemini ANALYSIS failed | tester={tester_name} | error={e}"
        )
        raise HTTPException(status_code=500, detail=str(e))

    gemini_analysis_latency = round(time.time() - gemini_analysis_start, 2)
    logging.info(
        f"🤖 Gemini ANALYSIS finished | latency={gemini_analysis_latency}s"
    )

    # -----------------------------
    # GEMINI RECOMMENDATION (THREADED)
    # -----------------------------
    gemini_reco_start = time.time()
    logging.info("🤖 Gemini RECOMMENDATION started")

    try:
        recommendations = await loop.run_in_executor(
            GEMINI_EXECUTOR,
            recommend_clips_with_gemini,
            analysis,
            COURSES
        )
    except Exception as e:
        logging.error(
            f"❌ Gemini RECOMMENDATION failed | tester={tester_name} | error={e}"
        )
        raise HTTPException(status_code=500, detail=f"Clip recommendation failed: {e}")

    gemini_reco_latency = round(time.time() - gemini_reco_start, 2)
    logging.info(
        f"🤖 Gemini RECOMMENDATION finished | latency={gemini_reco_latency}s"
    )

    # -----------------------------
    # FILE LOGGING (UNCHANGED)
    # -----------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(LOG_DIR, f"{tester_name}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    with open(os.path.join(session_folder, video.filename), "wb") as f:
        f.write(video_bytes)

    with open(os.path.join(session_folder, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=4)

    with open(os.path.join(session_folder, "recommended_clips.json"), "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4)

    with open(os.path.join(session_folder, "raw_gemini.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    # -----------------------------
    # TOTAL LATENCY
    # -----------------------------
    total_latency = round(time.time() - request_start, 2)
    logging.info(
        f"✅ API DONE | tester={tester_name} | total_latency={total_latency}s"
    )

    return {
        "status": "success",
        "model": "gemini-2.5-pro",
        "tester_name": tester_name,
        "analysis": analysis,
        "recommended_clips": recommendations["recommended_clips"]
    }
