from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gemini_engine import analyze_video_raw, recommend_clips_with_gemini
import os
import json
import datetime

LOG_DIR = "logs"
COURSES_FILE = "courses.json"

os.makedirs(LOG_DIR, exist_ok=True)

with open(COURSES_FILE, "r", encoding="utf-8") as f:
    COURSES = json.load(f)

app = FastAPI(title="Basketball Video Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-video")
async def analyze_video_endpoint(
    tester_name: str = Form(...),
    video: UploadFile = File(...)
):
    if tester_name.strip() == "":
        raise HTTPException(status_code=400, detail="tester_name is required")

    if video.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    video_bytes = await video.read()

    # 1️⃣ Video analysis
    try:
        analysis, raw = analyze_video_raw(video_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2️⃣ Clip recommendation
    try:
        recommendations = recommend_clips_with_gemini(analysis, COURSES)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clip recommendation failed: {e}")

    # 3️⃣ Logging
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

    return {
        "status": "success",
        "model": "gemini-2.5-pro",
        "tester_name": tester_name,
        "analysis": analysis,
        "recommended_clips": recommendations["recommended_clips"]
    }
