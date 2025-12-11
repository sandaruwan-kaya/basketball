import os
import time
import json
import uuid
import cv2
import math
import tempfile
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- Pydantic Schema Definition ---
# This class defines the EXACT structure the model must return for each chunk.
class ChunkSummary(BaseModel):
    """Defines the required JSON structure for video analysis per chunk."""

    shots_attempted: int = Field(
        ...,
        description="The total number of shots attempted in this video segment, including both makes and misses."
    )
    shots_made: int = Field(
        ...,
        description="The total number of successful shots (makes) in this video segment."
    )
    # NEW: per-chunk, relative timestamps (seconds) as strings
    shot_attempt_events: list[str] = Field(
        default_factory=list,
        description="Timestamps in seconds (relative to this chunk) for each shot attempt."
    )
    shot_made_events: list[str] = Field(
        default_factory=list,
        description="Timestamps in seconds (relative to this chunk) for each made shot."
    )

# --- Initialization ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY")
client = genai.Client(api_key=api_key)  # The client object is initialized here


# -----------------------------------------------------
# Video Utility Functions
# -----------------------------------------------------
def chunk_video_bytes(video_bytes, chunk_duration=12):
    """Chunks video bytes into segments with a 1.0 second gap between them."""
    temp_name = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")

    with open(temp_name, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs = total_frames / fps
    print(f"Detected Video FPS: {fps}")
    chunks = []
    start = 0

    while start < total_secs:
        end = min(start + chunk_duration, total_secs)
        chunks.append((start, end))

        # Introduces a 1.0s gap between chunks as requested
        start += chunk_duration + 1.0

    cap.release()
    return temp_name, chunks, fps


def encode_video_segment(video_path, start_sec, end_sec, fps):
    """Extract a MP4 segment using ffmpeg without re-encoding."""
    temp_segment = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_segment.mp4")

    duration = end_sec - start_sec

    cmd = (
        f'ffmpeg -y -loglevel error -ss {start_sec} -i "{video_path}" '
        f'-t {duration} -c copy "{temp_segment}"'
    )

    result = os.system(cmd)

    if result != 0 or not os.path.exists(temp_segment):
        print("FFmpeg failed to create segment:", temp_segment)
        return None

    with open(temp_segment, "rb") as f:
        segment_bytes = f.read()

    os.remove(temp_segment)
    return segment_bytes


# -----------------------------------------------------
# Generic Gemini call (2 models use this)
# -----------------------------------------------------
def call_gemini_model(client, model_name: str, prompt: str, video_bytes: bytes) -> ChunkSummary | dict:
    """
    Calls Gemini model with video + prompt and returns a Pydantic object (parsed JSON),
    or an error dict if something goes wrong.
    """
    video_part = types.Part.from_bytes(
        data=video_bytes,
        mime_type="video/mp4"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                video_part,
                prompt
            ],
            config=types.GenerateContentConfig(
                # Enforce JSON output using the Pydantic class
                response_mime_type="application/json",
                response_schema=ChunkSummary,
            )
        )
        return response.parsed

    except Exception as e:
        error_message = f"Model {model_name} failed: {e}"
        print(f"Error: {error_message}")
        return {"error": error_message}


# -----------------------------------------------------
# Single Analysis Run Function (one pass over all chunks)
# -----------------------------------------------------
def _get_single_run_totals(client, prompt, video_path, chunks, fps):
    """
    Executes one full analysis pass over all chunks and returns totals
    for BOTH models, including absolute timestamps.
    """
    cumulative_totals = {
        "gemini-3-pro-preview": {
            "shots_attempted": 0,
            "shots_made": 0,
            "attempt_events": [],  # absolute timestamps (seconds)
            "made_events": [],     # absolute timestamps (seconds)
        },
        "gemini-2.5-pro": {
            "shots_attempted": 0,
            "shots_made": 0,
            "attempt_events": [],
            "made_events": [],
        },
    }

    for i, (start, end) in enumerate(chunks):
        print(f"--- [Chunk {i+1}/{len(chunks)}] {start:.2f}s → {end:.2f}s ---")

        segment_bytes = encode_video_segment(video_path, start, end, fps)
        if segment_bytes is None:
            print(f"Warning: Segment {i+1} failed to encode. Skipping.")
            continue

        # Call both models on this chunk
        g30_result = call_gemini_model(client, "gemini-3-pro-preview", prompt, segment_bytes)
        g25_result = call_gemini_model(client, "gemini-2.5-pro", prompt, segment_bytes)

        # Convert Pydantic object to dictionary for robust parsing (handles error dict)
        g30_parsed = g30_result.model_dump() if isinstance(g30_result, ChunkSummary) else g30_result
        g25_parsed = g25_result.model_dump() if isinstance(g25_result, ChunkSummary) else g25_result

        # ---- GEMINI 3.0 PREVIEW ACCUMULATION ----
        if "error" not in g30_parsed:
            model_key = "gemini-3-pro-preview"

            # counts
            cumulative_totals[model_key]["shots_attempted"] += g30_parsed.get("shots_attempted", 0)
            cumulative_totals[model_key]["shots_made"] += g30_parsed.get("shots_made", 0)

            # timestamps: convert relative → absolute
            rel_attempts = g30_parsed.get("shot_attempt_events", [])
            rel_mades = g30_parsed.get("shot_made_events", [])

            abs_attempts = []
            abs_mades = []
            for t in rel_attempts:
                try:
                    if str(t).strip() != "":
                        abs_attempts.append(start + float(t))
                except ValueError:
                    continue
            for t in rel_mades:
                try:
                    if str(t).strip() != "":
                        abs_mades.append(start + float(t))
                except ValueError:
                    continue

            cumulative_totals[model_key]["attempt_events"].extend(abs_attempts)
            cumulative_totals[model_key]["made_events"].extend(abs_mades)

        # ---- GEMINI 2.5 PRO ACCUMULATION ----
        if "error" not in g25_parsed:
            model_key = "gemini-2.5-pro"

            # counts
            cumulative_totals[model_key]["shots_attempted"] += g25_parsed.get("shots_attempted", 0)
            cumulative_totals[model_key]["shots_made"] += g25_parsed.get("shots_made", 0)

            # timestamps: convert relative → absolute
            rel_attempts = g25_parsed.get("shot_attempt_events", [])
            rel_mades = g25_parsed.get("shot_made_events", [])

            abs_attempts = []
            abs_mades = []
            for t in rel_attempts:
                try:
                    if str(t).strip() != "":
                        abs_attempts.append(start + float(t))
                except ValueError:
                    continue
            for t in rel_mades:
                try:
                    if str(t).strip() != "":
                        abs_mades.append(start + float(t))
                except ValueError:
                    continue

            cumulative_totals[model_key]["attempt_events"].extend(abs_attempts)
            cumulative_totals[model_key]["made_events"].extend(abs_mades)

    return cumulative_totals


# -----------------------------------------------------
# Modified Main Analysis Function (runs N times)
# -----------------------------------------------------
def gemini_analyse(prompt, video_bytes, log_dir, num_runs=5):

    # Use the global 'client' object
    global client

    # 1. Setup and Chunking (Done only once)
    session = str(int(time.time()))
    session_dir = os.path.join(log_dir, session)
    os.makedirs(session_dir, exist_ok=True)
    video_path, chunks, fps = chunk_video_bytes(video_bytes)

    # 2. Initialize Arrays for Final Output (KEEPING ORIGINAL STRUCTURE + NEW ARRAYS)
    final_results = {
        "gemini-3-pro-preview": {
            "shots_attempted": [],  # list of totals per run
            "shots_made": [],       # list of totals per run
            "attempt_events": [],   # list of [timestamps...] per run
            "made_events": [],      # list of [timestamps...] per run
        },
        "gemini-2.5-pro": {
            "shots_attempted": [],
            "shots_made": [],
            "attempt_events": [],
            "made_events": [],
        },
    }

    try:
        # 3. Main Loop: Run the analysis N times
        for run_number in range(1, num_runs + 1):
            print("\n=======================================================")
            print(f"STARTING RUN {run_number}/{num_runs}")
            print("=======================================================")

            # Execute a single analysis pass, passing the client object
            run_totals = _get_single_run_totals(client, prompt, video_path, chunks, fps)

            # Aggregate the results into the final array structure
            # --- gemini-3-pro-preview ---
            final_results["gemini-3-pro-preview"]["shots_attempted"].append(
                run_totals["gemini-3-pro-preview"]["shots_attempted"]
            )
            final_results["gemini-3-pro-preview"]["shots_made"].append(
                run_totals["gemini-3-pro-preview"]["shots_made"]
            )
            final_results["gemini-3-pro-preview"]["attempt_events"].append(
                run_totals["gemini-3-pro-preview"]["attempt_events"]
            )
            final_results["gemini-3-pro-preview"]["made_events"].append(
                run_totals["gemini-3-pro-preview"]["made_events"]
            )

            # --- gemini-2.5-pro ---
            final_results["gemini-2.5-pro"]["shots_attempted"].append(
                run_totals["gemini-2.5-pro"]["shots_attempted"]
            )
            final_results["gemini-2.5-pro"]["shots_made"].append(
                run_totals["gemini-2.5-pro"]["shots_made"]
            )
            final_results["gemini-2.5-pro"]["attempt_events"].append(
                run_totals["gemini-2.5-pro"]["attempt_events"]
            )
            final_results["gemini-2.5-pro"]["made_events"].append(
                run_totals["gemini-2.5-pro"]["made_events"]
            )

            print(f"RUN {run_number} COMPLETE. Totals: {run_totals}")

    finally:
        # 4. Cleanup (Only run once after all analysis is done)
        if os.path.exists(video_path):
            os.remove(video_path)

    # 5. Format Final Output (KEEPING ORIGINAL TOP-LEVEL SHAPE)
    final_output = {
        "model_consistency_analysis": final_results
    }

    final_output_path = os.path.join(session_dir, "consistency_analysis.json")
    with open(final_output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    # Return the full combined dictionary and the session directory
    return final_output, session_dir
