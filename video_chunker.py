from moviepy.editor import VideoFileClip
import math
import io

def chunk_video_bytes(video_bytes, chunk_size=10, overlap=0.1):
    # Write bytes to a temporary file-like object
    temp_file = "temp_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(video_bytes)

    video = VideoFileClip(temp_file)
    duration = video.duration

    chunks = []
    start = 0

    while start < duration:
        end = min(start + chunk_size, duration)

        subclip = video.subclip(start, end)

        # Save to in-memory bytes
        buffer = io.BytesIO()
        subclip.write_videofile(
            "temp_chunk.mp4",
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None
        )
        with open("temp_chunk.mp4", "rb") as f:
            chunk_bytes = f.read()

        chunks.append({
            "start_time": start,
            "end_time": end,
            "bytes": chunk_bytes
        })

        start = start + chunk_size - overlap

    video.close()
    return chunks
