from openai import OpenAI
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def gpt5_analyse(video_bytes):
    response = client.responses.create(
        model="gpt-5-vision",
        input="Analyze this basketball training video. Provide corrections for form, balance, and technique.",
        attachments=[
            {
                "file_name": "video.mp4",
                "data": video_bytes,
                "mime_type": "video/mp4"
            }
        ]
    )
    
    return response.output_text
