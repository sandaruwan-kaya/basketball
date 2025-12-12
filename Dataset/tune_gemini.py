import vertexai
from vertexai.preview.tuning import SupervisedTuningJob

PROJECT_ID = "inner-doodad-481015-a5"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

job = SupervisedTuningJob.create(
    display_name="basketball-shot-counter-v1",
    source_model="gemini-2.5-pro",
    training_dataset="gs://basketball-gemini-train/data/train.jsonl",
    validation_dataset="gs://basketball-gemini-train/data/val.jsonl",
    epochs=3
)

job.wait()
print("✅ Tuning completed successfully")
