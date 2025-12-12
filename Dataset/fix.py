import json

input_file = "val.jsonl"
output_file = "val_fixed.jsonl"

objects = []
buffer = ""

with open(input_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        buffer += line

        if buffer.endswith("}"):
            try:
                obj = json.loads(buffer)
                objects.append(obj)
                buffer = ""
            except json.JSONDecodeError:
                # keep accumulating lines
                continue

if buffer.strip():
    raise ValueError("❌ Incomplete JSON object at end of file")

with open(output_file, "w", encoding="utf-8") as f:
    for obj in objects:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")

print(f"✅ Converted {len(objects)} JSON objects into proper JSONL: {output_file}")
