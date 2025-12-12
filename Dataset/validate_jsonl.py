import json

filename = "val_fixed.jsonl"   # change if needed

with open(filename, encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            json.loads(line)
        except Exception as e:
            print(f"❌ Error in {filename} at line {i}: {e}")
            break
        else:
            print(f"✅ Line {i} OK")

print("Validation complete.")
