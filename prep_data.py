import json

from datasets import load_dataset

print("Downloading prompt injection dataset from Hugging Face...")

# Pull open-source security dataset (contains safe & attack prompts)
dataset = load_dataset("deepset/prompt-injections")

def format_and_save(split_data, filename):
    formatted_rows = []
    for item in split_data:
        text = item.get("text") or item.get("prompt")
        label = item.get("label")
        
        if text is not None and label is not None:
            formatted_rows.append({
                "text": str(text).strip(),
                "label": int(label) # 0 = Safe, 1 = Injection
            })
            
    with open(filename, "w", encoding="utf-8") as f:
        for row in formatted_rows:
            f.write(json.dumps(row) + "\n")
            
    print(f"Successfully saved {len(formatted_rows)} records to {filename}")

# Create both dataset files
format_and_save(dataset["train"], "train.jsonl")
format_and_save(dataset["test"], "test.jsonl")

print("Dataset preparation complete!")
