import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

print("🚀 Starting Dataset Expansion Pipeline...\n")

records = []

# ==========================================
# 1. FETCH & PROCESS DATASETS
# ==========================================

# --- Dataset A: deepset/prompt-injections (Baseline) ---
try:
    print("📦 Loading: deepset/prompt-injections...")
    ds_a = load_dataset("deepset/prompt-injections")
    for split in ["train", "test"]:
        for row in ds_a[split]:
            records.append({
                "text": str(row["text"]).strip(),
                "label": int(row["label"]),
                "source": "deepset"
            })
except Exception as e:
    print(f"⚠️ Could not load deepset/prompt-injections: {e}")

# --- Dataset B: S-Labs/prompt-injection-dataset (Obfuscation & Hard Negatives) ---
try:
    print("📦 Loading: S-Labs/prompt-injection-dataset...")
    ds_b = load_dataset("S-Labs/prompt-injection-dataset")
    for split in ds_b.keys():
        for row in ds_b[split]:
            text = row.get("text") or row.get("prompt") or row.get("instruction")
            label = row.get("label")
            if text and label is not None:
                records.append({
                    "text": str(text).strip(),
                    "label": int(label),
                    "source": "s-labs"
                })
except Exception as e:
    print(f"⚠️ Could not load S-Labs: {e}")

# --- Dataset C: prodnull/prompt-injection-repo-dataset (Indirect/Context Attacks) ---
try:
    print("📦 Loading: prodnull/prompt-injection-repo-dataset...")
    ds_c = load_dataset("prodnull/prompt-injection-repo-dataset")
    for split in ds_c.keys():
        for row in ds_c[split]:
            text = row.get("text") or row.get("prompt")
            label = row.get("label")
            if text and label is not None:
                records.append({
                    "text": str(text).strip(),
                    "label": int(label),
                    "source": "prodnull"
                })
except Exception as e:
    print(f"⚠️ Could not load prodnull: {e}")

print(f"\n📊 Total raw samples collected: {len(records)}")

# ==========================================
# 2. CLEANING, DEDUPLICATION & BALANCING
# ==========================================
df = pd.DataFrame(records)

# Drop missing or empty strings
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 3]

# Deduplicate based on text content
before_dedup = len(df)
df = df.drop_duplicates(subset=["text"])
print(f"🧹 Removed {before_dedup - len(df)} duplicate prompts.")

# Check balance
benign_df = df[df["label"] == 0]
injection_df = df[df["label"] == 1]

print(f"\n📈 Current Class Counts:")
print(f"   • Safe / Benign (0): {len(benign_df)}")
print(f"   • Injection (1):    {len(injection_df)}")

# Balance the classes to 1:1 ratio to prevent model bias
min_samples = min(len(benign_df), len(injection_df))
balanced_benign = benign_df.sample(n=min_samples, random_state=42)
balanced_injection = injection_df.sample(n=min_samples, random_state=42)

balanced_df = pd.concat([balanced_benign, balanced_injection]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n⚖️ Balanced Dataset Total: {len(balanced_df)} samples ({min_samples} benign, {min_samples} malicious)")

# ==========================================
# 3. TRAIN / TEST SPLIT & SAVE
# ==========================================
train_df, test_df = train_test_split(balanced_df, test_size=0.15, random_state=42, stratify=balanced_df["label"])

def save_jsonl(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            f.write(json.dumps({"text": row["text"], "label": int(row["label"])}) + "\n")
    print(f"✅ Saved {len(dataframe)} records to {filename}")

print("\n💾 Writing updated JSONL files...")
save_jsonl(train_df, "train.jsonl")
save_jsonl(test_df, "test.jsonl")

print("\n🎉 Dataset expansion complete! You are ready to re-run train_shield.py.")
