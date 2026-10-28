import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# The base model is gated: read the token from the environment, never hardcode it.
HF_TOKEN = os.environ.get("HF_TOKEN") 

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

print(f"Loading Tokenizer and Base Model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# Prompt Guard 2 uses binary classification: 0 (Benign) and 1 (Malicious)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=2,
    token=HF_TOKEN
)

# Use the GPU when there is one (trained on an RTX 4080 laptop, 12 GB)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load locally formatted JSONL dataset files
data_files = {"train": "train.jsonl", "test": "test.jsonl"}
dataset = load_dataset("json", data_files=data_files)

# Tokenize incoming text strings
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training configurations optimized for RTX 4080 (12GB VRAM)
training_args = TrainingArguments(
    output_dir="./guardrail_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # mixed precision needs CUDA; CPU runs use fp32
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Starting Fine-Tuning on RTX 4080...")
trainer.train()

print("Saving custom fine-tuned model...")
model.save_pretrained("./my_custom_prompt_guard")
tokenizer.save_pretrained("./my_custom_prompt_guard")
print("🎉 Success! Model saved to ./my_custom_prompt_guard")
