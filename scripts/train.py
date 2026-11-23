"""
Basic Unsloth fine-tuning for Qwen3.5-4B.

Adapted from the official Unsloth Qwen3.5 guide:
https://unsloth.ai/docs/models/qwen3.5

This is a minimal SFT (supervised fine-tuning) recipe using LoRA.
Run with the project venv activated:

    source ../venv/bin/activate
    python scripts/train.py

Configuration is centralized at the top so you can tweak it without
touching the training loop.
"""

import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("UNSLOTH_MODEL", "unsloth/Qwen3.5-4B")

# Start small; scale up once it runs. 4B handles 2048 comfortably on 16GB.
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

# LoRA rank / alpha. Keep alpha == r for best results.
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))

# Dataset: a tiny text-only JSONL stream (the guide's example dataset).
# Swap DATA_FILES for your own dataset (see README for options).
DATA_FILES = os.environ.get(
    "DATA_FILES",
    "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl",
)

# Training schedule.
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRAD_ACCUM", "4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "1"))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/qwen35-4b")
SEED = int(os.environ.get("SEED", "3407"))


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
# 4B dense model fits in bf16 LoRA on a 16GB card. We keep load_in_16bit=True
# (bf16 LoRA) rather than QLoRA, matching the guide's recommendation that QLoRA
# is not advised for Qwen3.5.
print(f"Loading {MODEL_NAME} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,   # dense 4B is fine in bf16 LoRA
    load_in_16bit=True,   # bf16/16-bit LoRA
    full_finetuning=False,
)

# ---------------------------------------------------------------------------
# Attach LoRA adapters
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # trades compute for lower VRAM
    random_state=SEED,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ---------------------------------------------------------------------------
# Load dataset (needs a "text" column)
# ---------------------------------------------------------------------------
print(f"Loading dataset from {DATA_FILES} ...")
dataset = load_dataset("json", data_files={"train": DATA_FILES}, split="train")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        seed=SEED,
        dataset_num_proc=1,
    ),
)

print("Starting training ...")
trainer.train()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. LoRA adapters saved to {OUTPUT_DIR}")
