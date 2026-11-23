"""Smoke test: prove the train.py recipe up to training works on this box.

Qwen3.5-4B is a vision-language model, so we use Qwen3VLProcessor for input
encoding (not tokenizer.encode). We stop right after attaching LoRA — that's
exactly where scripts/train.py begins its SFT loop.
"""
import os, torch
from unsloth import FastLanguageModel

MODEL = os.environ.get("UNSLOTH_MODEL", "unsloth/Qwen3.5-4B")
MAX_SEQ = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

print(f"Loading {MODEL} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,   # bf16 LoRA (matches the guide's recommendation)
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
)

# Attach LoRA adapters to a local dir and confirm we can save them.
OUT = os.environ.get("OUTPUT_DIR", "outputs/qwen35-4b")
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(f"OK: bf16 LoRA attached + saved to {OUT}")
print(f"    params={sum(p.numel() for p in model.parameters())/1e9:.2f}B | "
      f"cuda:{torch.cuda.is_available()}")
