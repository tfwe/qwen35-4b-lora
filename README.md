# Unsloth fine-tuning: Qwen3.5-4B

A minimal, working project for basic supervised fine-tuning (SFT) of
**Qwen3.5-4B** with [Unsloth](https://github.com/unslothai/unsloth), based on
the official [Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5).

## Why 4B (and not 9B)

The guide recommends **bf16 LoRA** for Qwen3.5 dense models and explicitly
discourages QLoRA (4-bit) for them. That recipe needs roughly:

| Model | bf16 LoRA VRAM |
|-------|----------------|
| 4B    | ~10GB          |
| 9B    | ~22GB          |

This box has a single **RTX 5070 Ti (16GB VRAM)**, so **9B bf16 LoRA (~22GB)
would OOM**. The 4B model (~10GB) fits comfortably. If you later want 9B,
you'll need a bigger card (or accept QLoRA, which the guide warns against for
Qwen3.5).

## Layout

```
qwen3.5-9b-training/
├── venv/                 # Python 3.14 virtualenv (already created)
├── scripts/
│   ├── train.py          # The training recipe (config at the top)
│   ├── smoke_test.py     # Load + attach LoRA + save (verifies the stack)
│   └── inference.py      # Load base + adapter, generate text
├── data/                 # put your own dataset files here
├── outputs/              # saved LoRA adapters land here
├── requirements.txt
└── README.md
```

## Setup (venv already exists)

The venv was created on **Python 3.14** on purpose: PyTorch's CUDA wheels
(`cu124`) don't yet ship wheels for Python 3.15, but 3.14 works and detects
the RTX 5070 Ti.

```bash
cd qwen3.5-9b-training
source venv/bin/activate

# If you need to (re)create the venv:
#   python3.14 -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
```

Verify CUDA is live before training:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True, NVIDIA GeForce RTX 5070 Ti
```

## Run

```bash
source venv/bin/activate
python scripts/train.py
```

## Results

This run was a 100-step LoRA fine-tune of `unsloth/Qwen3.5-4B` (LoRA r=16, alpha=16) on a single RTX 5070 Ti (16 GB VRAM) under WSL2, with sequence length 2048, batch size 2, and gradient accumulation 4.

The run completed 100 steps. The final logged training loss at step 100 was 1.0766, with grad_norm 0.678 and learning rate 5.56e-07.

Committed to the repo:
- `adapter_model.safetensors` (81 MB), the final LoRA adapter
- `adapter_config.json`, tokenizer, chat template and processor config
- `trainer_state.json` from the checkpoint

The venv and the checkpoint's optimizer state are excluded.

The saved adapter loads and generates text; running `scripts/inference.py` against it produces a response, so the run is verified end to end rather than only by its loss curve.

This is a stack-validation run: 100 steps on a small sample shows the training pipeline works end to end, but it is not a quality benchmark and does not show that the model improved at any task.

## Configure

Everything is controlled by environment variables at the top of
`scripts/train.py` (and overridable at runtime):

| Var | Default | Meaning |
|-----|---------|---------|
| `UNSLOTH_MODEL` | `unsloth/Qwen3.5-4B` | Model to fine-tune |
| `MAX_SEQ_LENGTH` | `2048` | Context length (start small) |
| `LORA_R` / `LORA_ALPHA` | `16` / `16` | LoRA rank / alpha |
| `BATCH_SIZE` | `2` | Per-device train batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `MAX_STEPS` | `100` | Total training steps |
| `DATA_FILES` | OIG JSONL | Dataset URL or local file |
| `OUTPUT_DIR` | `outputs/qwen35-4B` | Where adapters are saved |

Example:

```bash
MAX_STEPS=300 BATCH_SIZE=2 python scripts/train.py
```

## OOM? (out of memory)

Drop `BATCH_SIZE` to `1`, lower `MAX_SEQ_LENGTH`, and keep
`use_gradient_checkpointing="unsloth"` on. See the guide's OOM section.

## Your own dataset

`train.py` expects a dataset with a `"text"` column. Point `DATA_FILES` at it:

```bash
DATA_FILES=./data/my_data.jsonl python scripts/train.py
```

For a different format (Alpaca-style `prompt`/`completion`, or a HF dataset
name), adjust the `load_dataset` call in `train.py`.

## Inference

After training, load the base model with your saved LoRA adapters and generate:

```bash
source venv/bin/activate
python scripts/inference.py --prompt "Hello, how are you?"
# or point at a specific adapter dir:
python scripts/inference.py --output-dir outputs/qwen35-4b
```

`scripts/inference.py` loads `--output-dir` (default `outputs/qwen35-4b`) via
`FastLanguageModel.from_pretrained`, so it works for any adapter directory.

> Qwen3.5 is a vision-language model, so the inference script encodes text with
> the underlying text tokenizer (`tokenizer.tokenizer`) rather than the
> multimodal processor — a plain text prompt would otherwise be treated as an
> image.

## After training

- LoRA adapters are saved to `OUTPUT_DIR`.
- Merge or export (GGUF / vLLM) per the guide's "Saving / export" section.
- Use the **same chat template you trained with** at inference time.
