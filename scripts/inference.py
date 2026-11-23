"""
Inference test for a trained LoRA adapter on Qwen3.5-4B.

Usage:
    source venv/bin/activate
    python scripts/inference.py            # uses OUTPUT_DIR default
    python scripts/inference.py --prompt "Hello, how are you?"
    python scripts/inference.py --output-dir outputs/qwen35-4b

Loads the base model with the saved LoRA adapters attached, formats the
prompt with the model's own chat template, and generates a response.
"""
import os
import argparse
from unsloth import FastLanguageModel

DEFAULT_PROMPT = "Hello, how are you?"


def main():
    parser = argparse.ArgumentParser(description="Infer with a trained LoRA adapter")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "outputs/qwen35-4b"),
        help="Directory containing the saved LoRA adapters",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    # Point straight at the saved adapter dir. Unsloth reads the LoRA
    # adapter_config.json there and loads base + adapter together.
    print(f"Loading base model + adapters from {args.output_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.output_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Qwen3.5 is a vision-language model: the returned object is a
    # Qwen3VLProcessor that tries to treat a plain string as an image.
    # For text-only inference, encode with the underlying text tokenizer.
    text_tokenizer = tokenizer.tokenizer

    # Format the prompt with the model's own chat template.
    messages = [{"role": "user", "content": args.prompt}]
    chat = text_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = text_tokenizer([chat], return_tensors="pt").to("cuda")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.6,
        top_p=0.9,
    )
    # Strip the input prompt, keep only the generated text.
    generated = output_ids[0][len(inputs["input_ids"][0]):]
    print("\n=== PROMPT ===")
    print(args.prompt)
    print("\n=== RESPONSE ===")
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
