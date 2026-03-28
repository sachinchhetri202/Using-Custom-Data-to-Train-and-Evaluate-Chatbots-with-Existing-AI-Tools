from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with a LoRA/QLoRA adapter checkpoint.")
    parser.add_argument("--config", required=True, help="Path to LoRA/QLoRA config.")
    parser.add_argument("--adapter-dir", required=True, help="Directory containing trained adapter checkpoint.")
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model"]["name"])
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    formatted = f"Instruction:\n{args.prompt}\n\nResponse:\n"
    encoded = tokenizer(formatted, return_tensors="pt")
    outputs = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
