from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def first_jsonl_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared LoRA/QLoRA SFT training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to training config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset only, no model training.")
    return parser.parse_args()


def validate_and_preview(config: Dict[str, Any]) -> Dict[str, Any]:
    train_file = Path(config["data"]["train_file"])
    val_file = Path(config["data"]["val_file"])
    text_field = str(config["data"].get("text_field", "text"))
    train_row = first_jsonl_row(train_file)
    val_row = first_jsonl_row(val_file)

    if train_row is None:
        raise ValueError(f"No rows found in {train_file}")
    if val_row is None:
        raise ValueError(f"No rows found in {val_file}")
    if text_field not in train_row:
        raise ValueError(f"Configured text_field '{text_field}' not found in train row keys: {sorted(train_row.keys())}")

    return {
        "train_file": str(train_file),
        "val_file": str(val_file),
        "text_field": text_field,
        "train_keys": sorted(train_row.keys()),
        "val_keys": sorted(val_row.keys()),
        "sample_text_preview": str(train_row[text_field])[:240],
    }


def _detect_trl_api() -> Dict[str, Any]:
    """Inspect installed TRL/transformers to determine which argument names are live."""
    import inspect
    import trl
    import transformers
    from trl import SFTConfig, SFTTrainer

    cfg_params = set(inspect.signature(SFTConfig).parameters)
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)

    # Eval-strategy argument name changed in TRL ~0.10 / transformers ~4.46.
    if "eval_strategy" in cfg_params:
        eval_strategy_key: Optional[str] = "eval_strategy"
    elif "evaluation_strategy" in cfg_params:
        eval_strategy_key = "evaluation_strategy"
    else:
        eval_strategy_key = None  # fall back: no per-step eval

    # dataset_text_field moved from SFTTrainer into SFTConfig in TRL ~0.10.
    text_field_in_config = "dataset_text_field" in cfg_params

    # tokenizer= was replaced by processing_class= in SFTTrainer in TRL ~0.11.
    # Prefer processing_class when available; fall back to tokenizer.
    tokenizer_arg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    api = {
        "trl_version": trl.__version__,
        "transformers_version": transformers.__version__,
        "eval_strategy_arg": eval_strategy_key or "none (disabled)",
        "dataset_text_field_in": "SFTConfig" if text_field_in_config else "SFTTrainer",
        "trainer_tokenizer_arg": tokenizer_arg,
        "text_field_in_config": text_field_in_config,
        "eval_strategy_key": eval_strategy_key,
        "tokenizer_arg": tokenizer_arg,
    }
    return api


def run_training(config: Dict[str, Any]) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    api = _detect_trl_api()
    print({"trl_api_probe": {k: v for k, v in api.items() if not k.startswith("_")}})

    model_name = str(config["model"]["name"])
    train_file = str(config["data"]["train_file"])
    val_file = str(config["data"]["val_file"])
    text_field = str(config["data"].get("text_field", "text"))
    output_dir = str(config["training"]["output_dir"])

    quant_cfg = dict(config.get("quantization", {}))
    qlora_enabled = bool(quant_cfg.get("enabled", False))
    if qlora_enabled and not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU environment with bitsandbytes support.")

    compute_dtype_name = str(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))
    compute_dtype = getattr(torch, compute_dtype_name)
    bnb_config = None
    if qlora_enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs: Dict[str, Any] = {}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    max_train_samples = int(config["data"].get("max_train_samples", 0))
    max_eval_samples = int(config["data"].get("max_eval_samples", 0))
    if max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    adapter_cfg = config["adapter"]
    peft_config = LoraConfig(
        r=int(adapter_cfg.get("r", 16)),
        lora_alpha=int(adapter_cfg.get("lora_alpha", 32)),
        lora_dropout=float(adapter_cfg.get("lora_dropout", 0.05)),
        target_modules=adapter_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=str(adapter_cfg.get("bias", "none")),
        task_type=str(adapter_cfg.get("task_type", "CAUSAL_LM")),
    )

    train_cfg = config["training"]

    # Build SFTConfig kwargs dynamically to stay compatible across TRL versions.
    sft_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 4)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "save_strategy": str(train_cfg.get("save_strategy", "steps")),
        "max_length": int(train_cfg.get("max_length", 1024)),
        "report_to": train_cfg.get("report_to", []),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
    }

    # Eval strategy: use whichever argument name this TRL version accepts.
    # Read from YAML under either key name for backward compat.
    eval_strategy_key = api["eval_strategy_key"]
    if eval_strategy_key is not None:
        eval_strategy_value = str(
            train_cfg.get("eval_strategy") or train_cfg.get("evaluation_strategy", "steps")
        )
        sft_kwargs[eval_strategy_key] = eval_strategy_value
        sft_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 100))

    # dataset_text_field: belongs in SFTConfig on newer TRL, SFTTrainer on older TRL.
    if api["text_field_in_config"]:
        sft_kwargs["dataset_text_field"] = text_field

    sft_args = SFTConfig(**sft_kwargs)

    # Build SFTTrainer kwargs dynamically.
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        api["tokenizer_arg"]: tokenizer,
    }
    if not api["text_field_in_config"]:
        trainer_kwargs["dataset_text_field"] = text_field

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    seed = int(config.get("runtime", {}).get("seed", 42))
    set_seed(seed)

    preview = validate_and_preview(config)
    print({"config": args.config, "preview": preview})

    if args.dry_run:
        print({"status": "dry_run_ok", "note": "No training executed."})
        return

    run_training(config)
    print({"status": "train_complete", "output_dir": config["training"]["output_dir"]})


if __name__ == "__main__":
    main()
