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

    # gradient_checkpointing_kwargs added to TrainingArguments in transformers ~4.36.
    # Required for PEFT + gradient checkpointing (QLoRA) to set use_reentrant=False.
    supports_gc_kwargs = "gradient_checkpointing_kwargs" in cfg_params

    api = {
        "trl_version": trl.__version__,
        "transformers_version": transformers.__version__,
        "eval_strategy_arg": eval_strategy_key or "none (disabled)",
        "dataset_text_field_in": "SFTConfig" if text_field_in_config else "SFTTrainer",
        "trainer_tokenizer_arg": tokenizer_arg,
        "text_field_in_config": text_field_in_config,
        "eval_strategy_key": eval_strategy_key,
        "tokenizer_arg": tokenizer_arg,
        "supports_gc_kwargs": supports_gc_kwargs,
    }
    return api


def run_training(config: Dict[str, Any]) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, cast_mixed_precision_params
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

    if qlora_enabled:
        _tc = config["training"]
        _qc = config.get("quantization", {})
        _fp16 = bool(_tc.get("fp16", False))
        _bf16 = bool(_tc.get("bf16", False))
        _cdt  = str(_qc.get("bnb_4bit_compute_dtype", "float16"))
        _mdtype = "bfloat16" if _bf16 else "float16"
        import inspect as _insp_diag
        _diag_fp_params = set(_insp_diag.signature(AutoModelForCausalLM.from_pretrained).parameters)
        _diag_dtype_kwarg = "dtype" if "dtype" in _diag_fp_params else "torch_dtype"
        print({
            "qlora_precision": {
                "fp16": _fp16,
                "bf16": _bf16,
                "bnb_4bit_compute_dtype": _cdt,
                "grad_scaling_expected": _fp16 and not _bf16,
                "model_dtype_kwarg": _diag_dtype_kwarg,
                "model_dtype_value": _mdtype,
                "note": (
                    "OK — fp16+float16 compute dtype (T4-safe)"
                    if (_fp16 and not _bf16 and _cdt == "float16")
                    else "WARN — bf16 mode requires A100+" if _bf16
                    else "CHECK — non-standard combination"
                ),
            }
        })

    model_kwargs: Dict[str, Any] = {}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        # Modern LLMs (including TinyLlama) store torch_dtype: bfloat16 in
        # their config.json. Without an explicit override, from_pretrained
        # loads non-quantized layers (embeddings, layer norms, LM head) in
        # bf16, and PEFT LoRA adapters inherit that dtype. When fp16=True
        # activates AMP GradScaler, it crashes unscaling bf16 adapter grads:
        # "not implemented for BFloat16". Forcing the dtype to match the
        # training precision keeps adapter weights and GradScaler aligned.
        # Transformers 5.x deprecates `torch_dtype` in favor of `dtype`.
        # Prefer `dtype` on 5.x, keep `torch_dtype` for older versions.
        import transformers as _transformers
        _tf_major = int(str(_transformers.__version__).split(".", 1)[0])
        _dtype_kwarg = "dtype" if _tf_major >= 5 else "torch_dtype"
        _use_bf16 = bool(config["training"].get("bf16", False))
        model_kwargs[_dtype_kwarg] = torch.bfloat16 if _use_bf16 else torch.float16

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
    model_context_limit = 2048
    requested_max_length = int(train_cfg.get("max_length", train_cfg.get("max_seq_length", model_context_limit)))
    max_seq_length = min(requested_max_length, model_context_limit)
    if requested_max_length > model_context_limit:
        print(
            {
                "sequence_length": {
                    "requested_max_length": requested_max_length,
                    "applied_max_length": max_seq_length,
                    "model_context_limit": model_context_limit,
                    "note": "Requested length exceeds model context limit; clamped.",
                }
            }
        )

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
        "max_length": max_seq_length,
        "report_to": train_cfg.get("report_to", []),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
    }

    # PEFT + gradient checkpointing requires use_reentrant=False to avoid
    # "None of the inputs have requires_grad=True" errors with 4-bit (QLoRA) models.
    if sft_kwargs["gradient_checkpointing"] and api["supports_gc_kwargs"]:
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

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
    run_fp16 = bool(train_cfg.get("fp16", True))
    run_bf16 = bool(train_cfg.get("bf16", False))
    if qlora_enabled and run_fp16 and not run_bf16:
        # Keep trainable adapters in full precision for AMP safety while casting
        # frozen/base weights to fp16 mixed precision.
        cast_mixed_precision_params(trainer.model, dtype=torch.float16)

    trainable_dtype_counts: Dict[str, int] = {}
    for param in trainer.model.parameters():
        if param.requires_grad:
            key = str(param.dtype).replace("torch.", "")
            trainable_dtype_counts[key] = trainable_dtype_counts.get(key, 0) + 1

    print(
        {
            "startup_precision_diagnostics": {
                "fp16": run_fp16,
                "bf16": run_bf16,
                "bnb_4bit_compute_dtype": compute_dtype_name if qlora_enabled else None,
                "trainable_param_dtype_counts": trainable_dtype_counts,
                "max_seq_length": max_seq_length,
                "model_context_limit": model_context_limit,
            }
        }
    )

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
