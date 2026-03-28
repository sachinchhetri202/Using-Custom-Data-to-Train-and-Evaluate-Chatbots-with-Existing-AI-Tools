"""Evaluate a LoRA/QLoRA adapter on the curated benchmark.

Produces outputs compatible with the RAG prediction-review format so results
can be compared directly against the locked RAG baseline.

Usage (from repo root, GPU required for non-trivial throughput):
    python src/ft_eval.py \\
        --config configs/lora.yaml \\
        --adapter-dir outputs/ft/lora_tinyllama \\
        --benchmark data/processed/benchmark_questions_curated.jsonl \\
        --output-dir outputs/ft

Outputs:
    outputs/ft/ft_eval_lora_tinyllama_<ts>.jsonl      per-question trace
    outputs/ft/ft_eval_summary_lora_tinyllama_<ts>.json  compact summary
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Match rag_infer.py convention exactly.
SHORT_ANSWER_CHAR_THRESHOLD = 80


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def weak_reference_overlap(answer: str, reference_text: str) -> Optional[float]:
    """Token-overlap fraction of reference covered by answer tokens.

    Identical formula to rag_infer.py so scores are directly comparable.
    Returns None if either string is empty.
    """
    if not answer or not reference_text:
        return None
    ans_tokens = _token_set(answer)
    ref_tokens = _token_set(reference_text)
    if not ref_tokens:
        return None
    return len(ans_tokens & ref_tokens) / len(ref_tokens)


def generate_answer(model: Any, tokenizer: Any, question: str, max_new_tokens: int, device: str) -> str:
    """Run greedy decode and return only the Response portion."""
    import torch

    prompt = f"Instruction:\n{question}\n\nResponse:\n"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    marker = "Response:\n"
    idx = full_text.find(marker)
    return full_text[idx + len(marker):].strip() if idx != -1 else full_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA/QLoRA adapter on the curated benchmark."
    )
    parser.add_argument("--config", required=True, help="Path to LoRA/QLoRA config YAML.")
    parser.add_argument("--adapter-dir", required=True, help="Directory with trained adapter weights.")
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark_questions_curated.jsonl",
        help="Benchmark file (default: curated subset).",
    )
    parser.add_argument("--output-dir", default="outputs/ft", help="Output directory.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max generation tokens.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N questions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    adapter_path = Path(args.adapter_dir).resolve()
    benchmark_path = Path(args.benchmark)
    output_dir = Path(args.output_dir)

    # Resolve output paths early so they are visible before any model loading.
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = adapter_path.name
    ts = now_ts()
    detail_path = output_dir / f"ft_eval_{run_name}_{ts}.jsonl"
    summary_path = output_dir / f"ft_eval_summary_{run_name}_{ts}.json"

    print(f"[ft_eval] adapter    : {adapter_path}")
    print(f"[ft_eval] benchmark  : {benchmark_path.resolve()}")
    print(f"[ft_eval] output_dir : {output_dir.resolve()}")
    print(f"[ft_eval] detail     : {detail_path}")
    print(f"[ft_eval] summary    : {summary_path}")

    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter not found at: {adapter_path}\n"
            "Run training first: python src/finetune.py --config <config.yaml>"
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model"]["name"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ft_eval] device     : {device}  |  model: {model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    benchmark = read_jsonl(benchmark_path)
    if args.limit:
        benchmark = benchmark[: args.limit]
    if not benchmark:
        raise ValueError(f"No items found in {benchmark_path}")
    print(f"[ft_eval] questions  : {len(benchmark)}")

    overlap_values: List[float] = []
    short_count = 0
    empty_count = 0
    records: List[Dict[str, Any]] = []

    for i, item in enumerate(benchmark, start=1):
        question = str(item.get("question", "")).strip()
        reference_text = str(item.get("reference_text", ""))

        answer = generate_answer(model, tokenizer, question, args.max_new_tokens, device)

        overlap = weak_reference_overlap(answer, reference_text)
        if overlap is not None:
            overlap_values.append(overlap)

        answer_len = len(answer)
        answer_empty = answer_len == 0
        answer_short = answer_len > 0 and answer_len < SHORT_ANSWER_CHAR_THRESHOLD
        if answer_empty:
            empty_count += 1
        if answer_short:
            short_count += 1

        record: Dict[str, Any] = {
            "question_id": item.get("question_id"),
            "dataset": item.get("dataset"),
            "question": question,
            "expected_doc_id": item.get("doc_id"),
            "expected_source_file": item.get("source_file"),
            "reference_text": reference_text,
            "generated_answer": answer,
            "metrics": {
                "weak_reference_overlap": round(overlap, 4) if overlap is not None else None,
                "answer_empty": answer_empty,
                "answer_short": answer_short,
                "answer_length_chars": answer_len,
                "short_answer_char_threshold": SHORT_ANSWER_CHAR_THRESHOLD,
            },
            # Placeholders matching RAG prediction-review format for manual review parity.
            "human_judgment_answer_supported": None,
            "human_judgment_answer_quality": None,
            "human_review_notes": "",
            "run_info": {
                "adapter_dir": str(adapter_path),
                "model_name": model_name,
                "max_new_tokens": args.max_new_tokens,
                "retrieval_context_injected": False,
            },
        }
        records.append(record)
        overlap_str = f"{overlap:.3f}" if overlap is not None else "n/a"
        print(
            f"[{i}/{len(benchmark)}] overlap={overlap_str}"
            f"  short={answer_short}  {question[:70]}",
            flush=True,
        )

    print(f"[ft_eval] writing detail  -> {detail_path}", flush=True)
    with detail_path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ft_eval] detail written  OK ({detail_path.stat().st_size:,} bytes)")

    n = len(records)
    avg_overlap = round(sum(overlap_values) / len(overlap_values), 4) if overlap_values else None
    summary: Dict[str, Any] = {
        "timestamp_utc": ts,
        "run_name": run_name,
        "adapter_dir": str(adapter_path),
        "model_name": model_name,
        "benchmark_path": str(args.benchmark),
        "n_questions": n,
        "average_weak_reference_overlap": avg_overlap,
        "answer_empty_count": empty_count,
        "answer_short_count": short_count,
        "short_answer_char_threshold": SHORT_ANSWER_CHAR_THRESHOLD,
        "retrieval_context_injected": False,
        "human_judgment_answer_supported_count": None,
        "detail_path": str(detail_path),
        "notes": [
            "weak_reference_overlap is a token-overlap fraction, not a correctness proof.",
            "Matches the formula used in rag_infer.py for direct comparison with RAG baseline.",
            "No retrieval context was injected; model answers from fine-tuned weights only.",
            "human_judgment_* fields are placeholders for manual review.",
        ],
    }

    print(f"[ft_eval] writing summary -> {summary_path}", flush=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[ft_eval] summary written OK ({summary_path.stat().st_size:,} bytes)")

    print(f"\nDone. {n} questions evaluated.")
    print(f"  avg_weak_reference_overlap : {avg_overlap}")
    print(f"  answer_short_count         : {short_count}")
    print(f"  answer_empty_count         : {empty_count}")
    print(f"\n  Per-question: {detail_path}")
    print(f"  Summary     : {summary_path}")


if __name__ == "__main__":
    main()
