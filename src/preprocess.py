from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from utils import read_jsonl, setup_logging, stable_id, write_jsonl


def deterministic_bucket(doc_id: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_docs(
    docs: List[Dict[str, object]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []
    threshold_train = train_ratio
    threshold_val = train_ratio + val_ratio

    for doc in sorted(docs, key=lambda item: str(item.get("doc_id", ""))):
        doc_id = str(doc.get("doc_id", ""))
        bucket = deterministic_bucket(doc_id, seed)
        if bucket < threshold_train:
            train.append(doc)
        elif bucket < threshold_val:
            val.append(doc)
        else:
            test.append(doc)

    return train, val, test


def build_benchmark_questions(test_docs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    for doc in test_docs:
        title = str(doc.get("title", "")).strip() or "this document"
        section = str(doc.get("section", "")).strip()
        section_hint = f" in section '{section}'" if section else ""
        prompt = f"According to '{title}', what are the key policies or instructions{section_hint}?"
        reference_text = str(doc.get("text", ""))[:1200]
        questions.append(
            {
                "question_id": f"q-{stable_id(str(doc.get('doc_id')), prompt)}",
                "doc_id": doc.get("doc_id"),
                "dataset": doc.get("dataset"),
                "question": prompt,
                "reference_text": reference_text,
                "source_file": doc.get("source_file"),
            }
        )
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic train/val/test splits and benchmark set.")
    parser.add_argument("--input", default="data/processed/docs.jsonl", help="Input normalized docs.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    docs = list(read_jsonl(Path(args.input)))
    train, val, test = split_docs(docs, args.seed, args.train_ratio, args.val_ratio)
    benchmark_questions = build_benchmark_questions(test)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(out_dir / "benchmark_questions.jsonl", benchmark_questions)

    logging.info("Split counts => train=%s val=%s test=%s", len(train), len(val), len(test))
    logging.info("Benchmark questions written: %s", len(benchmark_questions))


if __name__ == "__main__":
    main()
