from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from utils import read_jsonl, setup_logging, write_jsonl


def build_instruction(title: str, section: str) -> str:
    section_part = f" in section '{section}'" if section else ""
    return (
        f"Summarize the key policies, requirements, or instructions from '{title}'{section_part}. "
        "Use only the provided document text."
    )


def build_question(title: str, section: str) -> str:
    if section:
        return f"According to '{title}', what are the key policies or instructions in section '{section}'?"
    return f"According to '{title}', what are the key policies or instructions?"


def format_record(doc: Dict[str, object], split_name: str, mode: str) -> Dict[str, object]:
    title = str(doc.get("title", "")).strip() or "Untitled Document"
    section = str(doc.get("section", "")).strip()
    text = str(doc.get("text", "")).strip()
    if mode == "qa":
        instruction = build_question(title=title, section=section)
    else:
        instruction = build_instruction(title=title, section=section)
    output = text
    formatted_text = f"Instruction:\n{instruction}\n\nResponse:\n{output}"
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "text": formatted_text,
        "source_doc_id": doc.get("doc_id"),
        "source_dataset": doc.get("dataset"),
        "source_title": title,
        "source_section": section,
        "source_split": split_name,
        "source_file": doc.get("source_file"),
        "task_style": mode,
    }


def convert_split(input_path: Path, split_name: str, mode: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for doc in read_jsonl(input_path):
        text = str(doc.get("text", "")).strip()
        if not text:
            continue
        rows.append(format_record(doc, split_name=split_name, mode=mode))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT-ready JSONL from existing document splits.")
    parser.add_argument("--train-input", default="data/processed/train.jsonl", help="Input train split JSONL.")
    parser.add_argument("--val-input", default="data/processed/val.jsonl", help="Input val split JSONL.")
    parser.add_argument("--train-output", default="data/processed/sft_train.jsonl", help="Output SFT train JSONL.")
    parser.add_argument("--val-output", default="data/processed/sft_val.jsonl", help="Output SFT val JSONL.")
    parser.add_argument(
        "--mode",
        default="doc_reconstruction",
        choices=["doc_reconstruction", "qa"],
        help="SFT task style. 'doc_reconstruction' preserves previous behavior; 'qa' uses question-style instructions.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    train_rows = convert_split(Path(args.train_input), split_name="train", mode=args.mode)
    val_rows = convert_split(Path(args.val_input), split_name="val", mode=args.mode)

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.val_output), val_rows)

    print(
        {
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "train_output": args.train_output,
            "val_output": args.val_output,
            "mode": args.mode,
            "note": "Derived from existing deterministic train/val splits; benchmark/test docs are not used.",
        }
    )


if __name__ == "__main__":
    main()
