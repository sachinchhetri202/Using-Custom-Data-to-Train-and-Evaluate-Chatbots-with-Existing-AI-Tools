from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from utils import clean_text, read_jsonl, setup_logging, stable_id, write_jsonl


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if overlap > 0 and chunks:
            tail = chunks[-1][-overlap:]
            current = clean_text(f"{tail}\n\n{paragraph}")
            if len(current) > chunk_size:
                current = paragraph
        else:
            current = paragraph

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retrieval-ready chunks from docs.jsonl.")
    parser.add_argument("--input", default="data/processed/docs.jsonl", help="Input docs JSONL path.")
    parser.add_argument("--output", default="data/processed/chunks.jsonl", help="Output chunks JSONL path.")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=120, help="Overlap between chunks in characters.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    chunk_records: List[Dict[str, object]] = []
    for doc in read_jsonl(input_path):
        chunks = split_into_chunks(str(doc.get("text", "")), args.chunk_size, args.overlap)
        for idx, chunk_text in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"ch-{stable_id(str(doc.get('doc_id')), str(idx), chunk_text)}",
                    "doc_id": doc.get("doc_id"),
                    "chunk_index": idx,
                    "dataset": doc.get("dataset"),
                    "title": doc.get("title"),
                    "section": doc.get("section"),
                    "source_file": doc.get("source_file"),
                    "source_type": doc.get("source_type"),
                    "source_url": doc.get("source_url"),
                    "language": doc.get("language"),
                    "text": chunk_text,
                }
            )

    write_jsonl(output_path, chunk_records)
    logging.info("Wrote %s chunks to %s", len(chunk_records), output_path)


if __name__ == "__main__":
    main()
