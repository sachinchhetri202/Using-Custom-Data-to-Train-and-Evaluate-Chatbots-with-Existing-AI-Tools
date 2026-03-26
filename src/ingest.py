from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docx import Document  # type: ignore[import-untyped]
from pypdf import PdfReader

from utils import clean_text, dedupe_preserve_order, is_navigation_noise, read_jsonl, setup_logging, stable_id, write_jsonl


def dataset1_processed_quality(processed_path: Path) -> Dict[str, Any]:
    stats = {"records": 0, "empty_metadata": 0, "nav_like": 0}
    if not processed_path.exists():
        return stats

    for record in read_jsonl(processed_path):
        stats["records"] += 1
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict) and not metadata:
            stats["empty_metadata"] += 1
        text = clean_text(str(record.get("text", "")))
        if is_navigation_noise(text):
            stats["nav_like"] += 1
    return stats


def extract_block_text(block: Dict[str, Any]) -> str:
    block_type = block.get("type", "")
    if block_type == "paragraph":
        return clean_text(str(block.get("text", "")))
    if block_type == "heading":
        return clean_text(str(block.get("text", "")))
    if block_type == "list":
        if block.get("as_text"):
            return clean_text(str(block["as_text"]))
        items = block.get("items") or []
        return clean_text("\n".join(str(item) for item in items))
    return ""


def ingest_dataset1(raw_blocks_path: Path) -> List[Dict[str, Any]]:
    section_texts: Dict[Tuple[str, str, str, str, str], List[str]] = defaultdict(list)
    section_headings: Dict[Tuple[str, str, str, str, str], str] = {}

    for block in read_jsonl(raw_blocks_path):
        base_doc_id = str(block.get("doc_id", "")).strip()
        source_url = str(block.get("url", "")).strip()
        title = clean_text(str(block.get("title", "")))
        language = str(block.get("lang", "")).strip() or None
        section_id = str(block.get("section_id", "")).strip() or "root"

        key = (base_doc_id, section_id, source_url, title, language or "")
        text = extract_block_text(block)
        if not text or is_navigation_noise(text):
            continue

        if block.get("type") == "heading" and section_id != "root":
            section_headings[key] = text

        section_texts[key].append(text)

    records: List[Dict[str, Any]] = []
    for key in sorted(section_texts):
        base_doc_id, section_id, source_url, title, language = key
        parts = dedupe_preserve_order(section_texts[key])
        text = clean_text("\n\n".join(parts))
        if len(text) < 80:
            continue
        section_name = section_headings.get(key, section_id if section_id != "root" else "")
        record_doc_id = f"d1-{stable_id(base_doc_id, section_id, source_url)}"
        records.append(
            {
                "dataset": "dataset1",
                "doc_id": record_doc_id,
                "source_file": "data/raw/dataset1/blocks.jsonl",
                "source_type": "web_blocks_jsonl",
                "title": title or "Untitled Dataset 1 Document",
                "section": section_name,
                "source_url": source_url or None,
                "language": language or None,
                "text": text,
            }
        )
    return records


def read_docx_text(path: Path) -> str:
    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return clean_text("\n".join(paragraphs))


def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(page_text)
    return clean_text("\n".join(pages))


def ingest_dataset2(raw_dataset2_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    candidate_files = sorted(
        [*raw_dataset2_dir.rglob("*.docx"), *raw_dataset2_dir.rglob("*.pdf")]
    )

    for file_path in candidate_files:
        rel_path = file_path.relative_to(raw_dataset2_dir)
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            text = read_docx_text(file_path)
        elif suffix == ".pdf":
            text = read_pdf_text(file_path)
        else:
            continue

        if len(text) < 80:
            logging.info("Skipping short source: %s", rel_path)
            continue
        if is_navigation_noise(text):
            logging.info("Skipping likely boilerplate source: %s", rel_path)
            continue

        parent = rel_path.parent.as_posix()
        section = "" if parent == "." else parent
        records.append(
            {
                "dataset": "dataset2",
                "doc_id": f"d2-{stable_id(str(rel_path))}",
                "source_file": f"data/raw/dataset2/{rel_path.as_posix()}",
                "source_type": suffix.lstrip("."),
                "title": file_path.stem,
                "section": section,
                "source_url": None,
                "language": None,
                "text": text,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and normalize Dataset 1 + Dataset 2.")
    parser.add_argument(
        "--dataset1-blocks",
        default="data/raw/dataset1/blocks.jsonl",
        help="Path to Dataset 1 raw blocks file.",
    )
    parser.add_argument(
        "--dataset1-processed-check",
        default="data/raw/dataset1/processed/processed_blocks.jsonl",
        help="Path to Dataset 1 processed file to inspect for quality evidence.",
    )
    parser.add_argument(
        "--dataset2-dir",
        default="data/raw/dataset2",
        help="Path to Dataset 2 raw directory.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/docs.jsonl",
        help="Output normalized docs JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    dataset1_blocks = Path(args.dataset1_blocks)
    dataset1_processed_check = Path(args.dataset1_processed_check)
    dataset2_dir = Path(args.dataset2_dir)
    output_path = Path(args.output)

    quality = dataset1_processed_quality(dataset1_processed_check)
    if quality["records"] > 0:
        logging.info(
            "Dataset 1 processed quality check: records=%s empty_metadata=%s nav_like=%s",
            quality["records"],
            quality["empty_metadata"],
            quality["nav_like"],
        )

    d1_records = ingest_dataset1(dataset1_blocks)
    d2_records = ingest_dataset2(dataset2_dir)
    all_records = sorted(d1_records + d2_records, key=lambda row: (row["dataset"], row["doc_id"]))

    write_jsonl(output_path, all_records)
    logging.info("Wrote %s normalized records to %s", len(all_records), output_path)
    logging.info("Dataset 1 records: %s | Dataset 2 records: %s", len(d1_records), len(d2_records))


if __name__ == "__main__":
    main()
