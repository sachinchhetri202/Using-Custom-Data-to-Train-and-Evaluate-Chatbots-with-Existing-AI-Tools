from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from utils import is_navigation_noise, setup_logging


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def chunk_batches(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def normalize_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "doc_id": str(chunk.get("doc_id", "")),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "dataset": chunk.get("dataset"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "source_file": chunk.get("source_file"),
        "source_type": chunk.get("source_type"),
        "source_url": chunk.get("source_url"),
        "language": chunk.get("language"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma vector index from processed chunks.")
    parser.add_argument("--config", default="configs/rag.yaml", help="Path to RAG config file.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))

    chunks_path = Path(config["index"]["chunks_path"])
    chroma_path = Path(config["index"]["chroma_path"])
    collection_name = str(config["index"]["collection_name"])
    batch_size = int(config["index"].get("batch_size", 64))
    rebuild = bool(config["index"].get("rebuild", True))
    filter_navigation_noise = bool(config["index"].get("filter_navigation_noise", True))
    embedding_model_name = str(config["embedding"]["model_name"])

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    if filter_navigation_noise:
        before = len(chunks)
        chunks = [row for row in chunks if not is_navigation_noise(str(row.get("text", "")))]
        logging.info("Filtered navigation-noise chunks: %s removed, %s kept", before - len(chunks), len(chunks))

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    if rebuild:
        try:
            client.delete_collection(name=collection_name)
            logging.info("Deleted existing collection: %s", collection_name)
        except Exception:
            logging.info("No existing collection to delete: %s", collection_name)

    collection = client.get_or_create_collection(name=collection_name)
    model = SentenceTransformer(embedding_model_name)
    logging.info("Loaded embedding model: %s", embedding_model_name)
    logging.info("Indexing %s chunks into collection '%s'", len(chunks), collection_name)

    total_added = 0
    for batch in chunk_batches(chunks, batch_size):
        texts = [str(row.get("text", "")) for row in batch]
        ids = [str(row.get("chunk_id", "")) for row in batch]
        metadatas = [normalize_metadata(row) for row in batch]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        total_added += len(batch)

    logging.info("Index build complete. Added %s chunks.", total_added)


if __name__ == "__main__":
    main()
