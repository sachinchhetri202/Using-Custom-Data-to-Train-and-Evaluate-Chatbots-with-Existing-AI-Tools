from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from utils import ensure_parent, is_navigation_noise, setup_logging


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


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def provisional_answerability(reference_text: str) -> Tuple[str, str]:
    text = (reference_text or "").strip()
    if not text:
        return "unclear", "empty_reference_text"
    if is_navigation_noise(text):
        return "unclear", "reference_text_navigation_like"
    if len(text) < 80:
        return "unclear", "reference_text_too_short"
    return "likely_answerable", "has_nontrivial_reference_text"


def extract_title_hint(question: str) -> Optional[str]:
    match = re.search(r"According to '([^']+)'", question)
    if not match:
        return None
    title = match.group(1).strip()
    return title or None


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_where_filter(dataset_filter: Optional[str], title_hint: Optional[str]) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    if dataset_filter:
        clauses.append({"dataset": dataset_filter})
    if title_hint:
        clauses.append({"title": title_hint})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def dense_retrieve(
    collection: Any,
    embed_model: SentenceTransformer,
    query: str,
    n_results: int,
    where: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    embedding = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    rows: List[Dict[str, Any]] = []
    for rank, (doc_text, metadata, distance) in enumerate(zip(docs, metas, dists), start=1):
        rows.append(
            {
                "rank": rank,
                "distance": float(distance),
                "metadata": metadata or {},
                "text": doc_text,
            }
        )
    return rows


def rerank_by_overlap(rows: List[Dict[str, Any]], query: str, overlap_weight: float) -> List[Dict[str, Any]]:
    q_tokens = set(tokenize(query))
    rescored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        text = str(row.get("text", ""))
        meta = row.get("metadata", {}) or {}
        title = str(meta.get("title", ""))
        section = str(meta.get("section", ""))
        tokens = set(tokenize(text)) | set(tokenize(title)) | set(tokenize(section))
        overlap = len(q_tokens & tokens)
        distance = float(row.get("distance", 0.0))
        score = (-distance) + (overlap_weight * overlap)
        merged = dict(row)
        merged["rerank_overlap"] = overlap
        merged["rerank_score"] = score
        rescored.append((score, merged))

    rescored.sort(key=lambda item: item[0], reverse=True)
    output: List[Dict[str, Any]] = []
    for new_rank, (_, row) in enumerate(rescored, start=1):
        row["rank"] = new_rank
        output.append(row)
    return output


def retrieve_with_variant(
    collection: Any,
    embed_model: SentenceTransformer,
    query: str,
    top_k: int,
    dataset_filter: Optional[str],
    variant: str,
    candidate_k: int,
    overlap_weight: float,
) -> Dict[str, Any]:
    where: Optional[Dict[str, Any]] = build_where_filter(dataset_filter, None)

    title_hint = extract_title_hint(query)
    if variant == "title_filter" and title_hint:
        where = build_where_filter(dataset_filter, title_hint)

    if variant == "rerank_overlap":
        before_rows = dense_retrieve(
            collection=collection,
            embed_model=embed_model,
            query=query,
            n_results=max(candidate_k, top_k),
            where=where,
        )
        after_rows = rerank_by_overlap(before_rows, query, overlap_weight)[:top_k]
        return {
            "rows": after_rows,
            "extra": {
                "pre_rerank_rows": before_rows,
                "title_hint_used": title_hint,
                "where_filter": where,
            },
        }

    rows = dense_retrieve(
        collection=collection,
        embed_model=embed_model,
        query=query,
        n_results=top_k,
        where=where,
    )
    return {"rows": rows, "extra": {"title_hint_used": title_hint, "where_filter": where}}


def first_rank_match(retrieval_rows: List[Dict[str, Any]], key: str, expected_value: str) -> Optional[int]:
    if not expected_value:
        return None
    for row in retrieval_rows:
        got = str(row.get("metadata", {}).get(key, ""))
        if got == expected_value:
            return int(row["rank"])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval-only benchmark evaluation.")
    parser.add_argument("--config", default="configs/rag.yaml", help="Path to RAG config.")
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark_questions.jsonl",
        help="Benchmark questions file.",
    )
    parser.add_argument(
        "--test-docs",
        default="data/processed/test.jsonl",
        help="Held-out test document file for benchmark coverage checks.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rag",
        help="Directory for retrieval evaluation outputs.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k retrieval.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of benchmark questions to evaluate.",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "title_filter", "rerank_overlap"],
        default="baseline",
        help="Retrieval variant to evaluate.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate set size for rerank_overlap variant.",
    )
    parser.add_argument(
        "--overlap-weight",
        type=float,
        default=None,
        help="Token overlap weight for rerank_overlap.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))

    benchmark_path = Path(args.benchmark)
    test_docs_path = Path(args.test_docs)
    output_dir = Path(args.output_dir)
    top_k = int(args.top_k or config["retrieval"]["top_k"])
    variant = str(args.variant)
    candidate_k = int(args.candidate_k or config.get("retrieval", {}).get("candidate_k", 20))
    overlap_weight = float(args.overlap_weight or config.get("retrieval", {}).get("overlap_weight", 0.08))

    benchmark = read_jsonl(benchmark_path)
    test_docs = read_jsonl(test_docs_path)
    test_doc_ids = {str(row.get("doc_id", "")).strip() for row in test_docs}
    if args.limit is not None:
        benchmark = benchmark[: args.limit]
    if not benchmark:
        raise ValueError(f"No benchmark items found in {benchmark_path}")

    chroma_path = Path(config["index"]["chroma_path"])
    collection_name = str(config["index"]["collection_name"])
    embed_model_name = str(config["embedding"]["model_name"])

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    embed_model = SentenceTransformer(embed_model_name)
    logging.info("Loaded retrieval collection '%s'", collection_name)

    ts = now_ts()
    detailed_path = output_dir / f"retrieval_eval_{variant}_{ts}.jsonl"
    summary_path = output_dir / f"retrieval_eval_summary_{variant}_{ts}.json"
    ensure_parent(detailed_path)

    summary: Dict[str, Any] = {
        "timestamp_utc": ts,
        "benchmark_path": str(benchmark_path),
        "variant": variant,
        "benchmark_items_total": len(benchmark),
        "top_k": top_k,
        "benchmark_coverage": {
            "test_docs_path": str(test_docs_path),
            "benchmark_items_with_doc_id_in_test": 0,
            "benchmark_items_missing_doc_id_in_test": 0,
        },
        "provisional_answerability": {
            "likely_answerable_count": 0,
            "unclear_count": 0,
            "unclear_reasons": {},
        },
        "scoreability": {
            "with_expected_doc_id": 0,
            "with_expected_source_file": 0,
            "likely_answerable_with_expected_doc_id": 0,
        },
        "retrieval_metrics": {
            "doc_id_hit_at_k": None,
            "source_file_hit_at_k": None,
            "doc_id_hit_at_k_likely_answerable_subset": None,
        },
        "variant_analysis": {
            "rerank_doc_id_rank_improved_count": 0,
            "rerank_doc_id_rank_worsened_count": 0,
            "rerank_doc_id_rank_unchanged_count": 0,
            "title_hint_detected_count": 0,
            "title_hint_filter_applied_count": 0,
        },
        "notes": [
            "benchmark_questions.jsonl remains provisional scaffold unless manually reviewed.",
            "metrics are retrieval-only and computed from executed runs.",
        ],
    }

    doc_hits = 0
    doc_total = 0
    src_hits = 0
    src_total = 0
    doc_hits_answerable = 0
    doc_total_answerable = 0

    with detailed_path.open("w", encoding="utf-8") as handle:
        for item in benchmark:
            question = str(item.get("question", "")).strip()
            dataset = str(item.get("dataset", "")).strip() or None
            expected_doc_id = str(item.get("doc_id", "")).strip()
            expected_source_file = str(item.get("source_file", "")).strip()
            reference_text = str(item.get("reference_text", ""))

            doc_id_in_test = expected_doc_id in test_doc_ids if expected_doc_id else False
            if expected_doc_id:
                if doc_id_in_test:
                    summary["benchmark_coverage"]["benchmark_items_with_doc_id_in_test"] += 1
                else:
                    summary["benchmark_coverage"]["benchmark_items_missing_doc_id_in_test"] += 1

            answerability, answerability_reason = provisional_answerability(reference_text)
            if answerability == "likely_answerable":
                summary["provisional_answerability"]["likely_answerable_count"] += 1
            else:
                summary["provisional_answerability"]["unclear_count"] += 1
                unclear_reasons = summary["provisional_answerability"]["unclear_reasons"]
                unclear_reasons[answerability_reason] = int(unclear_reasons.get(answerability_reason, 0)) + 1

            if expected_doc_id:
                summary["scoreability"]["with_expected_doc_id"] += 1
                doc_total += 1
                if answerability == "likely_answerable":
                    summary["scoreability"]["likely_answerable_with_expected_doc_id"] += 1
                    doc_total_answerable += 1

            if expected_source_file:
                summary["scoreability"]["with_expected_source_file"] += 1
                src_total += 1

            variant_result = retrieve_with_variant(
                collection=collection,
                embed_model=embed_model,
                query=question,
                top_k=top_k,
                dataset_filter=dataset,
                variant=variant,
                candidate_k=candidate_k,
                overlap_weight=overlap_weight,
            )
            retrieval_rows = variant_result["rows"]
            extra = variant_result["extra"]

            doc_rank = first_rank_match(retrieval_rows, "doc_id", expected_doc_id)
            src_rank = first_rank_match(retrieval_rows, "source_file", expected_source_file)
            doc_hit = doc_rank is not None
            src_hit = src_rank is not None

            title_hint_used = extra.get("title_hint_used")
            if title_hint_used:
                summary["variant_analysis"]["title_hint_detected_count"] += 1
                if variant == "title_filter":
                    summary["variant_analysis"]["title_hint_filter_applied_count"] += 1

            pre_rerank_rows = extra.get("pre_rerank_rows", [])
            pre_doc_rank = (
                first_rank_match(pre_rerank_rows, "doc_id", expected_doc_id) if pre_rerank_rows else None
            )
            if variant == "rerank_overlap" and pre_doc_rank is not None and doc_rank is not None:
                if doc_rank < pre_doc_rank:
                    summary["variant_analysis"]["rerank_doc_id_rank_improved_count"] += 1
                elif doc_rank > pre_doc_rank:
                    summary["variant_analysis"]["rerank_doc_id_rank_worsened_count"] += 1
                else:
                    summary["variant_analysis"]["rerank_doc_id_rank_unchanged_count"] += 1

            if doc_hit:
                doc_hits += 1
                if answerability == "likely_answerable":
                    doc_hits_answerable += 1
            if src_hit:
                src_hits += 1

            record = {
                "question_id": item.get("question_id"),
                "dataset": dataset,
                "question": question,
                "original_benchmark_fields": {
                    "doc_id": item.get("doc_id"),
                    "source_file": item.get("source_file"),
                    "reference_text": reference_text,
                },
                "benchmark_audit": {
                    "provisional_answerability": answerability,
                    "answerability_reason": answerability_reason,
                    "expected_doc_id_in_test_split": doc_id_in_test,
                },
                "retrieval_eval": {
                    "variant": variant,
                    "top_k": top_k,
                    "doc_id_hit_at_k": doc_hit,
                    "doc_id_first_hit_rank": doc_rank,
                    "source_file_hit_at_k": src_hit,
                    "source_file_first_hit_rank": src_rank,
                    "where_filter": extra.get("where_filter"),
                    "title_hint_used": title_hint_used,
                    "pre_rerank_doc_id_first_hit_rank": pre_doc_rank,
                },
                "retrieval": retrieval_rows,
            }
            if pre_rerank_rows:
                record["retrieval_pre_rerank"] = pre_rerank_rows
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["retrieval_metrics"]["doc_id_hit_at_k"] = (doc_hits / doc_total) if doc_total else None
    summary["retrieval_metrics"]["source_file_hit_at_k"] = (src_hits / src_total) if src_total else None
    summary["retrieval_metrics"]["doc_id_hit_at_k_likely_answerable_subset"] = (
        (doc_hits_answerable / doc_total_answerable) if doc_total_answerable else None
    )

    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logging.info("Wrote per-question retrieval eval to %s", detailed_path)
    logging.info("Wrote retrieval eval summary to %s", summary_path)
    logging.info(
        "doc_id_hit@%s=%s over %s items | likely_answerable subset=%s over %s items",
        top_k,
        summary["retrieval_metrics"]["doc_id_hit_at_k"],
        doc_total,
        summary["retrieval_metrics"]["doc_id_hit_at_k_likely_answerable_subset"],
        doc_total_answerable,
    )


if __name__ == "__main__":
    main()
