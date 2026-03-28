from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from utils import ensure_parent, setup_logging


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_retrieval_context(
    collection: Any,
    embed_model: SentenceTransformer,
    query: str,
    top_k: int,
    dataset_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_embedding = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    where = {"dataset": dataset_filter} if dataset_filter else None
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    records: List[Dict[str, Any]] = []
    for doc_text, metadata, distance in zip(documents, metadatas, distances):
        records.append(
            {
                "distance": float(distance),
                "metadata": metadata,
                "text": doc_text,
            }
        )
    return records


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


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer_text(text: str) -> str:
    if not text:
        return ""
    # Remove prompt-like tokens that sometimes leak in generated text.
    cleaned = re.sub(r"\[Context\s+\d+\]\s*", "", text)
    return compact_whitespace(cleaned)


def short_snippet(text: str, max_chars: int = 320) -> str:
    normalized = compact_whitespace(text or "")
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + " ..."


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def weak_reference_overlap(answer: str, reference_text: str) -> Optional[float]:
    if not answer or not reference_text:
        return None
    ans_tokens = token_set(answer)
    ref_tokens = token_set(reference_text)
    if not ans_tokens:
        return 0.0
    return len(ans_tokens & ref_tokens) / len(ans_tokens)


def first_rank_match(retrieval_rows: List[Dict[str, Any]], key: str, expected_value: Optional[str]) -> Optional[int]:
    if not expected_value:
        return None
    for idx, row in enumerate(retrieval_rows, start=1):
        got = str((row.get("metadata") or {}).get(key, ""))
        if got == str(expected_value):
            return idx
    return None


def rerank_by_overlap(rows: List[Dict[str, Any]], query: str, overlap_weight: float) -> List[Dict[str, Any]]:
    q_tokens = set(tokenize(query))
    rescored: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("metadata", {}) or {}
        text = str(row.get("text", ""))
        title = str(meta.get("title", ""))
        section = str(meta.get("section", ""))
        overlap = len(q_tokens & (set(tokenize(text)) | set(tokenize(title)) | set(tokenize(section))))
        distance = float(row.get("distance", 0.0))
        row = dict(row)
        row["rerank_overlap"] = overlap
        row["rerank_score"] = (-distance) + (overlap_weight * overlap)
        rescored.append(row)

    rescored.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
    return rescored


def get_retrieval_context_variant(
    collection: Any,
    embed_model: SentenceTransformer,
    query: str,
    top_k: int,
    dataset_filter: Optional[str],
    variant: str,
    candidate_k: int,
    overlap_weight: float,
) -> Dict[str, Any]:
    if variant == "baseline":
        rows = get_retrieval_context(collection, embed_model, query, top_k, dataset_filter)
        return {"rows": rows, "variant_info": {"title_hint_used": None, "where_filter": build_where_filter(dataset_filter, None)}}

    title_hint = extract_title_hint(query)
    if variant == "title_filter" and title_hint:
        query_embedding = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        where = build_where_filter(dataset_filter, title_hint)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for doc_text, metadata, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            rows.append({"distance": float(distance), "metadata": metadata, "text": doc_text})
        return {"rows": rows, "variant_info": {"title_hint_used": title_hint, "where_filter": where}}

    if variant == "rerank_overlap":
        candidate_rows = get_retrieval_context(
            collection=collection,
            embed_model=embed_model,
            query=query,
            top_k=max(candidate_k, top_k),
            dataset_filter=dataset_filter,
        )
        reranked = rerank_by_overlap(candidate_rows, query, overlap_weight)[:top_k]
        return {
            "rows": reranked,
            "variant_info": {
                "title_hint_used": title_hint,
                "where_filter": build_where_filter(dataset_filter, None),
                "pre_rerank_count": len(candidate_rows),
            },
        }

    rows = get_retrieval_context(collection, embed_model, query, top_k, dataset_filter)
    return {"rows": rows, "variant_info": {"title_hint_used": title_hint, "where_filter": build_where_filter(dataset_filter, None)}}


def build_generation_context(
    retrieval_rows: List[Dict[str, Any]],
    max_context_chars: int,
    per_chunk_char_limit: int,
    context_policy: str,
) -> Dict[str, Any]:
    if context_policy == "top2_compact":
        retrieval_rows = retrieval_rows[:2]
        max_context_chars = min(max_context_chars, 1400)
        per_chunk_char_limit = min(per_chunk_char_limit, 650)

    used_rows: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    total = 0
    truncated = False

    for idx, row in enumerate(retrieval_rows, start=1):
        text = str(row.get("text", ""))
        if len(text) > per_chunk_char_limit:
            text = text[:per_chunk_char_limit].rstrip() + " ..."
            truncated = True

        block = f"[Context {idx}] {text}"
        proposed = total + len(block) + (2 if context_parts else 0)
        if proposed > max_context_chars:
            truncated = True
            break

        context_parts.append(block)
        total = proposed
        used_rows.append(row)

    return {
        "context_block": "\n\n".join(context_parts),
        "used_rows": used_rows,
        "context_truncated": truncated,
    }


def generate_answer(
    query: str,
    context_block: str,
    generation_cfg: Dict[str, Any],
    prompt_variant: str,
) -> Dict[str, Any]:
    if not generation_cfg.get("enabled", False):
        return {
            "answer": None,
            "status": "generation_disabled",
            "note": "Retrieval completed. Generation is disabled in configs/rag.yaml.",
        }

    try:
        from transformers import AutoTokenizer, pipeline
    except Exception as exc:
        return {
            "answer": None,
            "status": "generation_unavailable",
            "note": f"Transformers pipeline unavailable: {exc}",
            "input_token_truncated": False,
        }

    if prompt_variant == "grounded_brief":
        prompt = (
            "Use only the context to answer briefly in 1-3 sentences. "
            "Do not add facts not present in context. "
            "If insufficient, reply: Not enough information in provided context.\n\n"
            f"Question: {query}\n\n"
            f"{context_block}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            "Answer the question using only the provided context. "
            "If the answer is not present, say you do not have enough information.\n\n"
            f"Question: {query}\n\n"
            f"{context_block}\n\n"
            "Answer:"
        )

    model_name = str(generation_cfg["model_name"])
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 200))
    max_input_tokens = int(generation_cfg.get("max_input_tokens", 448))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Explicit token-budget truncation before generation call.
        tokenized = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors=None,
        )
        input_ids = tokenized.get("input_ids", [])
        input_token_truncated = len(input_ids) >= max_input_tokens
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

        generator = pipeline("text2text-generation", model=model_name)
        output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False, truncation=True)
        text = output[0]["generated_text"].strip() if output else ""
        text = normalize_answer_text(text)
        return {
            "answer": text,
            "status": "generated",
            "note": f"Model: {model_name}",
            "input_token_truncated": input_token_truncated,
        }
    except Exception as exc:
        return {
            "answer": None,
            "status": "generation_failed",
            "note": f"Failed to run generation model '{model_name}': {exc}",
            "input_token_truncated": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG retrieval and optional answer generation.")
    parser.add_argument("--config", default="configs/rag.yaml", help="Path to RAG config file.")
    parser.add_argument("--query", default=None, help="Single query string.")
    parser.add_argument(
        "--benchmark-path",
        default="data/processed/benchmark_questions.jsonl",
        help="Benchmark question file path.",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=1,
        help="Number of benchmark questions to run when --query is not set.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    parser.add_argument("--dataset-filter", default=None, help="Optional dataset filter.")
    parser.add_argument(
        "--variant",
        choices=["baseline", "title_filter", "rerank_overlap"],
        default=None,
        help="Retrieval variant.",
    )
    parser.add_argument("--candidate-k", type=int, default=None, help="Candidate set size for rerank_overlap.")
    parser.add_argument("--overlap-weight", type=float, default=None, help="Token overlap weight for rerank_overlap.")
    parser.add_argument(
        "--prompt-variant",
        choices=["baseline", "grounded_brief"],
        default=None,
        help="Generation prompt variant.",
    )
    parser.add_argument(
        "--context-policy",
        choices=["baseline", "top2_compact"],
        default=None,
        help="Generation context policy.",
    )
    parser.add_argument("--enable-generation", action="store_true", help="Force-enable generation for this run.")
    parser.add_argument("--disable-generation", action="store_true", help="Force-disable generation for this run.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to outputs/rag/predictions_<variant>_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary JSON path. Defaults to outputs/rag/prediction_summary_<variant>_<timestamp>.json",
    )
    parser.add_argument(
        "--review-output",
        default=None,
        help="Optional review JSONL path. Defaults to outputs/rag/prediction_review_<variant>_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--review-summary-output",
        default=None,
        help="Optional review summary JSON path. Defaults to outputs/rag/prediction_review_summary_<variant>_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(Path(args.config))

    chroma_path = Path(config["index"]["chroma_path"])
    collection_name = str(config["index"]["collection_name"])
    embedding_model_name = str(config["embedding"]["model_name"])
    top_k = int(args.top_k or config["retrieval"]["top_k"])
    variant = str(args.variant or config.get("retrieval", {}).get("default_variant", "baseline"))
    candidate_k = int(args.candidate_k or config.get("retrieval", {}).get("candidate_k", 20))
    overlap_weight = float(args.overlap_weight or config.get("retrieval", {}).get("overlap_weight", 0.08))
    generation_cfg = dict(config.get("generation", {}))
    max_context_chars = int(generation_cfg.get("max_context_chars", 2800))
    per_chunk_char_limit = int(generation_cfg.get("per_chunk_char_limit", 900))
    short_answer_char_threshold = int(generation_cfg.get("short_answer_char_threshold", 80))
    prompt_variant = str(args.prompt_variant or generation_cfg.get("prompt_variant", "baseline"))
    context_policy = str(args.context_policy or generation_cfg.get("context_policy", "baseline"))
    if args.enable_generation and args.disable_generation:
        raise ValueError("Use either --enable-generation or --disable-generation, not both.")
    if args.enable_generation:
        generation_cfg["enabled"] = True
    if args.disable_generation:
        generation_cfg["enabled"] = False

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(config["runtime"]["output_dir"])
        output_path = output_dir / f"predictions_{variant}_{ts}.jsonl"
    if args.summary_output:
        summary_path = Path(args.summary_output)
    else:
        output_dir = Path(config["runtime"]["output_dir"])
        summary_path = output_dir / f"prediction_summary_{variant}_{ts}.json"
    if args.review_output:
        review_output_path = Path(args.review_output)
    else:
        output_dir = Path(config["runtime"]["output_dir"])
        review_output_path = output_dir / f"prediction_review_{variant}_{ts}.jsonl"
    if args.review_summary_output:
        review_summary_path = Path(args.review_summary_output)
    else:
        output_dir = Path(config["runtime"]["output_dir"])
        review_summary_path = output_dir / f"prediction_review_summary_{variant}_{ts}.json"

    if args.query:
        requests = [{"question_id": "adhoc", "question": args.query, "dataset": args.dataset_filter}]
    else:
        benchmark = read_jsonl(Path(args.benchmark_path))
        requests = benchmark[: args.benchmark_limit]

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    embed_model = SentenceTransformer(embedding_model_name)
    logging.info("Loaded retrieval model: %s", embedding_model_name)
    logging.info("Loaded collection '%s' from %s", collection_name, chroma_path)

    ensure_parent(output_path)
    generated_count = 0
    failed_count = 0
    disabled_count = 0
    unavailable_count = 0
    total = 0
    answer_char_lengths: List[int] = []
    answer_empty_count = 0
    answer_short_count = 0
    context_truncated_count = 0
    expected_doc_retrieved_count = 0
    expected_source_retrieved_count = 0
    weak_overlap_values: List[float] = []
    input_token_truncated_count = 0
    review_rows: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row in requests:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            total += 1

            dataset_filter = args.dataset_filter or row.get("dataset")
            retrieval_result = get_retrieval_context_variant(
                collection=collection,
                embed_model=embed_model,
                query=question,
                top_k=top_k,
                dataset_filter=str(dataset_filter) if dataset_filter else None,
                variant=variant,
                candidate_k=candidate_k,
                overlap_weight=overlap_weight,
            )
            contexts = retrieval_result["rows"]
            context_pack = build_generation_context(
                retrieval_rows=contexts,
                max_context_chars=max_context_chars,
                per_chunk_char_limit=per_chunk_char_limit,
                context_policy=context_policy,
            )
            answer = generate_answer(question, context_pack["context_block"], generation_cfg, prompt_variant)
            if bool(answer.get("input_token_truncated", False)):
                input_token_truncated_count += 1
            if answer["status"] == "generated":
                generated_count += 1
            elif answer["status"] == "generation_failed":
                failed_count += 1
            elif answer["status"] == "generation_unavailable":
                unavailable_count += 1
            elif answer["status"] == "generation_disabled":
                disabled_count += 1

            answer_text = str(answer["answer"] or "")
            answer_len = len(answer_text)
            answer_char_lengths.append(answer_len)
            answer_empty = answer_len == 0
            answer_short = answer_len > 0 and answer_len < short_answer_char_threshold
            if answer_empty:
                answer_empty_count += 1
            if answer_short:
                answer_short_count += 1
            if context_pack["context_truncated"]:
                context_truncated_count += 1

            expected_doc_id = row.get("doc_id")
            expected_source_file = row.get("source_file")
            expected_doc_rank = first_rank_match(contexts, "doc_id", expected_doc_id)
            expected_source_rank = first_rank_match(contexts, "source_file", expected_source_file)
            expected_doc_retrieved = expected_doc_rank is not None
            expected_source_retrieved = expected_source_rank is not None
            if expected_doc_retrieved:
                expected_doc_retrieved_count += 1
            if expected_source_retrieved:
                expected_source_retrieved_count += 1

            overlap = weak_reference_overlap(answer_text, str(row.get("reference_text", "")))
            if overlap is not None:
                weak_overlap_values.append(overlap)

            top_used_chunk = context_pack["used_rows"][0] if context_pack["used_rows"] else None
            top_used_meta = (top_used_chunk or {}).get("metadata", {}) if top_used_chunk else {}

            review_rows.append(
                {
                    "question_id": row.get("question_id"),
                    "question": question,
                    "dataset": row.get("dataset"),
                    "variant": variant,
                    "expected_doc_id": expected_doc_id,
                    "expected_source_file": expected_source_file,
                    "expected_doc_retrieved": expected_doc_retrieved,
                    "expected_doc_first_hit_rank": expected_doc_rank,
                    "expected_source_retrieved": expected_source_retrieved,
                    "expected_source_first_hit_rank": expected_source_rank,
                    "answer_status": answer["status"],
                    "answer": answer_text,
                    "answer_empty": answer_empty,
                    "answer_short": answer_short,
                    "answer_char_length": answer_len,
                    "context_truncated": context_pack["context_truncated"],
                    "retrieval_variant_info": retrieval_result.get("variant_info", {}),
                    "retrieved_chunks": contexts,
                    "used_context_chunks": context_pack["used_rows"],
                    "top_evidence_snippet": short_snippet(str((top_used_chunk or {}).get("text", ""))),
                    "top_evidence_doc_id": top_used_meta.get("doc_id"),
                    "top_evidence_source_file": top_used_meta.get("source_file"),
                    "weak_reference_overlap": overlap,
                    "input_token_truncated": bool(answer.get("input_token_truncated", False)),
                    "prompt_variant": prompt_variant,
                    "context_policy": context_policy,
                    "human_judgment_answer_supported": None,
                    "human_judgment_answer_quality": None,
                    "human_review_notes": "",
                }
            )

            output_record = {
                "question_id": row.get("question_id"),
                "question": question,
                "dataset_filter": dataset_filter,
                "variant": variant,
                "top_k": top_k,
                "candidate_k": candidate_k if variant == "rerank_overlap" else None,
                "prompt_variant": prompt_variant,
                "context_policy": context_policy,
                "retrieval": contexts,
                "used_context_chunks": context_pack["used_rows"],
                "context_truncated": context_pack["context_truncated"],
                "retrieval_variant_info": retrieval_result.get("variant_info", {}),
                "expected_doc_id": expected_doc_id,
                "expected_source_file": expected_source_file,
                "expected_doc_retrieved": expected_doc_retrieved,
                "expected_source_retrieved": expected_source_retrieved,
                "answer": answer["answer"],
                "answer_status": answer["status"],
                "answer_note": answer["note"],
                "input_token_truncated": bool(answer.get("input_token_truncated", False)),
            }
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    ensure_parent(summary_path)
    summary = {
        "timestamp_utc": ts,
        "benchmark_path": args.benchmark_path if not args.query else None,
        "variant": variant,
        "top_k": top_k,
        "generation_enabled": bool(generation_cfg.get("enabled", False)),
        "generation_model": generation_cfg.get("model_name"),
        "prompt_variant": prompt_variant,
        "context_policy": context_policy,
        "total_questions": total,
        "generated_count": generated_count,
        "generation_failed_count": failed_count,
        "generation_unavailable_count": unavailable_count,
        "generation_disabled_count": disabled_count,
        "answer_empty_count": answer_empty_count,
        "answer_short_count": answer_short_count,
        "context_truncated_count": context_truncated_count,
        "input_token_truncated_count": input_token_truncated_count,
        "expected_doc_retrieved_count": expected_doc_retrieved_count,
        "expected_source_retrieved_count": expected_source_retrieved_count,
        "average_answer_char_length": (sum(answer_char_lengths) / len(answer_char_lengths)) if answer_char_lengths else 0.0,
        "average_weak_reference_overlap": (sum(weak_overlap_values) / len(weak_overlap_values)) if weak_overlap_values else None,
        "short_answer_char_threshold": short_answer_char_threshold,
        "max_context_chars": max_context_chars,
        "per_chunk_char_limit": per_chunk_char_limit,
        "max_input_tokens": int(generation_cfg.get("max_input_tokens", 448)),
        "predictions_file": str(output_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    ensure_parent(review_output_path)
    with review_output_path.open("w", encoding="utf-8") as handle:
        for review in review_rows:
            handle.write(json.dumps(review, ensure_ascii=False) + "\n")

    review_summary = {
        "timestamp_utc": ts,
        "variant": variant,
        "total_questions": total,
        "review_file": str(review_output_path),
        "expected_doc_retrieved_count": expected_doc_retrieved_count,
        "expected_source_retrieved_count": expected_source_retrieved_count,
        "answer_empty_count": answer_empty_count,
        "answer_short_count": answer_short_count,
        "context_truncated_count": context_truncated_count,
        "input_token_truncated_count": input_token_truncated_count,
        "average_answer_char_length": (sum(answer_char_lengths) / len(answer_char_lengths)) if answer_char_lengths else 0.0,
        "average_weak_reference_overlap": (sum(weak_overlap_values) / len(weak_overlap_values)) if weak_overlap_values else None,
        "note": "Inspection-oriented flags are weak aids and not correctness claims.",
    }
    ensure_parent(review_summary_path)
    with review_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(review_summary, handle, indent=2, ensure_ascii=False)

    logging.info("Wrote %s prediction records to %s", total, output_path)
    logging.info("Wrote prediction summary to %s", summary_path)
    logging.info("Wrote prediction review to %s", review_output_path)
    logging.info("Wrote prediction review summary to %s", review_summary_path)


if __name__ == "__main__":
    main()
