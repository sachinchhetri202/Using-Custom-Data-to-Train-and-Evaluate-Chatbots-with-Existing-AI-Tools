# Using Custom Data to Train and Evaluate Chatbots with Existing AI Tools

## Overview
I am building this repository as part of my MSCS graduate project to compare three practical approaches for domain-specific chatbot behavior:

1. Retrieval-Augmented Generation (RAG) baseline  
2. LoRA fine-tuning  
3. QLoRA fine-tuning

The current repository is intentionally focused on a reproducible data + RAG foundation before any fine-tuning begins.

## Why This Project Matters
In academic and institutional settings, teams often have useful internal documents but limited resources for full model retraining. This project is designed to test whether lightweight retrieval and parameter-efficient fine-tuning can produce grounded, useful answers on custom university-focused content.

## Research Question
Given custom CS-related institutional data, what are the practical tradeoffs between:

- a retrieval-first baseline (RAG),
- LoRA fine-tuning, and
- QLoRA fine-tuning

when evaluated in a transparent, reproducible workflow?

## Project Scope
- In scope: `dataset1` and `dataset2`
- Out of scope: `dataset3` (intentionally excluded)
- Current phase: data pipeline + RAG baseline and bridge-stage generation inspection
- Not started yet: LoRA and QLoRA training/inference

## Datasets
### Dataset 1 (Weber State CS web extraction)
- Location: `data/raw/dataset1`
- Raw source used: `blocks.jsonl`
- Prior artifact `processed/processed_blocks.jsonl` was inspected and not treated as authoritative.

### Dataset 2 (CS policy/curriculum/FAQ/how-to documents)
- Location: `data/raw/dataset2`
- Source types: `.docx`, `.pdf`, and subfolders with original material

Both datasets are normalized into one shared schema in `data/processed/docs.jsonl` with fields:
`dataset`, `doc_id`, `source_file`, `source_type`, `title`, `section`, `source_url`, `language`, `text`.

## Current Repository Structure
```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── rag.yaml
├── data/
│   ├── raw/
│   │   ├── dataset1/
│   │   └── dataset2/
│   ├── processed/
│   │   ├── docs.jsonl
│   │   ├── chunks.jsonl
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   ├── benchmark_questions.jsonl
│   │   └── benchmark_questions_curated.jsonl
│   └── chroma_db/              (generated local vector index)
├── src/
│   ├── utils.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── preprocess.py
│   ├── build_index.py
│   ├── evaluate.py
│   └── rag_infer.py
└── outputs/
    └── rag/                    (executed retrieval/generation artifacts)
```

## What Has Been Implemented So Far
- Raw-data ingestion and conservative text cleaning
- Shared normalization for dataset1 + dataset2
- Deterministic train/val/test document-level splits
- Benchmark scaffold generation from held-out data
- Chunk creation for retrieval
- Chroma index building with metadata-aware storage
- Retrieval evaluation harness (`hit@k`/inspection-style reporting)
- Retrieval variant comparison (`baseline`, `title_filter`, `rerank_overlap`)
- Curated benchmark subset creation (`benchmark_questions_curated.jsonl`)
- Optional generation path with explicit context and token-budget controls
- Prediction trace/review outputs for inspection-oriented analysis

## What Has Actually Been Executed
The following have been run and saved under `outputs/rag/`:
- Retrieval smoke outputs
- Retrieval evaluation outputs (`retrieval_eval_*.jsonl`, `retrieval_eval_summary_*.json`)
- Generation bridge outputs on curated subset (`predictions_*`, `prediction_summary_*`)
- Per-question review artifacts (`prediction_review_*`, comparison artifacts)

All these outputs are from actual script executions in this repository. No simulated metrics files were created.

## Executed Findings So Far (Preliminary)
- Retrieval: `title_filter` was the best observed backend in the executed provisional/curated runs.
- Generation bridge: expected documents were often retrieved, but answer quality remained mixed.
- Ablation signal: combined prompt/context changes removed input token truncation but increased short-answer/context truncation tradeoffs.
- Current interpretation: no strong quality-improvement claim is justified yet without deeper manual review.

## Current Limitations
- `benchmark_questions.jsonl` is still a provisional scaffold, not a final gold benchmark.
- `benchmark_questions_curated.jsonl` is derived from that provisional benchmark and remains preliminary.
- Automatic answer checks are weak support signals, not correctness proof.
- LoRA/QLoRA phases are not yet implemented in this repository state.

## How To Run The Current Pipeline (Repo Root)
### 0) Environment setup
```bash
python -m pip install -r requirements.txt
```

### 1) Data pipeline
```bash
python src/ingest.py
python src/chunking.py
python src/preprocess.py
```

### 2) Build RAG index
```bash
python src/build_index.py --config configs/rag.yaml
```

### 3) Retrieval evaluation (provisional benchmark)
```bash
python src/evaluate.py --config configs/rag.yaml --benchmark data/processed/benchmark_questions.jsonl --test-docs data/processed/test.jsonl --output-dir outputs/rag --variant baseline
python src/evaluate.py --config configs/rag.yaml --benchmark data/processed/benchmark_questions.jsonl --test-docs data/processed/test.jsonl --output-dir outputs/rag --variant title_filter
python src/evaluate.py --config configs/rag.yaml --benchmark data/processed/benchmark_questions.jsonl --test-docs data/processed/test.jsonl --output-dir outputs/rag --variant rerank_overlap
```

### 4) Curated-subset generation bridge
```bash
python src/rag_infer.py --config configs/rag.yaml --benchmark-path data/processed/benchmark_questions_curated.jsonl --benchmark-limit 7 --variant title_filter --enable-generation
```

### 5) Isolated generation ablations (executed in current project state)
```bash
python src/rag_infer.py --config configs/rag.yaml --benchmark-path data/processed/benchmark_questions_curated.jsonl --benchmark-limit 7 --variant title_filter --enable-generation --prompt-variant grounded_brief --context-policy baseline
python src/rag_infer.py --config configs/rag.yaml --benchmark-path data/processed/benchmark_questions_curated.jsonl --benchmark-limit 7 --variant title_filter --enable-generation --prompt-variant baseline --context-policy top2_compact
```

## Planned Next Phase
Next session will start fine-tuning work (LoRA and QLoRA) using this cleaned RAG baseline as the anchor.  
