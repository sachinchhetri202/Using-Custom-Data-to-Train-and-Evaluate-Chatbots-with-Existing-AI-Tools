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
- Current phase: baseline-complete handoff to LoRA/QLoRA pipeline preparation
- Not started yet: executed LoRA/QLoRA training runs

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

## Final Baseline Lock (Before LoRA/QLoRA)
The baseline is now locked using executed artifacts only:

- Retrieval setting: `title_filter` (`top_k=5`)  
  - Evidence: `outputs/rag/retrieval_eval_summary_title_filter_20260326T062055Z.json`
  - Final confirmation run: `outputs/rag/retrieval_eval_summary_title_filter_20260328T024726Z.json`
  - Observed on curated benchmark (`n=7`): `doc_id_hit@5 = 1.0`, `source_file_hit@5 = 1.0`
- Generation setting: `google/flan-t5-small` with `prompt_variant=baseline`, `context_policy=baseline`
  - Final run: `outputs/rag/prediction_summary_title_filter_final_baseline.json`
  - Observed on curated benchmark (`n=7`): `answer_short_count=1`, `context_truncated_count=1`, `input_token_truncated_count=4`
- Benchmark used for this lock:
  - `data/processed/benchmark_questions_curated.jsonl`
  - Note: this benchmark is curated from a provisional source and is not a final gold benchmark.

Interpretation for baseline lock:

- `title_filter` remains the strongest retrieval variant in executed curated runs.
- Combined ablations reduced input-token truncation but increased short-answer and context-truncation flags.
- The default baseline prompt/context setting remains the more stable anchor for comparison, despite known imperfections.
- Weak automatic overlap-style signals are retained as support signals only, not correctness judgments.

## Lightweight Human Review Artifact (Final Baseline)
Final per-question review support output:

- `outputs/rag/prediction_review_title_filter_final_baseline.jsonl`

Each row includes:
- question + expected `doc_id`/`source_file`
- retrieved evidence and a `top_evidence_snippet`
- generated answer and retrieval-hit indicators
- short-answer/context-truncation/input-token-truncation flags
- weak overlap support signal (`weak_reference_overlap`)
- human-judgment placeholders:
  - `human_judgment_answer_supported`
  - `human_judgment_answer_quality`
  - `human_review_notes`

This is intentionally inspection-oriented and does not claim automatic correctness.

## Current Limitations
- `benchmark_questions.jsonl` is still a provisional scaffold, not a final gold benchmark.
- `benchmark_questions_curated.jsonl` is derived from that provisional benchmark and remains preliminary.
- Automatic answer checks are weak support signals, not correctness proof.
- LoRA/QLoRA training has not yet been executed in this repository state.

## Fine-Tuning Startup (LoRA/QLoRA)
This repository now includes a minimal shared SFT path for LoRA and QLoRA, designed to keep comparison against the locked RAG baseline fair and reproducible.

What was prepared:

- Dataset preparation script: `src/prepare_sft_data.py`
  - Converts existing document splits (`train.jsonl`, `val.jsonl`) into SFT-ready instruction/response rows
  - Supports two deterministic modes:
    - `doc_reconstruction`: summary-style instruction with full source passage target
    - `qa`: QA-style question instruction with full source passage target
  - Output artifacts currently in repo:
    - `data/processed/sft_train.jsonl`, `data/processed/sft_val.jsonl` (doc-reconstruction/domain-adaptation style)
    - `data/processed/qa_sft_train.jsonl`, `data/processed/qa_sft_val.jsonl` (QA-style prompts from train/val docs)
  - Provenance note: these are derived from existing deterministic train/val splits; curated benchmark/test docs are not used for training data
- Shared training entrypoint: `src/finetune.py`
  - One code path controlled by config
  - Supports LoRA and QLoRA via config flags
  - Includes `--dry-run` validation mode (config + data checks without training)
- Adapter inference entrypoint: `src/ft_infer.py`
- Configs:
  - LoRA: `configs/lora.yaml`
  - QLoRA: `configs/qlora.yaml`
  - Current config default points to QA-style SFT files (`qa_sft_train.jsonl`, `qa_sft_val.jsonl`)

Important execution status:

- In this phase, dataset conversion and dry-run checks were executed.
- No LoRA/QLoRA training run is claimed in this phase.
- QLoRA requires CUDA GPU + `bitsandbytes`; this is expected to run in Colab or another GPU environment.

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

### 6) Prepare fine-tuning dataset artifacts (no training)
```bash
python src/prepare_sft_data.py --mode doc_reconstruction --train-output data/processed/sft_train.jsonl --val-output data/processed/sft_val.jsonl
python src/prepare_sft_data.py --mode qa --train-output data/processed/qa_sft_train.jsonl --val-output data/processed/qa_sft_val.jsonl
```

## Planned Next Phase
Next phase should execute LoRA and QLoRA training runs (preferably in Colab/GPU), then compare those outputs against the locked RAG baseline using the same benchmark framing and similarly cautious interpretation.  
