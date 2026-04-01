# Using Custom Data to Train and Evaluate Chatbots with Existing AI Tools

## 1. Project Overview
This repository contains an MSCS graduate project comparing three practical ways to adapt chatbot behavior to custom academic content:

- a Retrieval-Augmented Generation (RAG) baseline,
- LoRA fine-tuning, and
- QLoRA fine-tuning.

The project is structured as a reproducible pipeline from raw document ingestion to benchmark evaluation and manual review.

## 2. Why This Project Matters
Many organizations and academic units have useful internal documents but limited budget, compute, or governance flexibility for full model retraining. This project evaluates whether retrieval and parameter-efficient fine-tuning can improve response quality on institution-specific content while remaining feasible for small teams.

## 3. Research Question
For custom CS institutional data, what are the practical tradeoffs among:

- retrieval-first grounding (RAG),
- LoRA adapter fine-tuning, and
- QLoRA adapter fine-tuning,

when all are evaluated under a shared and transparent workflow?

## 4. Final Scope of the Project
The final scope includes:

- data ingestion and normalization for two datasets,
- retrieval indexing and variant testing,
- a locked RAG baseline run,
- LoRA and QLoRA fine-tuning runs on the same base model family,
- automated comparison using a lightweight overlap metric, and
- manual review artifacts for supportedness and hallucination analysis.

This repository reflects completed execution for RAG, LoRA, and QLoRA experiments, with manual qualitative review as the final interpretation step.

## 5. High-Level Methodology

### RAG baseline
- Normalize source documents into `data/processed/docs.jsonl`.
- Chunk and index text into Chroma (`data/chroma_db`) using sentence-transformer embeddings.
- Run retrieval variants (`baseline`, `title_filter`, `rerank_overlap`) and lock the final baseline on `title_filter`.
- Run generation using retrieved context on the curated benchmark subset.

### LoRA fine-tuning
- Convert train/validation document splits into SFT-ready JSONL.
- Train TinyLlama adapters with LoRA settings from `configs/lora.yaml`.
- Evaluate generated answers on the curated benchmark without retrieval context injection.

### QLoRA fine-tuning
- Reuse the same SFT data and base model family.
- Train adapters with 4-bit quantization settings from `configs/qlora.yaml`.
- Evaluate on the same curated benchmark for direct comparison with LoRA and RAG.

## 6. Datasets
Two datasets are used in the current repository state:

- **Dataset 1**: Weber State CS web extraction artifacts (JSONL blocks).
  - Location: `data/raw/dataset1`
- **Dataset 2**: CS policy/curriculum/FAQ/how-to source documents (`.pdf`, `.docx`).
  - Location: `data/raw/dataset2`

Both are normalized into a shared schema in `data/processed/docs.jsonl` with fields such as `doc_id`, `dataset`, `source_file`, `title`, `section`, and `text`.

## 7. Repository Structure
The structure below reflects the current project layout and executed artifact locations:

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── rag.yaml
│   ├── lora.yaml
│   └── qlora.yaml
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
│   │   ├── benchmark_questions_curated.jsonl
│   │   ├── sft_train.jsonl
│   │   ├── sft_val.jsonl
│   │   ├── qa_sft_train.jsonl
│   │   └── qa_sft_val.jsonl
│   └── chroma_db/                  (generated vector index)
├── src/
│   ├── utils.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── preprocess.py
│   ├── build_index.py
│   ├── evaluate.py
│   ├── rag_infer.py
│   ├── prepare_sft_data.py
│   ├── finetune.py
│   ├── ft_infer.py
│   └── ft_eval.py
├── notebooks/
│   └── colab_finetune.ipynb
└── outputs/
    ├── rag/
    │   ├── retrieval_eval_*.jsonl / retrieval_eval_summary_*.json
    │   ├── predictions_*.jsonl / prediction_summary_*.json
    │   └── prediction_review_*.jsonl / prediction_review_summary_*.json
    └── ft/
        ├── lora_tinyllama/         (adapter outputs + checkpoints)
        ├── qlora_tinyllama/        (adapter outputs + checkpoints)
        ├── ft_eval_*.jsonl
        ├── ft_eval_summary_*.json
        └── manual_review_lora_qlora_benchmark.csv
```

## 8. Experiment Workflow
The workflow is intentionally linear and reproducible:

1. **Data preparation**
   - Ingest and normalize raw sources into `docs.jsonl`.
   - Build deterministic train/val/test document splits.
   - Produce benchmark question files.

2. **Retrieval/indexing**
   - Chunk normalized documents.
   - Build Chroma index with embedding model from `configs/rag.yaml`.

3. **RAG evaluation**
   - Compare retrieval variants.
   - Use the curated benchmark subset for final baseline generation/evaluation.
   - Save retrieval, prediction, and review artifacts under `outputs/rag/`.

4. **Fine-tuning**
   - Convert split documents into SFT training rows (`prepare_sft_data.py`).
   - Train LoRA and QLoRA adapters via shared entry point (`finetune.py`) with different configs.

5. **Fine-tuning evaluation**
   - Evaluate LoRA/QLoRA adapters on `benchmark_questions_curated.jsonl` using `ft_eval.py`.
   - Store per-question traces and summary metrics in `outputs/ft/`.

6. **Human review**
   - Use review fields/artifacts to judge supportedness, output quality, and hallucination behavior beyond lexical overlap.

## 9. Final Results Summary
Current comparison on the curated benchmark (`n=7`):

| System | Avg weak_reference_overlap | Retrieval context injected |
|---|---:|---|
| RAG (`title_filter` final baseline) | **0.7331** | Yes |
| LoRA (`lora_tinyllama`) | **0.1172** | No |
| QLoRA (`qlora_tinyllama`) | **0.1941** | No |

`weak_reference_overlap` is a lightweight lexical-overlap support metric. In simple terms, it checks how much answer wording overlaps with reference text tokens. It is useful as a consistency signal, but it is **not** a full correctness judgment and does not reliably detect subtle hallucinations, unsupported claims, or answer usefulness.

Human review was therefore added to evaluate:

- whether claims are actually supported by available evidence,
- whether answers are useful and specific, and
- whether fine-tuned models produce plausible but unsupported responses.

## 10. Key Findings
- RAG remained strongest for grounded factual QA in this project’s benchmark framing.
- QLoRA improved over LoRA on the current overlap metric.
- Fine-tuning-only runs (without retrieval context injection) still showed unsupported-answer and hallucination-style behavior that overlap metrics alone do not resolve.

## 11. Limitations
- The benchmark is small (curated subset of 7 items), so conclusions are directional rather than broad performance claims.
- `weak_reference_overlap` is lexical and cannot replace rigorous factuality or faithfulness evaluation.
- Fine-tuning evaluation here does not inject retrieval context, so it tests adaptation behavior rather than grounded retrieval-assisted answering.
- Manual review is necessary to interpret quality and supportedness, especially for close LoRA/QLoRA comparisons.

## 12. How to Run the Project
Run from repository root.

### Environment setup
```bash
python -m pip install -r requirements.txt
```

### Data preparation and indexing
```bash
python src/ingest.py
python src/chunking.py
python src/preprocess.py
python src/build_index.py --config configs/rag.yaml
```

### RAG retrieval evaluation
```bash
python src/evaluate.py --config configs/rag.yaml --benchmark data/processed/benchmark_questions.jsonl --test-docs data/processed/test.jsonl --output-dir outputs/rag --variant baseline
python src/evaluate.py --config configs/rag.yaml --benchmark data/processed/benchmark_questions.jsonl --test-docs data/processed/test.jsonl --output-dir outputs/rag --variant title_filter
python src/evaluate.py --config configs/rag.yaml --benchmark data/processed/benchmark_questions.jsonl --test-docs data/processed/test.jsonl --output-dir outputs/rag --variant rerank_overlap
```

### RAG generation run (curated benchmark)
```bash
python src/rag_infer.py --config configs/rag.yaml --benchmark-path data/processed/benchmark_questions_curated.jsonl --benchmark-limit 7 --variant title_filter --enable-generation
```

### Prepare SFT data and run LoRA / QLoRA training
```bash
python src/prepare_sft_data.py --mode qa --train-output data/processed/qa_sft_train.jsonl --val-output data/processed/qa_sft_val.jsonl
python src/finetune.py --config configs/lora.yaml
python src/finetune.py --config configs/qlora.yaml
```

### Evaluate LoRA / QLoRA adapters
```bash
python src/ft_eval.py --config configs/lora.yaml --adapter-dir outputs/ft/lora_tinyllama --benchmark data/processed/benchmark_questions_curated.jsonl --output-dir outputs/ft
python src/ft_eval.py --config configs/qlora.yaml --adapter-dir outputs/ft/qlora_tinyllama --benchmark data/processed/benchmark_questions_curated.jsonl --output-dir outputs/ft
```

## 13. Current Project Status / Next Step
RAG, LoRA, and QLoRA experiments have been executed and logged in `outputs/rag/` and `outputs/ft/`.  
The remaining project step is to finalize manual qualitative review (`outputs/ft/manual_review_lora_qlora_benchmark.csv`) and integrate those judgments into the final graduate write-up.

## 14. References / Acknowledgment
This repository is part of an MSCS graduate project. The implementation and evaluation framing are informed by core literature and tooling practices around:

- Retrieval-Augmented Generation (RAG),
- parameter-efficient fine-tuning with LoRA, and
- quantized adaptation with QLoRA.

It also relies on open-source ecosystems including Hugging Face Transformers/TRL/PEFT, ChromaDB, and sentence-transformer embeddings.
# Using Custom Data to Train and Evaluate Chatbots with Existing AI Tools

## Overview
I am building this repository as part of my MSCS graduate project to compare three practical approaches for domain-specific chatbot behavior:

1. Retrieval-Augmented Generation (RAG) baseline  
2. LoRA fine-tuning  
3. QLoRA fine-tuning

The repository is focused on a reproducible data-to-evaluation workflow spanning the RAG baseline and both fine-tuning paths (LoRA and QLoRA).

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
- Current phase: completed RAG baseline + completed LoRA/QLoRA runs + human review in progress

## Current Experiment Snapshot (n=7 curated benchmark)

| System | Avg weak_reference_overlap | Empty answers | Short answers | Retrieval context injected |
|---|---:|---:|---:|---|
| RAG (title_filter final baseline) | 0.7331 | 0 | 1 | Yes |
| LoRA (TinyLlama adapter) | 0.1172 | 0 | 0 | No |
| QLoRA (TinyLlama adapter) | 0.1941 | 0 | 0 | No |

Interpretation:

- QLoRA improved over LoRA on the current metric (`0.1941` vs `0.1172`), but outputs still show generic/hallucination-style behavior.
- QLoRA training logs also showed suspicious stability signals (`loss=0.0`, `grad_norm=nan`), so metric gain is not treated as a full quality win.
- RAG remains strongest for grounded factual answering on the current benchmark framing.

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
- Human judgment is still in progress; automatic overlap remains a support signal only.

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

- Dataset conversion, training, and evaluation runs have been executed for both LoRA and QLoRA.
- QLoRA was executed in a CUDA environment with `bitsandbytes`.
- Manual human review is the remaining step before final conclusion lock.

Final fine-tuning evaluation artifacts:

- LoRA summary: `outputs/ft/ft_eval_summary_lora_tinyllama_20260328T211742Z.json`
- LoRA per-question: `outputs/ft/ft_eval_lora_tinyllama_20260328T211742Z.jsonl`
- QLoRA summary: `outputs/ft/ft_eval_summary_qlora_tinyllama_20260329T212049Z.json`
- QLoRA per-question: `outputs/ft/ft_eval_qlora_tinyllama_20260329T212049Z.jsonl`
- Manual review sheet (LoRA + QLoRA): `outputs/ft/manual_review_lora_qlora_benchmark.csv`

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

## Next Phase
Complete manual human evaluation in `outputs/ft/manual_review_lora_qlora_benchmark.csv`, then publish the final narrative using both automatic metrics and human-supportedness judgments.
