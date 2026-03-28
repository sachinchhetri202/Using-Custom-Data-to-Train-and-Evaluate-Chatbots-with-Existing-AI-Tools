# Fine-Tuning Plan: LoRA and QLoRA

**Status:** Pre-training — baseline locked, data prepared, code ready, training not yet executed.

---

## 1. Existing Files That Are the Source of Truth

These files already exist and must not be duplicated or replaced:

| File | Role |
|---|---|
| `src/finetune.py` | Single shared training entrypoint for both LoRA and QLoRA |
| `src/ft_infer.py` | Adapter inference (loads base model + PEFT adapter) |
| `src/prepare_sft_data.py` | Produced the SFT datasets; do not re-run unless data changes |
| `src/evaluate.py` | Existing evaluation harness; will be reused for fine-tuned model scoring |
| `configs/lora.yaml` | LoRA training configuration |
| `configs/qlora.yaml` | QLoRA training configuration (4-bit, nf4) |
| `data/processed/qa_sft_train.jsonl` | Primary SFT training split (QA-style) |
| `data/processed/qa_sft_val.jsonl` | Primary SFT validation split (QA-style) |
| `data/processed/benchmark_questions_curated.jsonl` | Held-out benchmark for comparison against RAG baseline |
| `outputs/rag/` | Locked RAG baseline artifacts — do not modify |
| `requirements.txt` | Dependency manifest; add only if a new package is strictly required |

---

## 2. Single Colab Notebook

**One notebook only:** `notebooks/colab_finetune.ipynb`

This notebook is the only new file to create for the training phase. It must:

1. Mount Google Drive and clone/pull the repo (or point to it in Drive)
2. Install dependencies: `pip install -r requirements.txt`
3. Upload or confirm the presence of `data/processed/qa_sft_train.jsonl` and `qa_sft_val.jsonl`
4. Run a dry-run check before any training:
   ```bash
   python src/finetune.py --config configs/lora.yaml --dry-run
   python src/finetune.py --config configs/qlora.yaml --dry-run
   ```
5. Execute the actual training run (one config at a time):
   ```bash
   python src/finetune.py --config configs/lora.yaml
   # or
   python src/finetune.py --config configs/qlora.yaml
   ```
6. Save adapter outputs back to Drive
7. Run a quick inference spot-check using `src/ft_infer.py`

The notebook must **not** reimplement training logic inline. All logic stays in `src/finetune.py`.

---

## 3. Minimal Helper Scripts — None New Required

All necessary code is already in place:

- Training → `src/finetune.py` (handles both LoRA and QLoRA via config flag)
- Inference → `src/ft_infer.py`
- Evaluation → `src/evaluate.py` (will need a thin inference wrapper or CLI flag for the fine-tuned path, but no new file)
- Data preparation → `src/prepare_sft_data.py` (already executed; outputs already committed)

If the evaluation harness needs minor extension to score fine-tuned model outputs, **edit `src/evaluate.py` in place** — do not create a parallel eval script.

---

## 4. Input / Output Artifacts

### Inputs (already present)

```
data/processed/qa_sft_train.jsonl     ← primary training set
data/processed/qa_sft_val.jsonl       ← validation set during training
data/processed/benchmark_questions_curated.jsonl  ← held-out evaluation set
configs/lora.yaml                     ← LoRA hyperparameters
configs/qlora.yaml                    ← QLoRA hyperparameters
```

### Outputs (to be produced by training runs)

```
outputs/ft/lora_tinyllama/            ← LoRA adapter weights + tokenizer
    adapter_config.json
    adapter_model.safetensors (or .bin)
    tokenizer files

outputs/ft/qlora_tinyllama/           ← QLoRA adapter weights + tokenizer
    adapter_config.json
    adapter_model.safetensors (or .bin)
    tokenizer files
```

> **Note:** `configs/lora.yaml` currently writes to `outputs/lora/first_lora_qa_tinyllama/`.
> Align this to `outputs/ft/lora_tinyllama/` before the first training run so both adapters
> live under the same `outputs/ft/` tree as `qlora_tinyllama/`.

### Evaluation outputs (after training)

```
outputs/ft/lora_eval_summary.json     ← benchmark scores for LoRA adapter
outputs/ft/qlora_eval_summary.json    ← benchmark scores for QLoRA adapter
```

These will be compared against the locked RAG baseline in `outputs/rag/`.

---

## 5. What Must NOT Be Created

- **No additional training scripts.** `finetune.py` is the single entrypoint for both LoRA and QLoRA.
- **No separate LoRA vs. QLoRA notebooks.** One notebook, two config files.
- **No new framework dependencies** (no axolotl, no LlamaFactory, no unsloth, no custom trainers).
- **No new config files** beyond `lora.yaml` and `qlora.yaml` unless the base model changes.
- **No dataset3 references** anywhere in any file, comment, or output.
- **No changes to `outputs/rag/`** — that directory is the locked RAG baseline.
- **No new data processing scripts.** SFT data is finalized; `prepare_sft_data.py` is not re-run.
- **No parallel evaluation pipelines.** Fine-tuned model scoring reuses `src/evaluate.py`.
- **No model merging scripts** for this phase (merge base + adapter is deferred to optional post-analysis).

---

## 6. Minimal Final Structure After Training

```
.
├── FINETUNING_PLAN.md               (this file)
├── README.md
├── requirements.txt
├── configs/
│   ├── lora.yaml
│   ├── qlora.yaml
│   └── rag.yaml
├── notebooks/
│   └── colab_finetune.ipynb         (ONE notebook — the only new file)
├── src/
│   ├── finetune.py                  (training)
│   ├── ft_infer.py                  (adapter inference)
│   ├── evaluate.py                  (shared eval harness)
│   ├── prepare_sft_data.py
│   ├── rag_infer.py
│   ├── build_index.py
│   ├── chunking.py
│   ├── ingest.py
│   ├── preprocess.py
│   └── utils.py
├── data/
│   ├── processed/
│   │   ├── qa_sft_train.jsonl       ← primary SFT input
│   │   ├── qa_sft_val.jsonl         ← primary SFT input
│   │   ├── benchmark_questions_curated.jsonl
│   │   └── (other existing files unchanged)
│   └── raw/
│       └── dataset1/
└── outputs/
    ├── rag/                         (locked — do not touch)
    └── ft/
        ├── lora_tinyllama/          ← produced by training
        ├── qlora_tinyllama/         ← produced by training
        ├── lora_eval_summary.json   ← produced by evaluation
        └── qlora_eval_summary.json  ← produced by evaluation
```

---

## 7. Immediate Next Actions (Before Training)

1. Fix the `output_dir` in `configs/lora.yaml` from `outputs/lora/first_lora_qa_tinyllama` to `outputs/ft/lora_tinyllama` for consistency.
2. Create `notebooks/colab_finetune.ipynb` with Drive setup, dry-run, training, and spot-check cells.
3. Run `--dry-run` in Colab to confirm dataset loading and config parsing before any GPU time is spent.
4. Execute LoRA first (no quantization, faster iteration), then QLoRA.
5. After each run, score on `benchmark_questions_curated.jsonl` and save summary to `outputs/ft/`.
