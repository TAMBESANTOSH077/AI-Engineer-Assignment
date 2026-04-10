# Vexoo AI Engineer Assignment

This repo contains:

- **Part 1**: Document ingestion + retrieval via a **4-layer Knowledge Pyramid** (`ingest.py`)
- **Part 2**: GSM8K fine-tuning scripts with **LoRA** (`train.py`, `train_gsm8k.py`)

## Setup

Create/activate a Python environment (recommended) then install dependencies.

### Recommended (single command)

```bash
pip install -r requirements.txt
```

### Manual install (alternative)

```bash
pip install -U datasets transformers peft torch accelerate scikit-learn sentencepiece requests
```

### Optional for retrieval in Part 1

`ingest.py` will use `scikit-learn` for TF‑IDF if available; otherwise it falls back to a pure‑Python TF‑IDF implementation.

```bash
pip install -U scikit-learn
```

### Helpful extras (often required for LLaMA checkpoints)

```bash
pip install -U sentencepiece
```

## Part 1: Knowledge Pyramid ingestion + retrieval

### What it does

- **Sliding window**: splits text into ~2-page chunks (**4000 chars**) with ~1-page overlap (**2000 chars**)
- **4 layers per chunk**:
  - **Level 1**: raw text
  - **Level 2**: extractive summary (top sentences by in-chunk word frequency, length-capped)
  - **Level 3**: rule-based category label (`Finance`, `Tech`, `General`)
  - **Level 4**: keywords (frequency-based)
- **Retrieval**: TF‑IDF vectorization + cosine similarity over **all layers across all windows**; returns the best-matching layer.

### Run

Put your input in `document.txt`, then:

```bash
python ingest.py
```

You’ll be prompted for a query and it will print the best match (window index + level + score).

Optional flags: `--document`, `--window-chars`, `--overlap-chars`, `--keyword-top-k`, `--summary-max-chars`.

### Tests (Part 1)

```bash
python -m unittest test_ingest.py -v
```

## Part 2: GSM8K training

### `train.py` (lightweight baseline)

Uses GSM8K (CPU-friendly default: 300 train / 100 eval) with a small model (`distilgpt2`) and LoRA.

```bash
python train.py
```

### `train_gsm8k.py` (LLaMA 3.2 1B + LoRA)

Fine-tunes **LLaMA 3.2 1B** on GSM8K using LoRA (**r=16, alpha=32**) and logs:
- training loss (Trainer logging)
- **exact match accuracy** computed by regex-extracting the final numeric answer after `####`

Run:

```bash
python train_gsm8k.py
```

Common overrides:

```bash
python train_gsm8k.py --epochs 1 --batch_size 1 --grad_accum 8 --max_length 512
```

Notes:
- **Model access**: `meta-llama/Llama-3.2-1B` may require Hugging Face authentication/accepting license terms.
- **Hardware**: 1B models are much more feasible on a GPU; on CPU it will be very slow.

## Files

- `ingest.py`: KnowledgePyramid class + retrieval (`sliding_window_chunks`, `extractive_summary`)
- `test_ingest.py`: unit tests for chunking, summarization, ingest/retrieve
- `train.py`: simple LoRA baseline on GSM8K
- `train_gsm8k.py`: LLaMA 3.2 1B LoRA fine-tuning + regex exact-match evaluation
- `SUMMARY_REPORT.md`: one-page technical summary of the approach