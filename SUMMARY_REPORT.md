## One-page Summary Report (Architecture + Decisions)

### Overview

This submission is split into two parts:

- **Part 1** builds a **document ingestion + retrieval** pipeline that stores information at multiple abstraction levels (“Knowledge Pyramid”).
- **Part 2** fine-tunes a language model on **GSM8K** using **LoRA** to reduce compute and memory cost while preserving base-model capabilities.

---

### Part 1: Sliding Window Ingestion

**Goal**: preserve local context while enabling efficient downstream retrieval.

- **Windowing strategy**: character-based sliding windows sized as “2 pages” (approximately **4000 characters**) with a “1 page” overlap (approximately **2000 characters**).
- **Why overlap matters**: it prevents boundary effects where important context is split between windows; overlapping windows ensure that a query-relevant span is likely to be fully contained in at least one chunk.
- **Trade-off**: character-based chunking is simple and robust across file types, but it can cut mid-sentence. This is acceptable here because retrieval is performed across multiple representations (raw/summary/keywords), which increases robustness.

---

### Part 1: Knowledge Pyramid (4 layers)

For each window, the pipeline builds a 4-layer structure:

- **Level 1 — Raw Text**: maximal fidelity; best for exact phrasing, details, and evidence.
- **Level 2 — Summary (placeholder)**: lightweight “compression” intended to capture the gist; designed as a replaceable module (can be upgraded to a real summarizer later).
- **Level 3 — Category (rule-based)**: coarse routing label (`Finance`, `Tech`, `General`) to support fast narrowing and structured filtering.
- **Level 4 — Keywords**: sparse topical representation (frequency-based top‑k terms) to improve retrieval when queries are short or when raw text is noisy.

**Key idea**: each layer provides a different bias (precision vs abstraction). Retrieval over all layers increases the chance that at least one layer aligns well with the user query.

---

### Part 1: Retrieval (TF‑IDF + cosine similarity)

**Approach**: index **every layer of every window** using **TF‑IDF** and rank with **cosine similarity**, returning the single best-matching (window, layer, content).

- **Why TF‑IDF**: it’s deterministic, fast to compute, and strong for keyword-driven retrieval.
- **Operational detail**: `ingest.py` uses `scikit-learn` TF‑IDF when available, and includes a pure‑Python TF‑IDF fallback to keep the script runnable in minimal environments.

---

### Part 2: GSM8K fine-tuning decisions (LoRA)

**Dataset**: `openai/gsm8k` (`main`) with a small split for iteration speed:

- **Train**: 3000 examples
- **Eval**: 1000 examples

**Training format**: instruction-style text:

- `Question: <question>\nAnswer: <answer>`

**Why LoRA**:

- reduces the number of trainable parameters
- lowers GPU memory requirements
- enables quicker experimentation and adapter swapping (useful for domain specialization)

**LoRA configuration**:

- `r = 16`
- `alpha = 32`

---

### Part 2: Exact Match Accuracy (regex over GSM8K marker)

Evaluation computes **exact match** by extracting the final numeric answer after the GSM8K marker:

- looks for `#### <number>` in the ground-truth `answer` column
- generates a model response and applies the same extraction logic
- reports **exact match accuracy** (string match after normalization, e.g., comma removal)

**Why this metric**: GSM8K answers have a canonical final numeric target; extracting the post-`####` value matches the dataset’s intended evaluation pattern and avoids penalizing intermediate reasoning text.

---

### Summary of strengths

- **Modular design**: windowing, summarization, categorization, keywording, and retrieval are separable units.
- **Multi-resolution retrieval**: querying across layers improves robustness across query styles.
- **Efficient training**: LoRA enables practical fine-tuning under constrained resources and supports adapter-based specialization.

