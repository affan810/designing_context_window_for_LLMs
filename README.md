# Context Window Optimization for LLMs

A modular NLP research project that explores **how to optimally select context** for small LLMs (TinyLlama) to answer questions — balancing **accuracy** against **token usage** — using heuristic and lightweight reinforcement learning methods.

Designed to run on a **MacBook Air (Apple Silicon, CPU/MPS)** with no GPU required.

---

## Project Overview

```
Input Story → Chunking → Context Selection → TinyLlama Inference → Evaluation
                              ↑
              [Full | Truncated | TopK | Sliding Window | Keyword | RL]
```

The key insight: feeding the **entire document** to a small LLM wastes context, confuses it, and is slow. Smart selection of the **most relevant chunks** improves both efficiency and accuracy.

---

## Project Structure

```
NLP_DL/
├── config.yaml                  # All hyperparameters in one place
├── requirements.txt
│
├── data/
│   ├── raw_stories/             # .txt story files
│   └── processed/               # Generated dataset.json + embedding cache
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py    # Load/save dataset JSON
│   │   └── qa_generator.py      # Synthetic dataset + optional LLM-based QA
│   │
│   ├── models/
│   │   ├── tinyllama.py         # TinyLlama wrapper (MPS/CPU)
│   │   └── embeddings.py        # SentenceTransformer wrapper + cache
│   │
│   ├── selectors/
│   │   ├── base_selector.py     # Abstract base class
│   │   ├── topk_selector.py     # Embedding similarity (+ hybrid TF-IDF)
│   │   ├── sliding_window.py    # Sliding window over chunks
│   │   ├── keyword_selector.py  # TF-IDF keyword matching
│   │   └── rl_selector.py       # RL agent wrapper
│   │
│   ├── rl/
│   │   ├── environment.py       # ContextSelectionEnv (OpenAI-gym style)
│   │   └── agent.py             # EpsilonGreedyBandit + PolicyGradientAgent
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # Exact match, substring match, F1, efficiency
│   │   └── evaluator.py         # Full pipeline + baseline selectors
│   │
│   └── utils/
│       ├── chunking.py          # Fixed-size & sentence chunking
│       └── logging.py           # Logger + ResultsLogger
│
├── experiments/
│   ├── run_baselines.py         # Run all heuristic selectors
│   ├── run_rl.py                # Train + evaluate RL selector
│   └── compare_results.py       # Load results, generate plots
│
├── notebooks/
│   └── exploration.ipynb        # Interactive exploration
│
└── results/                     # Auto-generated: JSON, CSV, PNG outputs
```

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (First run) Download NLTK data

```python
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## Running Experiments

All commands should be run from the project root (`NLP_DL/`).

### Run all baseline selectors

```bash
python experiments/run_baselines.py
```

This will:
1. Generate the synthetic dataset (8 stories, ~40 QA pairs) if not present
2. Download TinyLlama on first run (~2.2GB)
3. Run 7 context selection methods
4. Save results to `results/baselines_results.json` and `.csv`
5. Print a summary table

Expected output (example):
```
======================================================================
Method                    Sub.Match      F1  Avg Tokens   Efficiency
----------------------------------------------------------------------
topk (semantic)               0.680   0.512      312.4     0.002177
topk (hybrid)                 0.660   0.498      298.1     0.002213
keyword                       0.620   0.481      245.8     0.002523
sliding_window                0.600   0.462      389.2     0.001542
truncated_head_tail           0.540   0.421      410.5     0.001316
truncated_head                0.480   0.380      220.1     0.002181
full_context                  0.560   0.440      892.3     0.000628
truncated_tail                0.420   0.335      220.1     0.001908
======================================================================
```

### Run the RL selector

```bash
# Epsilon-greedy bandit (fastest, recommended for Mac)
python experiments/run_rl.py --agent bandit --episodes 200

# Policy gradient (slower but more flexible)
python experiments/run_rl.py --agent pg --episodes 200
```

### Compare and visualize all results

```bash
python experiments/compare_results.py
```

Generates in `results/`:
- `accuracy_vs_tokens.png` — scatter: accuracy vs token count
- `efficiency_comparison.png` — bar: efficiency score per method
- `accuracy_comparison.png` — grouped bar: accuracy metrics
- `summary_table.csv`

---

## Context Selection Methods

| Method | Description | Strength |
|--------|-------------|----------|
| **Full Context** | All chunks concatenated | Highest recall (baseline) |
| **Head / Tail / Head+Tail** | Fixed number of chunks from start/end | Very fast, no model needed |
| **Top-K (Semantic)** | Top K chunks by cosine similarity to question | Good precision |
| **Top-K (Hybrid)** | Blends embedding + TF-IDF scores | Best general performance |
| **Sliding Window** | Scores overlapping windows, keeps top-N | Preserves local coherence |
| **Keyword** | TF-IDF keyword matching + neighbours | Interpretable, no model |
| **RL Bandit** | Learns optimal chunk positions via reward signal | Adapts to dataset |
| **RL Policy Gradient** | Neural policy network (NumPy) | Most flexible |

---

## RL Design

**State:**
```
[question_embedding (384d) | mean_selected_embedding (384d) | selection_mask (max_chunks)]
```

**Actions:** Select chunk i | Stop

**Reward:**
```
reward = cosine_sim(selected_context, gold_answer) - λ × token_fraction
```

The surrogate reward (embedding similarity) avoids running the LLM during training, making training 100x faster. The λ penalty discourages selecting unnecessary chunks.

---

## Key Hyperparameters (`config.yaml`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 150 | Tokens per chunk |
| `overlap` | 20 | Overlapping tokens between chunks |
| `k` | 3 | Chunks selected by Top-K |
| `alpha` | 0.5 | Semantic/lexical blend in Top-K hybrid |
| `window_size` | 3 | Chunks per sliding window |
| `lambda_penalty` | 0.001 | RL token count penalty |
| `max_chunks` | 5 | Max chunks RL agent can select |
| `epsilon` | 0.3 | Initial exploration rate |

---

## Performance Notes (Apple Silicon)

- TinyLlama loads in ~10s on MPS, runs each QA in ~1-3s
- Embeddings are cached to disk after first computation
- Full baseline run: ~15-30 min (40 QA pairs × 7 methods)
- RL training (200 episodes): ~2-5 min (uses embedding reward, no LLM)
- Use `use_fp16: true` in config for 2x speedup on MPS

---

## Example Output

```
Question: What elements did Marie Curie discover?
Gold:     polonium and radium
Context (Top-K, 2 chunks):
  "She discovered two elements: polonium, named after her homeland Poland..."
Prediction: polonium and radium
Match: ✓  Tokens: 187
```

---

## Optional: LLM-Based QA Generation

If you have an OpenAI API key, you can generate higher-quality QA pairs:

```python
from src.data.qa_generator import generate_qa_with_llm

pairs = generate_qa_with_llm(
    story="Your story text...",
    api_key="sk-...",
    num_pairs=5
)
```

This is optional — the synthetic dataset is sufficient for experiments.
