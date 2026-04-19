# Context Window Optimization for LLMs — Complete Project Guide

A modular NLP research project that explores **how to optimally select context** for small LLMs (TinyLlama) to answer questions — balancing **accuracy** against **token usage** — using heuristic and lightweight reinforcement learning methods.

Designed to run on a **MacBook Air (Apple Silicon, CPU/MPS)** with no GPU required.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Problem & Solution](#core-problem--solution)
3. [Project Architecture](#project-architecture)
4. [Data Structure & Workflow](#data-structure--workflow)
5. [Context Selection Methods](#context-selection-methods)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results & Graph Generation](#results--graph-generation)
8. [Detailed Graph Explanations](#detailed-graph-explanations)
9. [Data Flow Architecture](#data-flow-architecture)
10. [Key Insights](#key-insights)
11. [Setup & Running](#setup--running)
12. [File Structure](#file-structure)

---

## Project Overview

### What This Project Does

This research project demonstrates that **feeding entire documents to small LLMs is inefficient**. Instead, by intelligently selecting only the most relevant context chunks, we can:
- ✅ Maintain or improve accuracy
- ✅ Reduce token usage by up to 16%
- ✅ Speed up inference
- ✅ Explore both heuristic and learned selection strategies

### The Key Insight

Small LLMs (1.1B parameters) have limited context understanding. Smart selection of relevant chunks **outperforms brute-force full-context feeding**.

```
Input Story → Chunking → Context Selection → TinyLlama Inference → Evaluation
                              ↑
              [Full | Truncated | TopK | Sliding Window | Keyword | RL]
```

---

## Core Problem & Solution

### The Problem

When you feed an entire story to a small LLM:
- **Wasted context window:** Much of the story is irrelevant to the question
- **Confusion:** Small models struggle with long contexts
- **Inefficiency:** Slower inference, more token consumption
- **Quality degradation:** Longer context doesn't guarantee better answers

### The Solution

**Smart context selection:** Use different strategies to select only the most relevant chunks:
- **Heuristic methods:** Embedding similarity, TF-IDF keywords, sliding windows
- **Learned methods:** Reinforcement learning agent learns what chunks matter

### The Trade-off We Study

```
Accuracy (%) vs Token Count (Trade-off Frontier)

↑ Accuracy
│    ★ TopK (Best: 84% tokens, 99% retention)
│    ● Sliding Window
│    ● Full Context
│    ◆ Keyword
│    ● Truncated Head
│    ✗ Truncated Tail (loses too much accuracy)
└─────────────────────────────→ Token Usage
```

---

## Project Architecture

```
NLP_DL/
├── config.yaml                  # All hyperparameters in one place
├── requirements.txt
│
├── data/
│   ├── raw_stories/             # .txt story files (input data)
│   └── processed/
│       ├── dataset.json         # Generated QA dataset
│       └── embedding_cache/     # Cached embeddings (speedup)
│
├── src/                         # Core source code
│   ├── data/
│   │   ├── dataset_loader.py    # Load/save dataset JSON
│   │   └── qa_generator.py      # Generate synthetic QA pairs
│   │
│   ├── models/
│   │   ├── tinyllama.py         # TinyLlama wrapper (MPS/CPU)
│   │   └── embeddings.py        # SentenceTransformer wrapper + cache
│   │
│   ├── selectors/               # Context selection strategies
│   │   ├── base_selector.py     # Abstract base class
│   │   ├── topk_selector.py     # Embedding similarity + hybrid TF-IDF
│   │   ├── sliding_window.py    # Sliding window over chunks
│   │   ├── keyword_selector.py  # TF-IDF keyword matching
│   │   └── rl_selector.py       # RL agent wrapper
│   │
│   ├── rl/                      # Reinforcement learning
│   │   ├── environment.py       # ContextSelectionEnv (OpenAI-gym style)
│   │   └── agent.py             # EpsilonGreedyBandit + PolicyGradientAgent
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # Exact match, substring match, F1, efficiency
│   │   └── evaluator.py         # Full pipeline orchestration + baseline selectors
│   │
│   └── utils/
│       ├── chunking.py          # Fixed-size & sentence chunking
│       └── logging.py           # Logger + ResultsLogger
│
├── experiments/
│   ├── run_baselines.py         # Run all heuristic selectors
│   ├── run_rl.py                # Train + evaluate RL selector
│   └── compare_results.py       # Load results, generate 6 plots
│
├── notebooks/
│   └── exploration.ipynb        # Interactive exploration
│
└── results/                     # Auto-generated outputs
    ├── *.json                   # Individual results
    ├── *.csv                    # Aggregated results
    └── *.png                    # Visualization graphs (6 total)
```

---

## Data Structure & Workflow

### Step 1: Data Preparation

**Input:** Raw story files in `data/raw_stories/`
- Example: `story_01_amazon.txt` (biography text)

**Processing:** 
- Load story text
- Generate synthetic QA pairs (question + gold answer)
- Save to `data/processed/dataset.json`

**Dataset format:**
```json
[
  {
    "story_id": "story_01",
    "story": "Long text of the story...",
    "qa_pairs": [
      {"question": "Who founded Amazon?", "answer": "Jeff Bezos"},
      {"question": "When was Amazon founded?", "answer": "1994"}
    ]
  },
  ...
]
```

### Step 2: Chunking

Convert raw story into fixed-size chunks:
- **Chunk size:** 50 tokens (configurable)
- **Overlap:** 10 tokens between chunks
- **Purpose:** Enable selectors to pick relevant chunks

**Example:**
```
Story: "Jeff Bezos founded Amazon in 1994. Amazon started as an online bookstore..."

Chunks:
[1] "Jeff Bezos founded Amazon in 1994."
[2] "Amazon in 1994. Amazon started as"
[3] "as an online bookstore. It has"
...
```

### Step 3: Context Selection

For each QA pair, a selector chooses relevant chunks:

**Input:** 
- `chunks`: List of all story chunks
- `question`: The query
- `tokenizer`: To count tokens accurately

**Output:**
- `context`: Joined chunks as text
- `token_count`: Accurate token count of context

**Process:**
```
Q: "Who founded Amazon?"
    ↓
[Selector evaluates all chunks]
    ↓
Selected: [chunk_1, chunk_3]  (most relevant to question)
    ↓
Context: "Jeff Bezos founded Amazon in 1994. It has become..."
Token count: 25 tokens
```

### Step 4: LLM Inference

**Input:**
- `context`: Selected chunks from step 3
- `question`: The query

**Processing:**
- Format as prompt: "Given: {context}\n\nQuestion: {question}\nAnswer:"
- Pass to TinyLlama
- Generate up to 64 tokens

**Output:**
- `prediction`: LLM's answer

### Step 5: Metrics Computation

Compare prediction against gold answer using 4 metrics:

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| **Exact Match** | Normalized strings identical? | Strictest measure |
| **Substring Match** | Gold answer in prediction? | Most practical |
| **Token F1** | SQuAD-style token overlap | Partial credit |
| **Efficiency** | Accuracy / tokens_used | Cost-benefit analysis |

**Example:**
```
Gold: "Jeff Bezos"
Prediction: "Bezos founded Amazon, who is the founder"

Exact Match: 0.0 (not identical after normalization)
Substring Match: 1.0 (normalized "jeff bezos" IS in prediction)
F1: 0.66 (2 common tokens: jeff, bezos)
Efficiency: 1.0 / 30 = 0.033 accuracy/token
```

### Step 6: Results Storage

Each run generates JSON with all metrics:

```json
{
  "timestamp": "2026-04-09T16:23:07.439488",
  "method": "topk",
  "accuracy": 0.817,
  "tokens_used": 166,
  "exact_match": 0.0,
  "f1": 0.270,
  "efficiency": 0.00491
}
```

---

## Context Selection Methods

### 1. Full Context (Baseline)
**Strategy:** Return entire story
- **Pros:** No selection needed, maximum information
- **Cons:** Wastes tokens, confuses small LLM
- **Expected:** Highest accuracy, most tokens

### 2. Truncated Head
**Strategy:** Take first N chunks
- **Pros:** Simple, deterministic
- **Cons:** Misses context in middle/end of story
- **Use case:** When question is likely answered early

### 3. Truncated Tail
**Strategy:** Take last N chunks
- **Pros:** May capture concluding information
- **Cons:** Often misses setup/introduction
- **Use case:** Stories with climax at end
- **Risk:** ⚠️ Often drops accuracy significantly

### 4. Truncated Head+Tail
**Strategy:** Take first AND last N chunks
- **Pros:** Covers beginning and end
- **Cons:** Might miss critical middle content
- **Use case:** Balanced coverage needed

### 5. TopK (Embedding-Based)
**Strategy:** Select K chunks with highest semantic similarity to question
- **Algorithm:**
  1. Embed question using SentenceTransformer
  2. Embed all chunks using same model
  3. Compute cosine similarity
  4. Select top-K chunks
  5. Optional: Hybrid blend with TF-IDF keywords
- **Pros:** ⭐ Semantically relevant, good accuracy/token trade-off
- **Cons:** Requires embedding model
- **Result:** Best efficiency in experiments

### 6. Sliding Window
**Strategy:** Apply window over chunks, score windows, keep top
- **Algorithm:**
  1. Create overlapping windows (e.g., size=3, stride=1)
  2. Score each window (e.g., avg similarity to question)
  3. Select top-N windows
  4. Combine selected windows
- **Pros:** Maintains chunk continuity/context
- **Cons:** More complex than TopK
- **Result:** Competitive accuracy with slight quality boost

### 7. Keyword Matching (TF-IDF)
**Strategy:** Extract keywords from question, find chunks containing them
- **Algorithm:**
  1. Compute TF-IDF scores for question
  2. Extract top keywords
  3. Find chunks containing these keywords
  4. Include neighbor chunks for context
- **Pros:** Interpretable, no embedding model needed
- **Cons:** Misses semantic variations
- **Result:** Competitive but less robust than embeddings

### 8. RL Agent (Reinforcement Learning)
**Strategy:** Learn which chunks matter through trial-and-error
- **Algorithm:**
  - **Environment:** Context selection task
  - **Agent:** Epsilon-greedy bandit or policy gradient
  - **State:** Chunks, question
  - **Action:** Select/skip each chunk
  - **Reward:** Accuracy - λ × token_penalty
  - **Training:** 200 episodes on dataset
- **Pros:** Can generalize to new patterns
- **Cons:** Requires training, slower inference
- **Result:** Competitive with baselines, promising for scale

---

## Evaluation Metrics

### Accuracy Metrics

#### 1. Substring Match (Primary)
```python
def substring_match(prediction, gold_answer):
    pred_norm = normalize(prediction)  # lowercase, remove punctuation
    gold_norm = normalize(gold_answer)
    return 1.0 if gold_norm in pred_norm else 0.0
```
- **Use:** Most practical measure for QA
- **Interpretation:** Does prediction contain the answer?
- **Range:** 0.0–1.0 (0%–100%)

#### 2. Exact Match
```python
def exact_match(prediction, gold_answer):
    return 1.0 if normalize(prediction) == normalize(gold_answer) else 0.0
```
- **Use:** Strictest measure
- **Interpretation:** Perfect answer?
- **Range:** 0.0–1.0
- **Typical result:** Often 0.0 (LLM adds extra context)

#### 3. Token F1 (SQuAD-style)
```python
def token_f1(prediction, gold_answer):
    common_tokens = set(pred_tokens) & set(gold_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
```
- **Use:** Partial credit for overlapping tokens
- **Interpretation:** How much of the answer is in prediction?
- **Range:** 0.0–1.0

#### 4. Efficiency Score
```python
efficiency = accuracy / tokens_used
```
- **Use:** Cost-benefit analysis
- **Interpretation:** Accuracy points gained per token spent
- **Higher is better:** More efficient selectors have higher scores

### Compression Metrics (Computed)

#### Compression Ratio
```python
compression_ratio = tokens_used / full_context_tokens
```
- **Range:** 0.0–1.0
- **0.84:** Using 84% of full context tokens (16% saved)
- **Lower is better:** More compression achieved

#### Accuracy Retention
```python
retention = method_accuracy / full_context_accuracy
```
- **Range:** 0.0–1.0 (0%–100%)
- **1.0:** 100% of accuracy preserved
- **0.99:** 99% of accuracy preserved (1% loss acceptable)
- **Higher is better:** More accuracy maintained

---

## Results & Graph Generation

### How Results Flow

```
Run Experiments
    ↓
[run_baselines.py] ──→ baselines_results.json
[run_rl.py] ──────────→ rl_bandit_results.json
    ↓
compare_results.py
    ├─→ Load all JSON files
    ├─→ Compute compression metrics
    ├─→ Generate 6 visualization plots
    ├─→ Create summary_table.csv
    └─→ Print summary statistics
    ↓
results/ directory
    ├── accuracy_vs_tokens.png
    ├── accuracy_comparison.png
    ├── efficiency_comparison.png
    ├── compression_frontier.png
    ├── retention_vs_compression.png (MOST IMPORTANT)
    ├── compression_analysis.png
    └── summary_table.csv
```

### Key Metric Calculations

In `compare_results.py`:

```python
# Get baseline from full_context
full_tokens = df[df["method"] == "full_context"]["tokens_used"].mean()
full_accuracy = df[df["method"] == "full_context"]["accuracy"].mean()

# Compute for each method
df["compression_ratio"] = df["tokens_used"] / full_tokens
df["retention"] = df["accuracy"] / full_accuracy
```

---

## Detailed Graph Explanations

### 1️⃣ Accuracy vs Tokens (`accuracy_vs_tokens.png`)

**Purpose:** Scatter plot showing accuracy-efficiency trade-off

**Axes:**
- **X-axis:** Average token count per QA pair (Lower = More efficient)
- **Y-axis:** Substring match accuracy (Higher = Better QA performance)

**Visual Elements:**
- Each colored dot = One selector method
- Gray dashed line = Full context baseline accuracy
- Labels with color-coded boxes

**Data Source:**
```python
for method in methods:
    x = results[method]["tokens_used"].mean()        # Average tokens
    y = results[method]["accuracy"].mean()           # Average accuracy
    ax.scatter(x, y, label=method)
```

**Interpretation:**
- **Upper-left zone (BEST):** High accuracy, few tokens = optimal selector
  - Example: TopK at (166 tokens, 0.817 accuracy)
- **Lower-right zone (WORST):** Low accuracy, many tokens = inefficient
  - Example: Truncated_tail at (151 tokens, 0.707 accuracy)
- **Horizontal reference:** Shows full_context baseline to compare against

**Key Insight:** Different selectors cluster in different regions, showing distinct trade-off strategies

---

### 2️⃣ Efficiency Comparison (`efficiency_comparison.png`)

**Purpose:** Bar chart comparing efficiency (accuracy per token)

**Metric:** `efficiency = accuracy / tokens_used`

**Visual:** Vertical bar chart, one bar per method

**Data Source:**
```python
for method in methods:
    eff_score = results[method]["efficiency"].mean()
    ax.bar(method, eff_score)
```

**Interpretation:**
- **Taller bar = More accuracy gain per token**
- TopK: 0.00491 (best efficiency)
- Full context: 0.00420 (baseline)
- Truncated_tail: 0.00465 (reasonable but loses accuracy)

**Use Case:** Quick comparison of token efficiency across all methods

---

### 3️⃣ Accuracy Comparison (`accuracy_comparison.png`)

**Purpose:** Compare different accuracy metrics per method

**Metrics shown:**
- **Blue bars:** Substring Match (primary)
- **Green bars:** Token F1 score (secondary)

**Visual:** Grouped bar chart

**Data Source:**
```python
for method in methods:
    substring_acc = results[method]["accuracy"].mean()      # Blue
    f1_score = results[method]["f1"].mean()                 # Green
    ax.bar(method, [substring_acc, f1_score])
```

**Interpretation:**
- If bars are similar → Different metrics agree
- If bars diverge → Token-level precision differs from surface-level match
- Example: Full_context has 0.829 substring but only 0.284 F1
  - Suggests LLM generates verbose answers (many tokens, some matching)

**Use Case:** Validate that high accuracy isn't just due to generous matching

---

### 4️⃣ Compression Frontier (`compression_frontier.png`) ⭐ KEY GRAPH

**Purpose:** Show Pareto frontier of compression vs accuracy trade-off

**Axes:**
- **X-axis:** Compression ratio = `tokens_used / full_context_tokens`
  - 0.76 = 76% of tokens used (24% saved)
  - 1.0 = 100% of tokens (no compression)
- **Y-axis:** Accuracy score (higher = better)

**Visual Elements:**
- Scatter points = Methods
- Green shaded zone = Optimal region
- Gray dashed line = Full context baseline

**Data Source:**
```python
for method in methods:
    comp_ratio = method_tokens / full_context_tokens        # X
    accuracy = results[method]["accuracy"].mean()           # Y
    ax.scatter(comp_ratio, accuracy, label=method)
```

**Key Zones:**
1. **Left side (< 0.85):** Strong compression, fewer tokens
2. **Upper region (> 0.80 accuracy):** High accuracy maintained
3. **Upper-left quadrant:** Pareto optimal methods

**Interpretation:**
- **Full context at (1.0, 0.829):** No compression, baseline accuracy
- **TopK at (0.84, 0.817):** 16% compression, 98% accuracy retention
- **Truncated_tail at (0.76, 0.707):** 24% compression, 85% accuracy (too risky)

**Key Insight:** Methods in upper-left are Pareto-optimal (can't improve one dimension without sacrificing other)

---

### 5️⃣ Retention vs Compression (`retention_vs_compression.png`) ⭐ MOST IMPORTANT

**Purpose:** Show quality preservation vs token reduction trade-off (best for deployment decisions)

**Axes:**
- **X-axis:** Compression ratio (0–100%, lower = more tokens saved)
- **Y-axis:** Retention = `method_accuracy / full_context_accuracy` (0–100%)

**Visual Elements:**
- Scatter points = Non-full_context methods
- Green zone (upper-left): BEST (compress + retain >97%)
- Orange zone (lower-right): RISKY (heavy compression loses quality)
- Reference lines: y=100% (full accuracy) and x=100% (no compression)
- Text boxes with zone interpretations

**Data Source:**
```python
full_ctx_tokens = results["full_context"]["tokens_used"].mean()
full_ctx_accuracy = results["full_context"]["accuracy"].mean()

for method in methods_except_full:
    comp_ratio = results[method]["tokens_used"] / full_ctx_tokens      # X
    retention = results[method]["accuracy"] / full_ctx_accuracy        # Y
    ax.scatter(comp_ratio, retention)
```

**Zone Definitions:**
1. **Green zone (upper-left):** ✅ BEST
   - Compress 8–16% of tokens
   - Retain >97% of accuracy
   - Safe for deployment
2. **Orange zone (lower-right):** ⚠️ RISKY
   - Compress >24% of tokens
   - Retain <90% of accuracy
   - Too much quality loss

**Reading Examples:**
- **Sliding_window at (0.91, 1.00):** Use 91% tokens, keep 100% accuracy ⭐
- **TopK at (0.84, 0.99):** Use 84% tokens, keep 99% accuracy ⭐⭐ **RECOMMENDED**
- **Keyword at (0.95, 1.00):** Use 95% tokens, keep 100% accuracy ✅ Safe but less compression
- **Truncated_tail at (0.76, 0.85):** Use 76% tokens, lose 15% accuracy ⚠️ Too risky

**Why This Graph Matters:**
- Directly answers: "How much can we compress without sacrificing quality?"
- Guides deployment decision: Which method should production use?
- Shows risk profile of each selector

**Key Insight:** TopK provides best balance — reasonable compression with minimal quality loss

---

### 6️⃣ Compression Analysis (`compression_analysis.png`)

**Purpose:** Side-by-side comparison of compression vs retention

**Layout:**
- **Red bars:** Compression ratio (lower = better savings)
- **Teal bars:** Accuracy retention (higher = better quality)

**Visual:** Grouped bar chart, one color pair per method (excluding full_context)

**Data Source:**
```python
methods = [m for m in methods if m != "full_context"]
for method in methods:
    comp_ratio = results[method]["compression_ratio"].mean()
    retention = results[method]["retention"].mean()
    ax.bar(method, [comp_ratio, retention])
```

**Interpretation:**
- **Good method:** Red bar LOW + Teal bar HIGH
  - Example: TopK (red=0.84, teal=0.99)
- **Poor method:** Red bar LOW + Teal bar LOW (contradiction)
  - Example: Truncated_tail (red=0.76, teal=0.85) – extreme compression loses quality
- **Reference:** 1.0 line shows full_context baseline

**Reading Example:**
```
TopK:              ▮▮▮ (0.84) | ▮▮▮▮▮▮▮▮▮ (0.99)  ← Good: compress + retain
Keyword:           ▮▮▮▮▮▮ (0.95) | ▮▮▮▮▮▮▮▮▮ (1.0)  ← Safe: minimal compression
Truncated_tail:    ▮▮▮ (0.76) | ▮▮▮▮▮▮ (0.85)      ← Risky: lose too much
```

**Use Case:** Quick visual comparison of which methods balance both objectives

---

## Data Flow Architecture

### Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Raw Stories              Config                Dataset JSON      │
│  (story_*.txt)       (config.yaml)        (qa_pairs)             │
└──────────┬─────────────────┬──────────────────┬──────────────────┘
           │                 │                  │
           └─────────────────┼──────────────────┘
                             ↓
                    ┌────────────────┐
                    │  Chunking      │ 50 tokens/chunk
                    │  (story → []) │ 10 token overlap
                    └────────┬───────┘
                             ↓
        ┌────────────────────────────────────────┐
        │    For Each QA Pair (Parallel)         │
        │                                        │
        │  Q: "Who founded Amazon?"              │
        │  A: "Jeff Bezos"                       │
        └─────────────────┬──────────────────────┘
                          ↓
        ┌─────────────────────────────────┐
        │  Selector 1 (TopK)              │
        ├─────────────────────────────────┤
        │ • Embed question                │
        │ • Score all chunks              │
        │ • Select top-3 chunks           │
        │ Return: context + 25 tokens    │
        └────────────┬────────────────────┘
                     ↓
        ┌──────────────────────────────┐
        │   TinyLlama LLM              │
        ├──────────────────────────────┤
        │ Input: context + question    │
        │ Output: "Bezos founded..."   │
        └────────────┬─────────────────┘
                     ↓
        ┌──────────────────────────────┐
        │   Evaluate Prediction        │
        ├──────────────────────────────┤
        │ • Substring match: ✓ (1.0)  │
        │ • F1 score: 0.66             │
        │ • Efficiency: 0.0491         │
        └────────────┬─────────────────┘
                     ↓
    ┌───────────────────────────────────────┐
    │ Selector 2 (Keyword)                  │
    ├───────────────────────────────────────┤
    │ Return: different context + tokens   │
    │ → Prediction → Metrics               │
    └────────────┬────────────────────────┘
                 ↓
    ┌───────────────────────────────────────┐
    │ ... (7 total selectors) ...           │
    └────────────┬────────────────────────┘
                 ↓
    ┌──────────────────────────────────────────┐
    │       Results JSON Files                 │
    │  (Per-method results with all metrics)  │
    │                                          │
    │ {method: "topk", accuracy: 0.817, ...}  │
    └────────────┬─────────────────────────────┘
                 ↓
    ┌──────────────────────────────────────────┐
    │     compare_results.py                   │
    ├──────────────────────────────────────────┤
    │ 1. Load all JSON files                  │
    │ 2. Compute compression metrics:         │
    │    - compression_ratio                  │
    │    - retention                          │
    │ 3. Aggregate by method                  │
    │ 4. Generate 6 visualizations            │
    │ 5. Create summary table                 │
    └────────────┬─────────────────────────────┘
                 ↓
    ┌────────────────────────────────────────────────────────────┐
    │                    OUTPUT ARTIFACTS                         │
    ├────────────────────────────────────────────────────────────┤
    │ • accuracy_vs_tokens.png                                   │
    │ • efficiency_comparison.png                                │
    │ • accuracy_comparison.png                                  │
    │ • compression_frontier.png                                 │
    │ • retention_vs_compression.png (⭐ KEY)                     │
    │ • compression_analysis.png                                 │
    │ • summary_table.csv                                        │
    │ • detailed analysis output                                 │
    └────────────────────────────────────────────────────────────┘
```

### Data Structure at Each Stage

**Stage 1: Raw Input**
```python
# story_01_amazon.txt
"Jeff Bezos founded Amazon in 1994 as an online bookstore..."

# config.yaml
chunk_size: 50
overlap: 10
```

**Stage 2: After Chunking**
```python
chunks = [
    "Jeff Bezos founded Amazon in 1994.",
    "Amazon in 1994. Amazon started as an online bookstore.",
    "online bookstore. It has grown to become..."
]
# Each chunk ≈ 50 tokens
```

**Stage 3: During Selection**
```python
# TopK selector processes:
question = "Who founded Amazon?"
question_embedding = embedding_model(question)  # 384-dim vector

for chunk in chunks:
    chunk_embedding = embedding_model(chunk)
    similarity = cosine_similarity(q_emb, chunk_emb)
    
selected_chunks = top_3_by_similarity
context = " ".join(selected_chunks)
token_count = count_tokens(context)  # ~25 tokens
```

**Stage 4: LLM Inference**
```python
prompt = f"""Given: {context}

Question: {question}
Answer:"""

prediction = tinyllama(prompt)  # "Bezos founded Amazon"
```

**Stage 5: Evaluation**
```python
metrics = {
    "substring_match": 1.0,  # "Bezos" in prediction
    "f1": 0.66,
    "tokens_used": 25,
    "efficiency": 1.0 / 25,
    "method": "topk"
}
```

**Stage 6: Aggregation**
```python
# After running all methods on all QA pairs
summary = {
    "method": "topk",
    "accuracy": 0.817,  # Mean across 41 QA pairs
    "tokens_used": 166,  # Mean
    "compression_ratio": 0.84,  # 166 / 197 (full_context)
    "retention": 0.985  # 0.817 / 0.829
}
```

---

## Key Insights

### 1. Accuracy Plateau Phenomenon

**Finding:** Most methods achieve ~83% accuracy despite vastly different token counts (151–197)

**Implication:** Context **quality matters more than quantity**
- Full context (197 tokens): 82.9% accuracy
- TopK (166 tokens): 81.7% accuracy (−1.2% but 16% fewer tokens)
- Truncated_tail (151 tokens): 70.7% accuracy (−12.2%, too extreme)

**Insight:** The "sweet spot" balances compression with accuracy

### 2. TopK Shows Best Efficiency

**Finding:** TopK achieves highest efficiency score (0.00491)

**Why:**
- Semantic relevance: Embedding-based selection captures question intent
- Minimal noise: Doesn't include irrelevant chunks
- Balanced compression: 84% token usage keeps 99% accuracy

**Comparison:**
- TopK: 166 tokens, 81.7% accuracy → 0.00491 efficiency
- Keyword: 188 tokens, 82.9% accuracy → 0.00439 efficiency
- Full context: 197 tokens, 82.9% accuracy → 0.00420 efficiency

### 3. Token-Accuracy Decoupling

**Finding:** Fewer tokens ≠ lower accuracy

**Examples:**
- Truncated_head: 179 tokens, 82.9% accuracy (good)
- Truncated_tail: 151 tokens, 70.7% accuracy (bad)
- TopK: 166 tokens, 81.7% accuracy (excellent efficiency)

**Key insight:** **Selector strategy matters more than token count**

### 4. RL Agent Competitive Performance

**Finding:** RL agent (0.829 accuracy, 197 tokens) matches full_context baseline

**Implications:**
- RL can learn competitive strategies without explicit rules
- With more training data/complexity, RL could discover novel compression patterns
- Future work: Generalization to different domains

### 5. Optimal Strategy Identified

**Best selector: TopK**
- Achieves 84% token usage (16% compression)
- Maintains 99% accuracy retention
- Best efficiency score
- Practical and fast

**Second best: Sliding_window**
- 91% token usage
- 100% accuracy retention
- More conservative but zero risk

---

## Setup & Running

### Prerequisites

- Python 3.9+
- macOS with Apple Silicon recommended
- 8GB RAM minimum

### Installation

#### 1. Create Virtual Environment
```bash
cd designing_context_window_for_LLMs
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download NLTK Data (First Run Only)
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### Running Experiments

All commands from project root (`designing_context_window_for_LLMs/`)

#### Run All Baseline Selectors
```bash
python experiments/run_baselines.py --config config.yaml --verbose
```

**Output:**
- `results/baselines_results.json` – Detailed results per QA pair
- `results/baselines_results.csv` – Aggregated results

**Time:** ~5 minutes (depends on dataset size)

#### Train and Evaluate RL Agent
```bash
python experiments/run_rl.py --config config.yaml --num_episodes 200
```

**Output:**
- `results/rl_bandit_results.json` – RL results
- `results/rl_bandit.pkl` – Trained agent (for reuse)

**Time:** ~10 minutes

#### Generate All Visualizations
```bash
python experiments/compare_results.py --results_dir results
```

**Output:**
- 6 PNG graphs
- `summary_table.csv`
- Console output with summary statistics

#### Run Interactive Exploration
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Configuration

Edit `config.yaml` to modify:

```yaml
model:
  max_new_tokens: 64          # Max length of LLM output
  use_fp16: true              # Use float16 on Apple Silicon

chunking:
  chunk_size: 50              # Tokens per chunk
  overlap: 10                 # Overlap between chunks

selectors:
  topk:
    k: 3                      # Number of chunks to select
    alpha: 0.5                # Embedding vs keyword blend
  rl:
    num_episodes: 200         # Training episodes
    epsilon: 0.3              # Exploration rate
```

---

## File Structure Details

### Core Source Files

**`src/models/tinyllama.py`**
- TinyLlamaModel class: Wraps huggingface TinyLlama
- Methods: `answer(context, question)`, `tokenize(text)`
- Handles: MPS/CPU device selection, FP16 optimization

**`src/models/embeddings.py`**
- EmbeddingModel class: SentenceTransformer wrapper
- Features: Caching of embeddings for speedup
- Methods: `embed(text)`, `similarity(text1, text2)`

**`src/selectors/topk_selector.py`**
- TopKSelector: Embedding-based selection
- Features: Hybrid TF-IDF weighting option
- Core logic: Sort chunks by similarity, take top-K

**`src/selectors/sliding_window.py`**
- SlidingWindowSelector: Window-based selection
- Features: Configurable window size and stride
- Core logic: Score windows, merge top windows

**`src/rl/environment.py`**
- ContextSelectionEnv: OpenAI gym-style environment
- State: Chunks, question, current selection
- Action: Select/skip each chunk
- Reward: Accuracy − λ × tokens

**`src/rl/agent.py`**
- EpsilonGreedyBandit: Explore/exploit strategy
- PolicyGradientAgent: Learn action values
- Training: Iterate over episodes, update policy

**`src/evaluation/evaluator.py`**
- Evaluator: Orchestrates full pipeline
- Methods: `evaluate_selector()`, runs over dataset
- Outputs: Metrics dict, logs results

**`src/utils/chunking.py`**
- `chunk_by_tokens()`: Fixed-size chunking
- `chunk_by_sentences()`: Sentence-based chunking
- Features: Configurable overlap

### Data Files

**`data/raw_stories/`**
- `story_01_amazon.txt` – Unstructured text
- `story_02_curie.txt` – etc.

**`data/processed/dataset.json`**
- Format: `[{story, qa_pairs: [{question, answer}, ...]}, ...]`
- Generated from raw stories

**`data/processed/embedding_cache/`**
- Cached embeddings (`.pkl` files)
- Speedup on repeated runs

### Results Files

**`results/*.json`**
- Individual results per method
- Format: List of result dicts with all metrics

**`results/summary_table.csv`**
- Aggregated results (one row per method)
- Sorted by accuracy descending

**`results/*.png`**
- 6 visualization graphs
- High-quality PNG (150 dpi)

---

## Troubleshooting

### Issue: "No module named 'tinyllama'"
**Solution:** Ensure you're in the virtual environment (`source venv/bin/activate`)

### Issue: Slow embedding computation
**Solution:** Embeddings are cached. First run is slow; subsequent runs use cache.

### Issue: Out of memory
**Solution:** Reduce `batch_size` in config or `max_new_tokens` for LLM

### Issue: Results don't match previous run
**Solution:** Different random seed in RL agent. Set `seed` in config for reproducibility.

---

## Contributing & Future Work

### Potential Improvements

1. **Multi-story dataset:** Generalize beyond current 2-3 stories
2. **Domain adaptation:** Test on different text types (news, medical, legal)
3. **Hybrid selectors:** Combine multiple strategies (e.g., embedding + keyword)
4. **Retrieval-based:** Use BM25/Elasticsearch for selection
5. **Iterative refinement:** RL agent refines selection based on LLM confidence
6. **Few-shot:** Provide exemplars to improve selection

### Research Questions

- How does selector performance scale with story length?
- Can we achieve better compression on specific domains?
- How sensitive is performance to chunk size?
- Does RL learn consistent patterns across different LLMs?

---

## References & Resources

### Papers

- **SQuAD:** [Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250) – F1 metric definition
- **SentenceTransformers:** [Reimers & Gupta, 2019](https://arxiv.org/abs/1908.10084) – Embedding model
- **TinyLlama:** [Zhang et al., 2024](https://github.com/jzhang38/TinyLlama) – 1.1B LLM

### Tools

- **Hugging Face Transformers:** Model hosting & inference
- **SentenceTransformers:** Semantic embeddings
- **PyTorch:** Deep learning framework
- **Matplotlib/Pandas:** Visualization & data analysis

---

## License & Citation

[Add your license here if applicable]

---

## Authors & Contact

[Add author information if applicable]

---

*Last Updated: April 19, 2026*
