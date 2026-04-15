"""
Run all heuristic context selection baselines.

Usage:
    python experiments/run_baselines.py [--config config.yaml] [--verbose]

Outputs:
    results/baselines_<timestamp>.json
    results/baselines_<timestamp>.csv
"""
import argparse
import json
import os
import sys

import yaml

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import load_dataset, save_dataset
from src.data.qa_generator import build_synthetic_dataset
from src.evaluation.evaluator import Evaluator, FullContextSelector, TruncatedSelector
from src.models.embeddings import EmbeddingModel
from src.models.tinyllama import TinyLlamaModel
from src.selectors.keyword_selector import KeywordSelector
from src.selectors.sliding_window import SlidingWindowSelector
from src.selectors.topk_selector import TopKSelector
from src.utils.logging import ResultsLogger, get_logger

logger = get_logger("run_baselines")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dataset", default=None, help="Override dataset path")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset_path = args.dataset or cfg["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found — generating synthetic dataset…")
        build_synthetic_dataset(dataset_path)

    dataset = load_dataset(dataset_path)
    logger.info(f"Dataset: {len(dataset)} stories, "
                f"{sum(len(d['qa_pairs']) for d in dataset)} QA pairs")

    # ------------------------------------------------------------------
    # 2. Models
    # ------------------------------------------------------------------
    logger.info("Loading embedding model…")
    emb_model = EmbeddingModel(
        model_id=cfg["embeddings"]["model_id"],
        cache_dir=cfg["embeddings"]["cache_dir"],
    )

    logger.info("Loading TinyLlama…")
    llm = TinyLlamaModel(
        model_id=cfg["model"]["tinyllama_model_id"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
        use_fp16=cfg["model"]["use_fp16"],
    )

    # ------------------------------------------------------------------
    # 3. Selectors
    # ------------------------------------------------------------------
    chunk_cfg = cfg["chunking"]
    sel_cfg = cfg["selectors"]

    selectors_with_params = [
        (
            FullContextSelector(),
            {"mode": "full"},
        ),
        (
            TruncatedSelector(mode="head", max_chunks=3),
            {"mode": "head", "max_chunks": 3},
        ),
        (
            TruncatedSelector(mode="tail", max_chunks=3),
            {"mode": "tail", "max_chunks": 3},
        ),
        (
            TruncatedSelector(mode="head_tail", max_chunks=4),
            {"mode": "head_tail", "max_chunks": 4},
        ),
        (
            TopKSelector(
                embedding_model=emb_model,
                k=sel_cfg["topk"]["k"],
                alpha=1.0,  # pure semantic
            ),
            {"k": sel_cfg["topk"]["k"], "alpha": 1.0, "variant": "semantic"},
        ),
        (
            TopKSelector(
                embedding_model=emb_model,
                k=sel_cfg["topk"]["k"],
                alpha=sel_cfg["topk"]["alpha"],
            ),
            {"k": sel_cfg["topk"]["k"], "alpha": sel_cfg["topk"]["alpha"], "variant": "hybrid"},
        ),
        (
            SlidingWindowSelector(
                embedding_model=emb_model,
                window_size=sel_cfg["sliding_window"]["window_size"],
                stride=sel_cfg["sliding_window"]["stride"],
                top_n=sel_cfg["sliding_window"]["top_n"],
            ),
            {
                "window_size": sel_cfg["sliding_window"]["window_size"],
                "stride": sel_cfg["sliding_window"]["stride"],
                "top_n": sel_cfg["sliding_window"]["top_n"],
            },
        ),
        (
            KeywordSelector(
                num_keywords=sel_cfg["keyword"]["num_keywords"],
                neighbor_chunks=sel_cfg["keyword"]["neighbor_sentences"],
            ),
            {
                "num_keywords": sel_cfg["keyword"]["num_keywords"],
                "neighbor_chunks": sel_cfg["keyword"]["neighbor_sentences"],
            },
        ),
    ]

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    results_logger = ResultsLogger(cfg["evaluation"]["results_dir"])
    evaluator = Evaluator(
        llm=llm,
        chunk_size=chunk_cfg["chunk_size"],
        overlap=chunk_cfg["overlap"],
        results_logger=results_logger,
    )

    all_metrics = []
    for selector, params in selectors_with_params:
        metrics = evaluator.evaluate_selector(
            selector=selector,
            dataset=dataset,
            hyperparams=params,
            verbose=args.verbose,
        )
        all_metrics.append(metrics)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    saved_path = results_logger.save("baselines_results.json")
    logger.info(f"\nResults saved → {saved_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'Sub.Match':>9} {'F1':>7} {'Avg Tokens':>11} {'Efficiency':>11}")
    print("-" * 70)
    for m in sorted(all_metrics, key=lambda x: -x["substring_match"]):
        print(
            f"{m['method']:<25} {m['substring_match']:>9.3f} "
            f"{m['f1']:>7.3f} {m['avg_tokens']:>11.1f} "
            f"{m['efficiency']:>11.6f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
