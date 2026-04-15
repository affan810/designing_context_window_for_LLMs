"""
Compare and visualize results across all experiments.

Usage:
    python experiments/compare_results.py [--results_dir results]

Generates:
    results/accuracy_vs_tokens.png
    results/efficiency_comparison.png
    results/summary_table.csv
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#E91E63", "#009688",
]


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all JSON result files and concatenate into a DataFrame."""
    records = []
    for jf in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(jf) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
    if not records:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    return pd.DataFrame(records)


def plot_accuracy_vs_tokens(df: pd.DataFrame, output_path: str) -> None:
    """Scatter plot: accuracy (y) vs average token count (x) per method."""
    methods = df["method"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        x = sub["tokens_used"].mean()
        y = sub["accuracy"].mean()
        color = COLORS[i % len(COLORS)]
        ax.scatter(x, y, s=120, color=color, zorder=5)
        ax.annotate(
            method,
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("Average Token Count", fontsize=12)
    ax.set_ylabel("Accuracy (Substring Match)", fontsize=12)
    ax.set_title("Accuracy vs. Token Usage by Context Selection Method", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved → {output_path}")


def plot_efficiency(df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of efficiency score (accuracy / tokens) per method."""
    methods = df["method"].unique()
    eff_scores = []
    for method in methods:
        sub = df[df["method"] == method]
        eff = sub["efficiency"].mean() if "efficiency" in sub.columns else 0.0
        eff_scores.append(eff)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, eff_scores, color=COLORS[: len(methods)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Efficiency Score (Accuracy / Tokens)", fontsize=11)
    ax.set_title("Context Selection Efficiency Comparison", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    # Annotate bars
    for bar, score in zip(bars, eff_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.00001,
            f"{score:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved → {output_path}")


def plot_accuracy_comparison(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: exact match, substring match, F1 per method."""
    methods = df["method"].unique()
    em = [df[df["method"] == m]["accuracy"].mean() for m in methods]
    f1 = [df[df["method"] == m]["f1"].mean() if "f1" in df.columns else 0 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, em, width, label="Substring Match", color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, f1, width, label="Token F1", color="#4CAF50", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Accuracy Metrics by Method", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved → {output_path}")


def print_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return a summary DataFrame grouped by method."""
    agg_cols = {"accuracy": "mean", "tokens_used": "mean"}
    if "f1" in df.columns:
        agg_cols["f1"] = "mean"
    if "efficiency" in df.columns:
        agg_cols["efficiency"] = "mean"

    summary = df.groupby("method").agg(agg_cols).reset_index()
    summary = summary.sort_values("accuracy", ascending=False)

    print("\n" + "=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(str(results_dir))
    print(f"Loaded {len(df)} result records from {results_dir}")

    summary = print_summary_table(df)
    summary.to_csv(results_dir / "summary_table.csv", index=False)

    plot_accuracy_vs_tokens(df, str(results_dir / "accuracy_vs_tokens.png"))
    plot_efficiency(df, str(results_dir / "efficiency_comparison.png"))
    plot_accuracy_comparison(df, str(results_dir / "accuracy_comparison.png"))

    print(f"\nAll plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
