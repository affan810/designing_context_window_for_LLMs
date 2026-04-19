"""
Compare and visualize results across all experiments.

Usage:
    python experiments/compare_results.py [--results_dir results]

Generates:
    results/accuracy_vs_tokens.png
    results/efficiency_comparison.png
    results/compression_frontier.png
    results/retention_vs_compression.png
    results/compression_analysis.png
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


def compute_compression_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compression ratio and retention metrics.
    
    Compression ratio = tokens_used / full_context_tokens
    Retention = method_accuracy / full_context_accuracy
    """
    df = df.copy()
    
    # Get full_context baseline metrics
    full_context_rows = df[df["method"] == "full_context"]
    if full_context_rows.empty:
        print("Warning: No full_context baseline found. Skipping compression metrics.")
        return df
    
    full_tokens = full_context_rows["tokens_used"].mean()
    full_accuracy = full_context_rows["accuracy"].mean()
    
    # Compute compression ratio and retention
    df["compression_ratio"] = df["tokens_used"] / full_tokens
    df["retention"] = df["accuracy"] / full_accuracy
    
    return df


def plot_accuracy_vs_tokens(df: pd.DataFrame, output_path: str) -> None:
    """Scatter plot: accuracy (y) vs average token count (x) per method."""
    methods = sorted(df["method"].unique())
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect data for smart label positioning
    points_data = []
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        x = sub["tokens_used"].mean()
        y = sub["accuracy"].mean()
        color = COLORS[i % len(COLORS)]
        points_data.append((method, x, y, color))
        ax.scatter(x, y, s=250, color=color, alpha=0.7, edgecolors="black", linewidth=2, zorder=5)
    
    # Smart label placement to avoid overlaps
    for method, x, y, color in points_data:
        # Determine offset based on position to avoid overlapping
        if x < 180:  # Left side
            xytext = (-15, 15)
        elif x > 190:  # Right side
            xytext = (15, 15)
        else:  # Middle
            xytext = (-15, -20)
        
        ax.annotate(
            method,
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            fontsize=10,
            weight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor=color, linewidth=1.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color, lw=1.5)
        )

    # Add reference zones
    ax.axhline(y=df[df["method"] == "full_context"]["accuracy"].mean(), 
              color="gray", linestyle="--", alpha=0.4, linewidth=1.5, label="Full context accuracy")
    
    ax.set_xlabel("Average Token Count per QA Pair\n(Lower = More efficient context compression)", 
                 fontsize=12, weight="bold")
    ax.set_ylabel("Accuracy Score (Substring Match)\n(Higher = Better QA Performance)", 
                 fontsize=12, weight="bold")
    ax.set_title("Accuracy vs Token Usage Trade-off\nOptimal: High accuracy with fewer tokens (upper-left region)", 
                fontsize=14, weight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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


def plot_compression_frontier(df: pd.DataFrame, output_path: str) -> None:
    """
    Pareto frontier: compression ratio (x) vs accuracy (y).
    
    Shows the trade-off between token usage and accuracy.
    Each point is one method (averaged across runs).
    """
    if "compression_ratio" not in df.columns:
        print("Skipping compression frontier plot (no compression_ratio column)")
        return
    
    methods = df["method"].unique()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    method_data = []
    for i, method in enumerate(sorted(methods)):
        sub = df[df["method"] == method]
        comp_ratio = sub["compression_ratio"].mean()
        accuracy = sub["accuracy"].mean()
        method_data.append((method, comp_ratio, accuracy))
        
        color = COLORS[i % len(COLORS)]
        ax.scatter(comp_ratio, accuracy, s=200, color=color, alpha=0.7, 
                  edgecolors="black", linewidth=2, zorder=5)
        ax.annotate(method, (comp_ratio, accuracy), 
                   textcoords="offset points", xytext=(10, 10), 
                   fontsize=9, color=color, weight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=color))
    
    # Add diagonal reference line: "ideal" compression (less tokens, same accuracy)
    baseline_acc = df[df["method"] == "full_context"]["accuracy"].mean()
    ax.axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.5, linewidth=2, label="Full context accuracy baseline")
    
    # Add shaded region for "optimal" area (low compression + high accuracy)
    ax.fill_between([0, 0.9], baseline_acc * 0.95, 1.05, alpha=0.1, color="green", label="Optimal region")
    
    # Add annotations for the axes
    ax.text(0.98, -0.08, "More Tokens Saved →", fontsize=10, ha="right", transform=ax.transAxes, style="italic", weight="bold")
    ax.text(-0.12, 0.98, "← Higher\nAccuracy", fontsize=10, ha="right", transform=ax.transAxes, style="italic", weight="bold")
    
    ax.set_xlabel("Compression Ratio: Compressed Tokens / Full Context Tokens\n(Lower = More tokens saved)", 
                 fontsize=11, weight="bold")
    ax.set_ylabel("Accuracy (Substring Match Score)\n(Higher = Better QA Performance)", fontsize=11, weight="bold")
    ax.set_title("Context Compression-Accuracy Trade-off Frontier\n" + 
                "Upper-left is best: Fewer tokens with high accuracy", 
                fontsize=14, weight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(baseline_acc * 0.8, 1.0)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_retention_vs_compression(df: pd.DataFrame, output_path: str) -> None:
    """
    Scatter: compression ratio (x) vs accuracy retention (y).
    
    Retention = method_accuracy / full_context_accuracy
    Shows what fraction of full-context accuracy is preserved under compression.
    """
    if "retention" not in df.columns or "compression_ratio" not in df.columns:
        print("Skipping retention plot (missing retention or compression_ratio columns)")
        return
    
    methods = sorted(df[df["method"] != "full_context"]["method"].unique())
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Collect data for smart label positioning
    points_data = []
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        comp_ratio = sub["compression_ratio"].mean()
        retention = sub["retention"].mean()
        color = COLORS[i % len(COLORS)]
        points_data.append((method, comp_ratio, retention, color))
        ax.scatter(comp_ratio, retention, s=280, color=color, alpha=0.75,
                  edgecolors="black", linewidth=2.5, zorder=5)
    
    # Smart label placement to avoid overlaps
    for method, comp_ratio, retention, color in points_data:
        # Determine offset based on position
        if comp_ratio < 0.88 and retention > 0.98:
            # Upper left - Best zone
            xytext = (-20, 15)
        elif comp_ratio > 0.95 and retention < 0.90:
            # Lower right - Risky zone
            xytext = (15, -20)
        elif comp_ratio > 0.95:
            # Right side
            xytext = (15, 10)
        else:
            # Left/middle
            xytext = (-20, -20)
        
        ax.annotate(method, (comp_ratio, retention),
                   textcoords="offset points", xytext=xytext,
                   fontsize=10.5, weight="bold", color=color,
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.85, edgecolor=color, linewidth=2),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color=color, lw=1.5))
    
    # Reference lines with better labels
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.6, linewidth=2.5, 
              label="100% Retention: Full-context accuracy preserved")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.6, linewidth=2.5,
              label="No Compression: Uses full token count")
    
    # Add shaded regions
    ax.fill_between([0, 0.9], 0.97, 1.08, alpha=0.12, color="green", label="Optimal zone (retain >97%)")
    ax.fill_between([0.92, 1.1], 0.75, 0.92, alpha=0.08, color="orange", label="Trade-off zone (retain <92%)")
    
    # Add zone annotations
    ax.text(0.50, 1.055, "BEST ZONE\nHigh retention + Good compression", 
           fontsize=11, ha="center", weight="bold",
           bbox=dict(boxstyle="round,pad=0.7", facecolor="#90EE90", alpha=0.85, edgecolor="darkgreen", linewidth=2.5))
    ax.text(0.97, 0.80, "RISKY ZONE\nLower retention", 
           fontsize=10, ha="center", weight="bold",
           bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFD700", alpha=0.85, edgecolor="darkorange", linewidth=2))
    
    ax.set_xlabel("Compression Ratio = (Compressed Tokens) / (Full Context Tokens)\nLower value = More tokens saved (Better)", 
                 fontsize=12, weight="bold")
    ax.set_ylabel("Accuracy Retention = (Method Accuracy) / (Full Context Accuracy)\nHigher value = More quality preserved (Better)", 
                 fontsize=12, weight="bold")
    ax.set_title("Context Compression Analysis: Quality Preservation vs Token Reduction\n" + 
                "Optimal strategy: Upper-left quadrant (fewer tokens + high quality)", 
                fontsize=14, weight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_xlim(0.74, 1.06)
    ax.set_ylim(0.78, 1.12)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.95)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_compression_analysis(df: pd.DataFrame, output_path: str) -> None:
    """
    Grouped bar chart comparing compression ratio and retention per method.
    """
    if "compression_ratio" not in df.columns or "retention" not in df.columns:
        print("Skipping compression analysis plot (missing required columns)")
        return
    
    # Exclude full_context for clarity
    methods = sorted(df[df["method"] != "full_context"]["method"].unique())
    comp_ratios = [df[df["method"] == m]["compression_ratio"].mean() for m in methods]
    retentions = [df[df["method"] == m]["retention"].mean() for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width/2, comp_ratios, width, label="Compression Ratio (Red = Fewer tokens used)", 
                  color="#FF6B6B", alpha=0.85, edgecolor="black", linewidth=1.2)
    bars2 = ax.bar(x + width/2, retentions, width, label="Accuracy Retention (Blue = More quality preserved)",
                  color="#4ECDC4", alpha=0.85, edgecolor="black", linewidth=1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=10, weight="bold")
    ax.set_ylabel("Value (0 to 1 / 0% to 100%)", fontsize=12, weight="bold")
    ax.set_title("Context Selection Method Efficiency Comparison\n" + 
                "Compression Ratio vs Accuracy Retention (Baseline = Full Context Method)",
                fontsize=14, weight="bold", pad=20)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1.25)
    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3, linewidth=1.5)
    ax.text(-0.5, 1.02, "100% = Baseline (full context)", fontsize=9, style="italic", color="gray", weight="bold")
    
    # Add value labels on bars with better formatting
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f"{height:.0%}", ha="center", va="bottom", fontsize=9, weight="bold")
    
    # Add a text box with interpretation guide
    textstr = "Interpretation:\n" + \
              "• Compression Ratio < 1.0: Uses fewer tokens than full context\n" + \
              "• Retention > 0.95: Maintains >95% of full-context accuracy\n" + \
              "• Best: Low ratio + High retention (top-left quadrant)"
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment="bottom", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray", linewidth=1.5))
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

    # Compute compression metrics
    df = compute_compression_metrics(df)

    summary = print_summary_table(df)
    summary.to_csv(results_dir / "summary_table.csv", index=False)

    # Original plots
    plot_accuracy_vs_tokens(df, str(results_dir / "accuracy_vs_tokens.png"))
    plot_efficiency(df, str(results_dir / "efficiency_comparison.png"))
    plot_accuracy_comparison(df, str(results_dir / "accuracy_comparison.png"))
    
    # New compression-focused plots
    plot_compression_frontier(df, str(results_dir / "compression_frontier.png"))
    plot_retention_vs_compression(df, str(results_dir / "retention_vs_compression.png"))
    plot_compression_analysis(df, str(results_dir / "compression_analysis.png"))

    print(f"\nAll plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
