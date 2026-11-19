"""Generate publication-style KV-cache compression plots into assets/."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

POLICY_ORDER = ["full", "sliding_window", "h2o", "snapkv"]
POLICY_LABELS = {
    "full": "Full KV-Cache (100%)",
    "sliding_window": "Sliding Window + Sinks",
    "h2o": "H2O (Heavy-Hitter Oracle)",
    "snapkv": "SnapKV (Observation Pooled)",
}
POLICY_COLORS = {
    "full": "#2c3e50",
    "sliding_window": "#e74c3c",
    "h2o": "#f39c12",
    "snapkv": "#27ae60",
}


def plot_niah_heatmaps(data: Dict[str, Any], out_path: str) -> None:
    """Create side-by-side 4-panel NIAH heatmap (Depth vs Context Length)."""
    meta = data["metadata"]
    contexts: List[int] = meta["exp1_contexts"]
    depths: List[float] = meta["exp1_depths"]
    runs: List[Dict[str, Any]] = data["exp1_niah"]
    budget_pct = int(round(meta.get("exp1_budget_ratio", 0.25) * 100))

    sns.set_theme(style="white", font_scale=1.05)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)

    depth_labels = [f"{int(round(d * 100))}%" for d in depths]
    ctx_labels = [f"{c // 1024}K" if c % 1024 == 0 else str(c) for c in contexts]

    for ax, policy in zip(axes, POLICY_ORDER):
        grid = np.zeros((len(depths), len(contexts)), dtype=float)
        p_runs = [r for r in runs if r["policy"] == policy]
        for r in p_runs:
            if r["depth"] in depths and r["context_len"] in contexts:
                di = depths.index(r["depth"])
                ci = contexts.index(r["context_len"])
                grid[di, ci] = 100.0 if r["correct"] else 0.0

        acc = grid.mean()
        sns.heatmap(
            grid,
            ax=ax,
            vmin=0,
            vmax=100,
            cmap="RdYlGn",
            annot=True,
            fmt=".0f",
            cbar=False,
            linewidths=0.8,
            linecolor="white",
            xticklabels=ctx_labels,
            yticklabels=depth_labels,
            annot_kws={"size": 10, "weight": "bold"},
        )
        title_suffix = "" if policy == "full" else f" ({budget_pct}% KV)"
        ax.set_title(f"{POLICY_LABELS[policy]}{title_suffix}\nAcc: {acc:.1f}%", fontsize=11, pad=10)
        ax.set_xlabel("Context Length (tokens)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Needle Depth (%)", fontsize=10)

    gpu_name = meta.get("gpu_name", "GPU")
    model_id = meta.get("model_id", "Qwen2.5-3B-Instruct")
    fig.suptitle(
        f"Needle-In-A-Haystack Retrieval Accuracy (%) — {model_id} on {gpu_name}",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def plot_budget_ablation(data: Dict[str, Any], out_path: str) -> None:
    """Create Accuracy vs KV Budget curve + KV-Cache RAM savings bar chart."""
    meta = data["metadata"]
    ratios: List[float] = meta["exp2_ratios"]
    ctx_len: int = meta["exp2_context"]
    runs: List[Dict[str, Any]] = data["exp2_ablation"]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    pct_ticks = [r * 100.0 for r in ratios]

    # Left panel: Accuracy (%) vs KV-Cache Memory Budget (%)
    for policy in ["sliding_window", "h2o", "snapkv"]:
        accs = []
        for ratio in ratios:
            subset = [
                r
                for r in runs
                if r["policy"] == policy and abs(r["budget_ratio"] - ratio) < 1e-6
            ]
            acc = 100.0 * sum(r["correct"] for r in subset) / max(1, len(subset))
            accs.append(acc)
        ax1.plot(
            pct_ticks,
            accs,
            marker="o",
            linewidth=2.4,
            markersize=8,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )

    # Full KV-Cache 100% reference line
    full_subset = [r for r in runs if abs(r["budget_ratio"] - 1.0) < 1e-6]
    if full_subset:
        full_acc = 100.0 * sum(r["correct"] for r in full_subset) / len(full_subset)
        ax1.axhline(
            full_acc,
            linestyle="--",
            linewidth=1.6,
            color=POLICY_COLORS["full"],
            label=f"Full KV-Cache ({full_acc:.1f}%)",
        )

    ax1.set_title(f"NIAH Accuracy vs KV-Cache Budget (L={ctx_len})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("KV-Cache Budget (% of Prompt Length)", fontsize=10.5)
    ax1.set_ylabel("Exact-Match Accuracy (%)", fontsize=10.5)
    ax1.set_xticks(pct_ticks)
    ax1.set_xticklabels([f"{p:g}%" for p in pct_ticks])
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="lower right", frameon=True)

    # Right panel: KV-Cache RAM Footprint (MB) bar chart across budgets
    avg_mbs = []
    for ratio in ratios:
        subset = [r for r in runs if abs(r["budget_ratio"] - ratio) < 1e-6]
        mb = sum(r["kv_cache_mb"] for r in subset) / max(1, len(subset))
        avg_mbs.append(mb)

    full_mb = avg_mbs[-1] if avg_mbs else 1.0
    bar_colors = ["#27ae60", "#2ecc71", "#3498db", "#2c3e50"][: len(ratios)]
    bars = ax2.bar(
        [f"{p:g}%" for p in pct_ticks],
        avg_mbs,
        color=bar_colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, mb, ratio in zip(bars, avg_mbs, ratios):
        saving = (1.0 - mb / full_mb) * 100.0 if full_mb > 0 else 0.0
        label = f"{mb:.1f} MB\n(-{saving:.0f}%)" if ratio < 1.0 else f"{mb:.1f} MB\n(baseline)"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(avg_mbs) * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax2.set_title(f"KV-Cache VRAM Footprint (L={ctx_len}, FP16)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("KV-Cache Budget (% of Prompt Length)", fontsize=10.5)
    ax2.set_ylabel("KV-Cache Memory (MiB)", fontsize=10.5)
    ax2.set_ylim(0, max(avg_mbs) * 1.24 if avg_mbs else 10)

    gpu_name = meta.get("gpu_name", "GPU")
    model_id = meta.get("model_id", "Qwen2.5-3B-Instruct")
    fig.suptitle(
        f"KV-Cache Compression Trade-Off — {model_id} on {gpu_name}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KV-cache benchmark results.")
    parser.add_argument("--input", type=str, default="benchmarks/results.json")
    parser.add_argument("--assets-dir", type=str, default="assets")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_niah_heatmaps(data, os.path.join(args.assets_dir, "niah_heatmaps.png"))
    plot_budget_ablation(data, os.path.join(args.assets_dir, "budget_ablation.png"))


if __name__ == "__main__":
    main()
