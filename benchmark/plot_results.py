"""Publication-quality HRB figure: paired before/after, one panel per host.

Reads an ``hrb_summary_*.csv`` (baseline vs +HindsightTag with bootstrap CIs)
and renders, for each host system, its four metrics with the host measured
WITHOUT vs WITH HindsightTag side by side, so the effect of the plug-in is
directly readable. Baseline = the host system alone (paper Section 5.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 1.1,
    "savefig.dpi": 300,
})

# Metric key -> (short label, arrow, "higher is better"?)
METRICS = [
    ("rescue_recall", "Rescue\nRecall", "\u2191", True),
    ("retention_lift", "Retention\nLift", "\u2191", True),
    ("false_rescue_rate", "False Rescue\nRate", "\u2193", False),
    ("coallocation_recall_at_k", "Co-Alloc\nRecall@k", "\u2191", True),
]

# Host key -> (baseline method name, +HT method name, panel title, accent colour)
HOSTS = [
    ("mem0", "mem0+HindsightTag",
     "Host: Mem0  (a forgetting memory system \u2014 the intended target)", "#3274A1"),
    ("vector_store", "vector_store+HindsightTag",
     "Host: Vector Store  (never forgets \u2014 control, nothing to rescue)", "#3A923A"),
]

BASELINE_COLOR = "#C9CDD2"


def _lookup(df, method, metric):
    row = df[(df["method"] == method) & (df["metric"] == metric)]
    if row.empty or pd.isna(row["point_estimate"].iloc[0]):
        return None
    v = float(row["point_estimate"].iloc[0])
    lo = row["ci_lower"].iloc[0]
    hi = row["ci_upper"].iloc[0]
    lo = float(lo) if pd.notna(lo) else v
    hi = float(hi) if pd.notna(hi) else v
    return v, max(0.0, v - lo), max(0.0, hi - v)


def _draw_panel(ax, df, base_method, ht_method, title, accent):
    x = np.arange(len(METRICS))
    bw = 0.38

    for i, (mkey, _, _, _) in enumerate(METRICS):
        base = _lookup(df, base_method, mkey)
        ht = _lookup(df, ht_method, mkey)

        # Baseline bar (or N/A marker).
        if base is None:
            ax.bar(x[i] - bw / 2, 0.02, bw, color="#F0F0F0", edgecolor="#BBBBBB", linewidth=1)
            ax.text(x[i] - bw / 2, 0.05, "N/A", ha="center", va="bottom", fontsize=8, color="#8A8A8A")
        else:
            v, elo, ehi = base
            ax.bar(x[i] - bw / 2, v, bw, color=BASELINE_COLOR, edgecolor="#2B2B2B",
                   linewidth=1.1, zorder=3)
            ax.errorbar(x[i] - bw / 2, v, yerr=[[elo], [ehi]], fmt="none", ecolor="#1A1A1A",
                        capsize=3, capthick=1.1, elinewidth=1.1, zorder=4)
            _label(ax, x[i] - bw / 2, v, ehi, elo)

        # +HindsightTag bar.
        if ht is None:
            ax.bar(x[i] + bw / 2, 0.02, bw, color="#F0F0F0", edgecolor="#BBBBBB", linewidth=1)
            ax.text(x[i] + bw / 2, 0.05, "N/A", ha="center", va="bottom", fontsize=8, color="#8A8A8A")
        else:
            v, elo, ehi = ht
            ax.bar(x[i] + bw / 2, v, bw, color=accent, edgecolor="#2B2B2B", linewidth=1.1,
                   hatch="///", zorder=3)
            ax.errorbar(x[i] + bw / 2, v, yerr=[[elo], [ehi]], fmt="none", ecolor="#1A1A1A",
                        capsize=3, capthick=1.1, elinewidth=1.1, zorder=4)
            _label(ax, x[i] + bw / 2, v, ehi, elo)

        # Delta annotation when the plug-in changes a defined baseline.
        if base is not None and ht is not None:
            delta = ht[0] - base[0]
            if abs(delta) >= 0.005:
                top = max(base[0] + base[2], ht[0] + ht[2]) + 0.14
                ax.annotate(f"{'+' if delta > 0 else ''}{delta:.2f}",
                            xy=(x[i], top), ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color=("#1A7A1A" if delta > 0 else "#B03030"))

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}  {arrow}" for _, lbl, arrow, _ in METRICS], fontsize=8.5)
    ax.set_ylim(-0.25, 1.28)
    ax.axhline(0, color="#999999", linewidth=1.0, zorder=1)
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)


def _label(ax, xpos, v, ehi, elo):
    if v >= 0:
        ax.text(xpos, v + ehi + 0.03, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    else:
        ax.text(xpos, v - elo - 0.03, f"{v:.2f}", ha="center", va="top",
                fontsize=8.5, fontweight="bold", color="#B03030")


def plot(summary_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (base_m, ht_m, title, accent) in zip(axes, HOSTS):
        _draw_panel(ax, df, base_m, ht_m, title, accent)

    legend = [
        mpatches.Patch(facecolor=BASELINE_COLOR, edgecolor="#2B2B2B", label="Host system alone (baseline)"),
        mpatches.Patch(facecolor="#6E6E6E", edgecolor="#2B2B2B", hatch="///", label="Host system + HindsightTag"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True, edgecolor="#DDDDDD",
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5)

    fig.suptitle("Hindsight Rescue Benchmark \u2014 each host measured WITHOUT vs WITH HindsightTag",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.955,
             "Real numbers: 3 LoCoMo conversations \u00d7 5 seeds, \u03c9_assoc=0, bootstrap 95% CIs.  "
             "\u2191 higher is better, \u2193 lower is better.  "
             "N/A = metric undefined for a baseline with no temporal-co-allocation mechanism.",
             ha="center", va="top", fontsize=8.5, color="#555555")

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.16, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close()
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary_csv", type=Path)
    p.add_argument("--output", type=Path, default=Path("benchmark/results/hrb_results_figure.png"))
    args = p.parse_args()
    plot(args.summary_csv, args.output)


if __name__ == "__main__":
    main()
