"""Publication-quality HRB figure: delay-stratified, one row of panels per metric.

Reads an ``hrb_summary_*.csv`` that contains delay-stratified rows
(``<metric>__<delay>``) and renders, per host and per delay bucket
(1 hour / 1 day / 1 week / 1 month), the host measured WITHOUT vs WITH
HindsightTag for four metrics:

* **Rescue Recall**       (higher is better)
* **False Rescue Rate**   (lower is better)
* **Retention Lift**      (higher is better; can be negative)
* **Co-Alloc Recall@k**   (+HindsightTag only; corrected non-circular ground truth)

Usage:
  python benchmark/plot_results.py benchmark/results/hrb_summary_<ts>.csv
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

DELAY_ORDER = ["1_hour", "1_day", "1_week", "1_month"]
DELAY_LABELS = ["1 hour", "1 day", "1 week", "1 month"]

HOSTS = [
    ("mem0", "mem0+HindsightTag",
     "Mem0  (forgetting host \u2014 the intended target)", "#3274A1"),
    ("vector_store", "vector_store+HindsightTag",
     "Vector Store  (never forgets \u2014 control)", "#3A923A"),
]

# metric_key, row title, show_baseline, (ymin, ymax), draw zero line
ROWS = [
    ("rescue_recall", "Rescue Recall \u2191", True, (0.0, 1.18), False),
    ("false_rescue_rate", "False Rescue Rate \u2193", True, (0.0, 1.28), False),
    ("retention_lift", "Retention Lift \u2191", True, (-0.25, 0.35), True),
    ("coallocation_recall_at_k", "Co-Alloc Recall@k \u2191", False, (0.0, 1.18), False),
]
BASELINE_COLOR = "#C9CDD2"

# Panels needing an interpretive caption, keyed by (metric_key, baseline method).
PANEL_NOTES = {
    ("false_rescue_rate", "vector_store"): (
        "Host retention floor \u2014 the vector store never decays, so these are its own\n"
        "undecayed negatives, NOT HindsightTag false rescues (attributable = 0.00)."
    ),
}


def _pt(df, method, metric):
    row = df[(df["method"] == method) & (df["metric"] == metric)]
    if row.empty or pd.isna(row["point_estimate"].iloc[0]):
        return None
    v = float(row["point_estimate"].iloc[0])
    lo = row["ci_lower"].iloc[0]
    hi = row["ci_upper"].iloc[0]
    lo = float(lo) if pd.notna(lo) else v
    hi = float(hi) if pd.notna(hi) else v
    return v, max(0.0, v - lo), max(0.0, hi - v)


def _delay_series(df, method, base_metric):
    return [_pt(df, method, f"{base_metric}__{d}") for d in DELAY_ORDER]


def _bars(ax, base_series, ht_series, accent, show_baseline, ylim, zero_line):
    x = np.arange(len(DELAY_ORDER))
    bw = 0.38
    for i in range(len(DELAY_ORDER)):
        if show_baseline and base_series[i] is not None:
            v, elo, ehi = base_series[i]
            ax.bar(x[i] - bw / 2, v, bw, color=BASELINE_COLOR, edgecolor="#2B2B2B",
                   linewidth=1.1, zorder=3)
            ax.errorbar(x[i] - bw / 2, v, yerr=[[elo], [ehi]], fmt="none",
                        ecolor="#1A1A1A", capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
            _label(ax, x[i] - bw / 2, v, elo, ehi)
        h = ht_series[i]
        xpos = x[i] + (bw / 2 if show_baseline else 0)
        if h is not None:
            v, elo, ehi = h
            ax.bar(xpos, v, bw if show_baseline else bw * 1.4, color=accent,
                   edgecolor="#2B2B2B", linewidth=1.1, hatch="///", zorder=3)
            ax.errorbar(xpos, v, yerr=[[elo], [ehi]], fmt="none", ecolor="#1A1A1A",
                        capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
            _label(ax, xpos, v, elo, ehi)
    ax.set_xticks(x)
    ax.set_xticklabels(DELAY_LABELS, fontsize=9)
    ax.set_ylim(*ylim)
    if zero_line:
        ax.axhline(0, color="#999999", linewidth=1.0, zorder=1)
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)


def _label(ax, xpos, v, elo, ehi):
    if v >= 0:
        ax.text(xpos, v + ehi + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    else:
        ax.text(xpos, v - elo - 0.02, f"{v:.2f}", ha="center", va="top",
                fontsize=8, fontweight="bold", color="#B03030")


def plot(summary_csv: Path, output_path: Path, caption: str = "") -> None:
    df = pd.read_csv(summary_csv)
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.7 * n_rows))
    fig.patch.set_facecolor("white")

    for r, (mkey, rtitle, show_base, ylim, zero_line) in enumerate(ROWS):
        for c, (base_m, ht_m, host_title, accent) in enumerate(HOSTS):
            ax = axes[r][c]
            base = _delay_series(df, base_m, mkey) if show_base else [None] * len(DELAY_ORDER)
            _bars(ax, base, _delay_series(df, ht_m, mkey), accent, show_base, ylim, zero_line)
            ax.set_title(f"{rtitle} by delay \u2014 {host_title}", fontsize=10,
                         fontweight="bold", loc="left", pad=8)
            note = PANEL_NOTES.get((mkey, base_m))
            if note:
                ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, style="italic", color="#7A3B00",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF4E5",
                                  edgecolor="#E0B074", linewidth=0.8))

    legend = [
        mpatches.Patch(facecolor=BASELINE_COLOR, edgecolor="#2B2B2B", label="Host alone (baseline)"),
        mpatches.Patch(facecolor="#6E6E6E", edgecolor="#2B2B2B", hatch="///",
                       label="Host + HindsightTag"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=True,
               edgecolor="#DDDDDD", bbox_to_anchor=(0.5, -0.005), fontsize=9.5)

    fig.suptitle("Hindsight Rescue Benchmark \u2014 delay-stratified (corrected metrics)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    if caption:
        fig.text(0.5, 0.978, caption, ha="center", va="top", fontsize=8.5, color="#555555")

    plt.tight_layout()
    plt.subplots_adjust(top=0.945, bottom=0.05, hspace=0.38, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close()
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary_csv", type=Path)
    p.add_argument("--output", type=Path, default=Path("benchmark/results/hrb_results_figure.png"))
    p.add_argument("--caption", type=str, default=(
        "20 conversations (10 LoCoMo + 10 LongMemEval) x 10 seeds. \u03c9_assoc=0, "
        "T_tag=6h, W_capture=35d, E_window=4h. Bootstrap 95% CIs. Co-Alloc ground truth "
        "is scenario-defined, independent of E_window."))
    args = p.parse_args()
    plot(args.summary_csv, args.output, args.caption)


if __name__ == "__main__":
    main()
