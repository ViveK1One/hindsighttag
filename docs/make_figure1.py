"""Render paper Figure 1 (HindsightTag lifecycle plug-in) in the paper's B&W style.

Faithful raster of the TikZ diagram in ``r2.tex`` for embedding in the README:
plain rounded rectangles, black arrows, dashed plug-in container, straight
vertical arrows centred on a single axis (mirrors the corrected r2.tex).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figure1_architecture.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 300,
})

CX = 5.0          # single vertical axis every box and arrow is centred on
WIDE = 6.6        # width of the main-column boxes


def _box(ax, y, h, lines, wide=WIDE, weights=None, sizes=None):
    x = CX - wide / 2
    ax.add_patch(FancyBboxPatch((x, y), wide, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.2, edgecolor="black", facecolor="white", zorder=2))
    n = len(lines)
    for i, line in enumerate(lines):
        yy = y + h * (n - i) / (n + 1)
        ax.text(CX, yy, line, ha="center", va="center", color="black", zorder=3,
                fontsize=(sizes[i] if sizes else 10.5),
                fontweight=(weights[i] if weights else "normal"))


def _subbox(ax, xc, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((xc - w / 2, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.1, edgecolor="black", facecolor="white", zorder=3))
    n = len(lines)
    for i, line in enumerate(lines):
        yy = y + h * (n - i) / (n + 1)
        ax.text(xc, yy, line, ha="center", va="center", color="black", zorder=4, fontsize=7.5)


def _arrow(ax, y0, y1):
    ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color="black", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(8.6, 8.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _box(ax, 12.2, 0.8, ["User / Environment"])
    _box(ax, 10.9, 0.9, ["Memory Encoding",
                         r"(salience-gated ingestion, $\mathrm{Sal}(\cdot)$)"], sizes=[10.5, 8.5])
    _box(ax, 9.5, 1.0, ["Host Memory System",
                        "(vector store / knowledge graph / existing agent",
                        "memory \u2014 e.g. ZenBrain, FadeMem, FSFM, Mem0)"],
         sizes=[10.5, 8, 8])

    # Dashed plug-in container + 5 sub-boxes.
    ax.add_patch(FancyBboxPatch((0.55, 6.55), 8.9, 2.2,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.2, edgecolor="black", facecolor="white",
                                linestyle=(0, (5, 3)), zorder=1))
    ax.text(0.75, 8.55, "HindsightTag Lifecycle Plug-in (Sec. 4)",
            fontsize=9.5, color="black", zorder=4, ha="left", va="center")

    sub = [
        (["Synaptic Tag", "Eq. 2"]),
        (["Tag Decay", "Eq. 2"]),
        (["Capture", "Detection", "Eq. 3\u20134"]),
        (["Backward", "Rescue", "Eq. 5"]),
        (["Temporal", "Co-allocation", "Eq. 6"]),
    ]
    sw, gap = 1.5, 0.24
    total = 5 * sw + 4 * gap
    start = CX - total / 2 + sw / 2
    for i, lines in enumerate(sub):
        _subbox(ax, start + i * (sw + gap), 6.95, sw, 1.4, lines)

    _box(ax, 5.35, 0.95, ["Updated Memory Store",
                          "(rescued + co-allocated items, provenance log)"], sizes=[10.5, 8.5])
    _box(ax, 4.05, 0.9, ["Retrieval",
                         "(salience-, trust-, and co-allocation-ranked)"], sizes=[10.5, 8.5])
    _box(ax, 2.85, 0.8, ["LLM Agent"], weights=["bold"])

    _arrow(ax, 12.2, 11.82)      # user -> encode
    _arrow(ax, 10.9, 10.52)      # encode -> host
    _arrow(ax, 9.5, 8.78)        # host -> plug-in (straight, centred)
    _arrow(ax, 6.55, 6.32)       # plug-in -> store
    _arrow(ax, 5.35, 4.97)       # store -> retrieval
    _arrow(ax, 4.05, 3.67)       # retrieval -> agent

    fig.suptitle("Figure 1: HindsightTag as a lifecycle plug-in attached to a host memory system",
                 fontsize=10.5, y=0.965, color="black")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    plt.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
