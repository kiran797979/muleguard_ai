"""
Method schematics for the conference paper.

`paper_figures.py` draws the figures that carry *results*. This module draws the
four that carry *method*, which the paper previously described in prose only:

    the behavioural signature the detector is built to measure,
    the four independent leakage gates and what each one removes,
    which components are fitted inside which split,
    and how a column is resolved by meaning rather than by name.

The nested-CV schematic in particular is the paper's central methodological
claim. Stating "the threshold is fitted inside the training fold" in a sentence
is easy to skim past; a reader who sees where the arrow starts cannot skim it.

Everything here is a schematic. Nothing reads a report, because none of these
figures assert a measured quantity -- they assert a design. Figures that quote
numbers live in `paper_figures.py` and are regenerated from the JSON reports.

Run:  python src/paper_fig_method.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

import config as C
from utils import log

FIG_DIR = C.REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TEX_FIG_DIR = C.ROOT / "paper" / "figures"

# Same geometry and palette as paper_figures.py, so the two sets sit together
# on a page without looking like they came from different papers.
COL_W, FULL_W = 3.39, 7.0
INK, ACCENT, WARN, MUTED = "#1a1a1a", "#128a7d", "#c0392b", "#8a8a8a"
FILL, ACCENT_FILL, WARN_FILL = "#f4f4f4", "#e2f1ef", "#fbeceb"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})


def _canvas(w: float, h: float):
    """A figure with one axis spanning it, in 0-100 x 0-100 coordinates."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, label=None, sub=None, *, ec=INK, fc=FILL, lw=1.1,
         fs=7.5, subfs=6.3, weight="bold", dashed=False, tc=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                           linestyle=(0, (3, 2)) if dashed else "solid",
                           zorder=2))
    tc = tc or ec
    if label and sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=subfs, color=MUTED, zorder=3)
    elif label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=tc, zorder=3)


def _arrow(ax, x1, y1, x2, y2, *, color=INK, lw=1.1, ls="-", mut=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=mut, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


def _save(fig, name: str) -> None:
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    if TEX_FIG_DIR.exists():
        fig.savefig(TEX_FIG_DIR / f"{name}.pdf", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    log(f"wrote {png}")


# ---------------------------------------------------------------------------
# 1. The behavioural signature
# ---------------------------------------------------------------------------
def fig_behaviour() -> None:
    """Balance over time for an accumulating account and a pass-through account.

    This is the whole detection premise in one picture. Both accounts hold a
    genuine identity; only the shape of the balance separates them, and that
    shape is arithmetic rather than paperwork, which is why it transfers to
    another institution's data.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 1.95), sharey=True)

    t1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    b1 = np.array([0, 34, 41, 62, 68, 84, 89, 91])
    a1.plot(t1, b1, color=INK, lw=1.5, zorder=3)
    a1.fill_between(t1, b1, color=INK, alpha=0.07, zorder=1)
    a1.scatter([1, 3, 5], [34, 62, 84], s=22, color=ACCENT, zorder=4,
               label="credit")
    a1.scatter([2, 4, 6], [41, 68, 89], s=22, color=WARN, zorder=4,
               marker="s", label="debit")
    a1.set_title("(a) accumulating account", fontsize=8, color=INK, pad=4)
    a1.set_xlabel("time", fontsize=7)
    a1.set_ylabel("balance", fontsize=7)

    # A pass-through: money arrives and leaves at nearly the same value, so the
    # balance returns to its floor between events instead of building.
    t2, b2 = [0.0], [3.0]
    for c in (1.2, 3.2, 5.2):
        t2 += [c, c + 0.02, c + 0.55, c + 0.57]
        b2 += [3, 88, 86, 3]
    t2.append(7.0)
    b2.append(3.0)
    a2.plot(t2, b2, color=INK, lw=1.5, zorder=3)
    a2.fill_between(t2, b2, color=WARN, alpha=0.10, zorder=1)
    a2.scatter([1.22, 3.22, 5.22], [88, 88, 88], s=22, color=ACCENT, zorder=4)
    a2.scatter([1.77, 3.77, 5.77], [86, 86, 86], s=22, color=WARN, marker="s",
               zorder=4)
    a2.annotate("", xy=(1.77, 96), xytext=(1.22, 96),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=MUTED))
    a2.text(1.5, 99, "in $\\approx$ out", ha="center", va="bottom",
            fontsize=6.4, color=MUTED)
    a2.set_title("(b) pass-through account", fontsize=8, color=INK, pad=4)
    a2.set_xlabel("time", fontsize=7)

    for ax in (a1, a2):
        ax.set_ylim(-4, 112)
        ax.set_xlim(-0.3, 7.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
            ax.spines[s].set_linewidth(0.8)

    h = [plt.Line2D([], [], marker="o", ls="", color=ACCENT, ms=4, label="credit"),
         plt.Line2D([], [], marker="s", ls="", color=WARN, ms=4, label="debit")]
    a1.legend(handles=h, loc="upper left", frameon=False, fontsize=6.4,
              handletextpad=0.3, borderpad=0.1)
    fig.subplots_adjust(wspace=0.08)
    _save(fig, "fig_method_behaviour")


# ---------------------------------------------------------------------------
# 2. The four leakage gates
# ---------------------------------------------------------------------------
def fig_leak_gates() -> None:
    """A funnel narrowing through four independent gates.

    Each gate catches a class of leakage the others would miss, which is the
    argument for having four rather than one tuned correlation filter.
    """
    fig, ax = _canvas(FULL_W, 2.35)

    _box(ax, 1, 52, 13, 26, "every", "column", fs=7.5)
    ax.text(7.5, 46, "as delivered", ha="center", fontsize=6.1, color=MUTED)

    left, right = 16, 84
    top_l, top_r, bot_l, bot_r = 82, 71, 48, 59
    ax.add_patch(Polygon([(left, top_l), (right, top_r), (right, bot_r),
                          (left, bot_l)], closed=True, facecolor=FILL,
                         edgecolor=INK, lw=1.0, zorder=2))

    gates = [
        (27, "identifiers", "name and\nnear-uniqueness"),
        (44, "post-outcome", "by meaning, not\nby correlation"),
        (61, "partition", "class-purity\nof the values"),
        (78, "correlation", "leaks and\ncollinear pairs"),
    ]
    for gx, name, sub in gates:
        f = (gx - left) / (right - left)
        ty = top_l + (top_r - top_l) * f
        by = bot_l + (bot_r - bot_l) * f
        ax.plot([gx, gx], [by, ty], color=INK, lw=1.6, zorder=3)
        _arrow(ax, gx, by - 1, gx, 33, color=WARN, lw=0.9, ls=(0, (2.5, 1.8)))
        ax.text(gx, 29.5, name, ha="center", va="top", fontsize=7,
                fontweight="bold", color=WARN)
        ax.text(gx, 24.5, sub, ha="center", va="top", fontsize=6.1,
                color=MUTED, linespacing=1.35)

    ax.text(52.5, 9, "removed before any model is fitted", ha="center",
            fontsize=6.8, color=WARN, style="italic")

    _box(ax, 86, 55, 13, 20, "model", "features", ec=ACCENT, fc=ACCENT_FILL,
         fs=7.5, tc=ACCENT)
    _save(fig, "fig_method_leakgates")


# ---------------------------------------------------------------------------
# 3. What is fitted where
# ---------------------------------------------------------------------------
def fig_nested_cv() -> None:
    """Which components are fitted on which split, and where they are applied.

    The decision threshold is drawn in the same emphasis as the model itself
    because selecting it on validation predictions is the most common route by
    which a reported precision stops being an out-of-sample quantity.
    """
    fig, ax = _canvas(FULL_W, 2.75)

    ax.text(4, 95, "one outer fold, repeated over folds and repeats",
            fontsize=6.6, color=MUTED, style="italic")

    seg_w = 15.2
    for k in range(4):
        _box(ax, 4 + k * seg_w, 78, seg_w - 0.6, 11, "train", fs=6.8,
             weight="normal", ec=MUTED, tc=MUTED)
    _box(ax, 4 + 4 * seg_w, 78, seg_w - 0.6, 11, "validation", fs=6.8,
         ec=ACCENT, fc=ACCENT_FILL, tc=ACCENT)

    brace_r = 4 + 4 * seg_w - 0.6
    ax.plot([4, brace_r], [74, 74], color=MUTED, lw=0.8)
    ax.plot([4, 4], [74, 77], color=MUTED, lw=0.8)
    ax.plot([brace_r, brace_r], [74, 77], color=MUTED, lw=0.8)
    _arrow(ax, (4 + brace_r) / 2, 74, (4 + brace_r) / 2, 66, color=MUTED, lw=0.9)

    ax.add_patch(Rectangle((4, 40), brace_r - 4, 25, facecolor="white",
                           edgecolor=INK, lw=1.0, zorder=2))
    ax.text(6.5, 61, "inner 3-fold CV", fontsize=6.8, fontweight="bold",
            color=INK, zorder=3)
    inner_w = (brace_r - 4 - 5) / 3
    for row in range(3):
        y = 55 - row * 5.2
        for col in range(3):
            x = 6.5 + col * inner_w
            held = col == row
            ax.add_patch(Rectangle((x, y - 3.4), inner_w - 1.2, 3.4,
                                   facecolor=ACCENT_FILL if held else "white",
                                   edgecolor=ACCENT if held else MUTED,
                                   lw=1.0 if held else 0.6, zorder=3))
    ax.text(brace_r - 2.5, 61, "shaded = held out within the fold",
            fontsize=6.0, color=MUTED, ha="right", zorder=3)

    _arrow(ax, (4 + brace_r) / 2, 40, (4 + brace_r) / 2, 33, color=MUTED, lw=0.9)

    ax.add_patch(Rectangle((4, 6), brace_r - 4, 26, facecolor="white",
                           edgecolor=ACCENT, lw=1.2, zorder=2))
    ax.text(6.5, 28, "fitted here, then frozen", fontsize=6.8,
            fontweight="bold", color=ACCENT, zorder=3)
    items = ["feature selection", "base models", "stacking weights",
             "imputation", "isotonic calibration", "decision threshold"]
    cw = (brace_r - 4 - 5) / 3
    for k, name in enumerate(items):
        r, c = divmod(k, 3)
        x = 6.5 + c * cw
        y = 21 - r * 8
        emph = name == "decision threshold"
        ax.add_patch(Rectangle((x, y - 5.6), cw - 1.6, 5.6,
                               facecolor=ACCENT_FILL if emph else FILL,
                               edgecolor=ACCENT if emph else MUTED,
                               lw=1.1 if emph else 0.6, zorder=3))
        ax.text(x + (cw - 1.6) / 2, y - 2.8, name, ha="center", va="center",
                fontsize=6.1, color=ACCENT if emph else INK,
                fontweight="bold" if emph else "normal", zorder=4)

    vx = 4 + 4 * seg_w + (seg_w - 0.6) / 2
    ax.add_patch(FancyArrowPatch(
        (brace_r, 21), (vx, 77),
        connectionstyle="angle,angleA=0,angleB=90,rad=0",
        arrowstyle="-|>", mutation_scale=8, color=ACCENT, lw=1.2, zorder=4))
    ax.text(vx + 2.2, 49, "applied frozen", rotation=90, ha="left",
            va="center", fontsize=6.4, color=ACCENT, fontweight="bold")
    _save(fig, "fig_method_nestedcv")


# ---------------------------------------------------------------------------
# 4. Resolution by role
# ---------------------------------------------------------------------------
def fig_roles() -> None:
    """Three schema-specific names decomposing to one shared role tuple."""
    fig, ax = _canvas(FULL_W, 1.75)

    names = ["TOT_TXNAMT_CR_L7D", "InwardAmount7Day", "sum.amt.in.7d"]
    for k, nm in enumerate(names):
        y = 68 - k * 26
        _box(ax, 2, y, 27, 18, nm, fs=6.6, weight="normal", ec=MUTED)
        _arrow(ax, 29.5, y + 9, 35, 58 - k * 7, color=MUTED, lw=0.8)

    _box(ax, 35, 40, 19, 22, "parse_role()", "decompose", ec=ACCENT,
         fc=ACCENT_FILL, fs=7.2, subfs=6.0, tc=ACCENT)
    _arrow(ax, 54.5, 51, 59.5, 51, color=ACCENT, lw=1.0)

    slots = [("TOTAL", "stat"), ("AMOUNT", "measure"),
             ("CREDIT", "direction"), ("7D", "window")]
    sw = 9.0
    for k, (val, role) in enumerate(slots):
        x = 58 + k * (sw + 0.9)
        _box(ax, x, 42, sw, 18, val, ec=ACCENT, fc="white", fs=6.5, tc=ACCENT)
        ax.text(x + sw / 2, 38, role, ha="center", va="top", fontsize=5.9,
                color=MUTED)

    ax.text(77.4, 26, "one role, one feature, no name has to match",
            ha="center", fontsize=6.3, color=MUTED, style="italic")
    _save(fig, "fig_method_roles")


def main() -> None:
    fig_behaviour()
    fig_leak_gates()
    fig_nested_cv()
    fig_roles()
    log("method schematics complete")


if __name__ == "__main__":
    main()
