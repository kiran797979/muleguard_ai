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


# --- fit a figure so its TIGHT bbox lands on the target width ---------------
# bbox_inches="tight" trims to content, so a figure authored at 7.0 in can save
# at 5.6 and then be stretched on insert, which changes its type size relative
# to every other figure. Scaling the canvas uniformly until the trimmed output
# measures the target keeps every label at its authored point size and means
# Word never rescales anything.
def _fit_width(fig, target, max_h=None, pad=0.1, tries=6):
    for _ in range(tries):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        # savefig(bbox_inches="tight") adds pad_inches on every side, so the
        # file is 2*pad wider than the bbox we are measuring here.
        k = (target - 2 * pad) / bb.width
        if max_h is not None and bb.height * k > max_h - 2 * pad:
            k = (max_h - 2 * pad) / bb.height
        if abs(k - 1.0) < 0.004:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * k, h * k)


def _save(fig, name: str) -> None:
    _fit_width(fig, 7.15, 4.3)          # every schematic spans the text width
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
    b1 = np.array([0, 42, 34, 74, 66, 98, 90, 90])
    a1.plot(t1, b1, color=INK, lw=1.5, zorder=3)
    a1.fill_between(t1, b1, color=INK, alpha=0.07, zorder=1)
    a1.scatter([1, 3, 5], [42, 74, 98], s=22, color=ACCENT, zorder=4,
               label="credit")
    a1.scatter([2, 4, 6], [34, 66, 90], s=22, color=WARN, zorder=4,
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

    # Five segments spanning x=4..96. At 15.2 the block stopped at x=78 and
    # the right quarter of the canvas stayed empty. The axis fills the
    # figure, so a tight bbox cannot crop that space away.
    seg_w = 18.4
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
             "imputation", "probability calibration", "decision threshold"]
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
    # The composition occupies y=16..86, so a full 0-100 canvas padded it with a
    # dead band of roughly 15 percent at top and bottom. The axis fills the
    # figure, so a tight bbox cannot crop that away. Cropping the view to the
    # content and shrinking the figure by the same fraction keeps the units-per-
    # inch scale identical: every box and label stays exactly the size it was on
    # the page, and only the whitespace goes.
    VIEW = (11, 91)
    fig, ax = _canvas(FULL_W, 1.75 * (VIEW[1] - VIEW[0]) / 100)
    ax.set_ylim(*VIEW)

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




# ---------------------------------------------------------------------------
# 5. The unified architecture
# ---------------------------------------------------------------------------
def fig_unified() -> None:
    """Four detectors, one fitted combiner, and the submission they produce.

    Section V-H describes this system in prose and Section V-I measures it, but
    the paper had no picture of it: the architecture figure predates the graph,
    motif and temporal work and stops at account scoring. The one thing a
    reader cannot get from the prose is that the temporal branch never reaches
    the score. It runs beside the ranking and meets it only at the output,
    which is why it is drawn as a separate line rather than a fifth input.
    """
    fig, ax = _canvas(FULL_W, 4.35)

    # ---- input ----------------------------------------------------------
    _box(ax, 32, 87, 34, 10, "TRANSACTION LEDGER",
         sub="src  .  dst  .  amount  .  timestamp", ec=ACCENT,
         fc=ACCENT_FILL, fs=8)

    # ---- fan out --------------------------------------------------------
    xs = [13, 37, 61, 85]
    ax.plot([49, 49], [87, 82.5], color=INK, lw=1.1, zorder=1)
    ax.plot([13, 85], [82.5, 82.5], color=INK, lw=1.1, zorder=1)
    for cx in xs:
        _arrow(ax, cx, 82.5, cx, 77.4)

    # ---- the four detectors ---------------------------------------------
    detectors = [
        ("BEHAVIOURAL", "pass-through, retention,", "burstiness, thresholds",
         "needs no graph at all", INK, FILL),
        ("MOTIF", "fan-in, fan-out,", "gather-scatter, chains",
         "needs no global structure", INK, FILL),
        ("STRUCTURAL", "dense inside,", "sparse outward",
         "needs a separable cell", INK, FILL),
        ("TEMPORAL", "day-level suspicion,", "Otsu cut, max subarray",
         "answers when, not whether", ACCENT, ACCENT_FILL),
    ]
    for cx, (name, s1, s2, note, ec, fc) in zip(xs, detectors):
        ax.add_patch(Rectangle((cx - 11, 58), 22, 19, facecolor=fc,
                               edgecolor=ec, lw=1.1, zorder=2))
        ax.text(cx, 73.2, name, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=ec, zorder=3)
        ax.text(cx, 69.0, s1, ha="center", va="center", fontsize=6.1,
                color=INK, zorder=3)
        ax.text(cx, 65.6, s2, ha="center", va="center", fontsize=6.1,
                color=INK, zorder=3)
        ax.text(cx, 61.2, note, ha="center", va="center", fontsize=5.9,
                color=MUTED, style="italic", zorder=3)

    # ---- three of them converge -----------------------------------------
    for cx in xs[:3]:
        ax.plot([cx, cx], [58, 52], color=INK, lw=1.1, zorder=1)
    ax.plot([13, 61], [52, 52], color=INK, lw=1.1, zorder=1)
    _arrow(ax, 37, 52, 37, 46.4)

    _box(ax, 14, 38, 46, 8, "FEATURE MATRIX",
         sub="28 behavioural + 10 network columns", fs=7.5)
    _arrow(ax, 37, 38, 37, 34.4)

    ax.add_patch(Rectangle((14, 22), 46, 12, facecolor=ACCENT_FILL,
                           edgecolor=ACCENT, lw=1.1, zorder=2))
    ax.text(37, 30.7, "FITTED ENSEMBLE", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=ACCENT, zorder=3)
    ax.text(37, 27.2, "learns what each signal is worth,", ha="center",
            va="center", fontsize=6.1, color=INK, zorder=3)
    ax.text(37, 24.2, "including that one is worth nothing", ha="center",
            va="center", fontsize=5.9, color=MUTED, style="italic", zorder=3)
    _arrow(ax, 37, 22, 37, 15.4)

    # ---- the temporal branch bypasses the score -------------------------
    ax.plot([85, 85], [58, 10.0], color=ACCENT, lw=1.1, ls=(0, (3, 2)),
            zorder=1)
    _arrow(ax, 85, 10.0, 66.6, 10.0, color=ACCENT, ls=(0, (3, 2)))
    ax.text(87, 36.0, "runs alongside;", ha="left", va="center",
            fontsize=6.1, color=ACCENT, style="italic")
    ax.text(87, 32.6, "never feeds", ha="left", va="center",
            fontsize=6.1, color=ACCENT, style="italic")
    ax.text(87, 29.2, "the score", ha="left", va="center",
            fontsize=6.1, color=ACCENT, style="italic")

    # ---- output ---------------------------------------------------------
    _box(ax, 8, 5, 58, 10, "SUBMISSION",
         sub="account_id  .  is_mule  .  suspicious_start  .  suspicious_end",
         ec=ACCENT, fc=ACCENT_FILL, fs=8, subfs=6.1)

    _save(fig, "fig_method_unified")




# ---------------------------------------------------------------------------
# 6. What happens when the file's contents are unknown
# ---------------------------------------------------------------------------
def fig_routing() -> None:
    """The three routes an incoming file can take.

    Section V-I argues that training needs labels and detection does not, and
    that conflating the two is what made an unlabelled file an error rather
    than a queue. The claim is easy to state and easy to skim; the branch
    structure is what makes it concrete, and only one of the three routes can
    report precision at all.
    """
    fig, ax = _canvas(FULL_W, 3.5)

    def diamond(cx, cy, hw, hh, label, sub=None):
        ax.add_patch(Polygon([(cx - hw, cy), (cx, cy + hh), (cx + hw, cy),
                              (cx, cy - hh)], closed=True, facecolor=ACCENT_FILL,
                             edgecolor=ACCENT, lw=1.1, zorder=2))
        ax.text(cx, cy + (1.2 if sub else 0), label, ha="center", va="center",
                fontsize=6.6, fontweight="bold", color=ACCENT, zorder=3)
        if sub:
            ax.text(cx, cy - 2.6, sub, ha="center", va="center", fontsize=6.6,
                    fontweight="bold", color=ACCENT, zorder=3)

    _box(ax, 36, 88, 28, 9, "UPLOADED FILE", ec=ACCENT, fc=ACCENT_FILL, fs=7.5)
    _arrow(ax, 50, 88, 50, 81.5)
    diamond(50, 75, 17, 5.5, "HAS A LABEL COLUMN?")

    # yes, to the left
    ax.plot([33, 12], [75, 75], color=INK, lw=1.1, zorder=1)
    _arrow(ax, 12, 75, 12, 60.5)
    ax.text(22, 76.6, "YES", fontsize=6.2, fontweight="bold", color=INK)
    # no, to the right
    ax.plot([67, 78], [75, 75], color=INK, lw=1.1, zorder=1)
    _arrow(ax, 78, 75, 78, 66.5)
    ax.text(70, 76.6, "NO", fontsize=6.2, fontweight="bold", color=MUTED)

    ax.add_patch(Rectangle((1, 44), 22, 16, facecolor=ACCENT_FILL,
                           edgecolor=ACCENT, lw=1.2, zorder=2))
    ax.text(12, 55.5, "TRAIN AND", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=ACCENT, zorder=3)
    ax.text(12, 52.2, "MEASURE", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=ACCENT, zorder=3)
    ax.text(12, 48.4, "the only route where", ha="center", va="center",
            fontsize=5.8, color=MUTED, style="italic", zorder=3)
    ax.text(12, 46.0, "precision exists", ha="center", va="center",
            fontsize=5.8, color=MUTED, style="italic", zorder=3)

    diamond(78, 61, 19, 5.5, "SCHEMA MATCHES", "THE TRAINED MODEL?")

    ax.plot([59, 44], [61, 61], color=INK, lw=1.1, zorder=1)
    _arrow(ax, 44, 61, 44, 38.5)
    ax.text(50, 62.6, "YES", fontsize=6.2, fontweight="bold", color=INK)
    ax.plot([97, 99], [61, 61], color=INK, lw=1.1, zorder=1)
    _arrow(ax, 99, 61, 99, 38.5)
    ax.text(92, 62.6, "NO", fontsize=6.2, fontweight="bold", color=MUTED)

    # Drawn longhand: _box centres its own sub line, which lands on top of the
    # italic notes below it.
    def route(x, w, cx, l1, l2, n1, n2):
        ax.add_patch(Rectangle((x, 22), w, 16, facecolor=FILL, edgecolor=INK,
                               lw=1.1, zorder=2))
        ax.text(cx, 34.6, l1, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=INK, zorder=3)
        ax.text(cx, 31.4, l2, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=INK, zorder=3)
        ax.text(cx, 27.3, n1, ha="center", va="center", fontsize=5.8,
                color=MUTED, style="italic", zorder=3)
        ax.text(cx, 24.6, n2, ha="center", va="center", fontsize=5.8,
                color=MUTED, style="italic", zorder=3)

    route(31, 26, 44, "SCORE WITH", "THE DEPLOYED MODEL",
          "calibrated bands,", "no target is read")
    route(66, 33, 82.5, "TYPOLOGY REBUILT", "BY ROLE",
          "directions fixed in advance,", "operating point by Otsu")

    for x, top in ((12, 44), (44, 22), (82.5, 22)):
        ax.plot([x, x], [top, 13], color=INK, lw=1.1, zorder=1)
    ax.plot([12, 82.5], [13, 13], color=INK, lw=1.1, zorder=1)
    _arrow(ax, 47, 13, 47, 9.5)
    _box(ax, 28, 1, 38, 8.5, "A RANKED QUEUE, EITHER WAY", ec=ACCENT,
         fc=ACCENT_FILL, fs=7.5)

    _save(fig, "fig_method_routing")


def main() -> None:
    fig_behaviour()
    fig_leak_gates()
    fig_nested_cv()
    fig_roles()
    fig_unified()
    fig_routing()
    log("method schematics complete")


if __name__ == "__main__":
    main()
