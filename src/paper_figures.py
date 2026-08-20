"""
Publication figures for the conference paper.

Generates the figures that the pipeline's own plots.py does not: the pipeline
architecture diagram, the integrity-audit comparison, the SHAP ranking with real
banking variable names, the band/triage breakdown, and a per-model ablation.

All figures are written at 300 dpi to reports/figures/ and are sized for a
single IEEE column (3.4 in) or the full text width (7.0 in).

Run:  python src/paper_figures.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as C
from utils import log

FIG_DIR = C.REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# The LaTeX build (spconf.sty) also consumes these, as vector PDFs.
TEX_FIG_DIR = C.ROOT / "paper" / "figures"

# spconf geometry: print area 7.0 in wide, two columns of 3.39 in with a
# 0.24 in gutter. Figures are drawn at exactly these widths so LaTeX never has
# to rescale them and the fonts stay at their intended point size.
COL_W, FULL_W = 3.39, 7.0
INK, ACCENT, WARN, MUTED = "#1a1a1a", "#128a7d", "#c0392b", "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def _load(name: str) -> dict:
    path = C.REPORTS_DIR / name
    if not path.exists():
        log(f"missing {path} — run the pipeline first")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, name: str) -> None:
    """Write the PNG (for the Word build) and a vector PDF (for the LaTeX build).

    PDF is what spconf/pdflatex wants: it scales without resampling and keeps
    the text selectable, which matters for a print-quality submission.
    """
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=300)
    TEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = TEX_FIG_DIR / (pathlib.Path(name).stem + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out.name} + {pdf.name}")


# --------------------------------------------------------------------------
def fig_architecture() -> None:
    """Full-width pipeline diagram.

    Design rules this layout follows, after the first version violated all four:
      * ONE unambiguous successor per stage. The main pipeline is a single
        left-to-right spine, so there is never a choice of which arrow to follow.
      * Every box is connected. Nothing floats decoratively.
      * Stage numbers are sequential and match the paper's sections. The old
        "Stage 2/3" and "Stage 4/5" labels were leftovers from a different
        numbering scheme and told the reader nothing.
      * Cross-cutting resources are drawn as buses, not as arrows crossing the
        spine: the integrity verdict gates every stage from above, and the data
        dictionary feeds three of them from below.
    """
    fig, ax = plt.subplots(figsize=(FULL_W, 3.95))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    GATE_Y, BUS_Y = 39.0, 17.0
    SPINE_Y, SPINE_H = 22.0, 13.0
    # Spine box left edges and width; centres drive every connector below.
    SP = [(1, 23), (27, 23), (53, 23), (79, 20)]
    centres = [x + w / 2 for x, w in SP]

    def box(x, y, w, h, title, sub="", color=ACCENT, lw=1.1, fs=7.5,
            dashed=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45", linewidth=lw,
            edgecolor=color, facecolor="white", zorder=3,
            linestyle="--" if dashed else "-"))
        ax.text(x + w / 2, y + h - 3.1, title, ha="center", va="center",
                fontsize=fs, color=INK, weight="bold", zorder=4)
        if sub:
            # 0.35 of the height leaves a three-line subtitle clear of both the
            # title above it and the box floor below.
            ax.text(x + w / 2, y + 0.35 * h, sub, ha="center", va="center",
                    fontsize=5.9, color=MUTED, zorder=4, linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=INK, lw=1.0, dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
            linewidth=lw, color=color, zorder=5,
            linestyle="--" if dashed else "-",
            shrinkA=0, shrinkB=0))

    # ---- Top row: the integrity gate, which runs before anything else -------
    box(1, 44, 25, 11, "Raw dataset",
        "9,082 accounts\n3,925 columns, 81 mules", MUTED)
    box(32, 44, 32, 11, "Stage 0  Integrity audit",
        "3 falsification tests\nmonth-split cross-tabulation", WARN)
    box(70, 44, 29, 11, "Contamination verdict",
        "00_INTEGRITY.md", WARN, lw=1.7)
    arrow(26, 49.5, 32, 49.5)
    arrow(64, 49.5, 70, 49.5, WARN)

    # ---- Gate bus: the verdict qualifies every stage underneath -------------
    # The bus starts right of x=16 so the raw-data arrow can drop into Stage 1
    # without crossing it; its Stage 1 tick sits inside that box regardless.
    gate_taps = [16.0] + centres[1:]
    arrow(48, 44, 48, GATE_Y + 0.4, WARN, dashed=True)
    ax.plot([gate_taps[0], gate_taps[-1]], [GATE_Y, GATE_Y], color=WARN,
            linewidth=0.9, linestyle="--", zorder=2)
    for cx in gate_taps:
        arrow(cx, GATE_Y, cx, SPINE_Y + SPINE_H, WARN, lw=0.9, dashed=True)
    ax.text(gate_taps[-1] + 1.5, GATE_Y + 1.7,
            "verdict gates interpretation of every metric below",
            fontsize=5.9, color=WARN, style="italic", ha="right", zorder=4)

    # ---- The spine: one path, one successor each ---------------------------
    # SP holds (left, width); box() takes (x, y, w, h).
    spine_text = [
        ("Stage 1  Cleaning",
         "four-layer leak defence\nsemantic / structural /\nextract hardening / audit", 1.1),
        ("Stage 2  Features",
         "29 mule-typology features\n+ row-profile aggregates", 1.1),
        ("Stage 3  Nested CV",
         "selection, stacking, calibration\nand THRESHOLD all fitted\ninside the training fold", 1.7),
        ("Stage 4  Scoring",
         "0-1000 risk score\nLOW / MEDIUM / HIGH\n+ SHAP reason lists", 1.1),
    ]
    for (x, w), (title, sub, lw) in zip(SP, spine_text):
        box(x, SPINE_Y, w, SPINE_H, title, sub, ACCENT, lw=lw)
    for (x, w), (nx, _) in zip(SP, SP[1:]):
        arrow(x + w, SPINE_Y + SPINE_H / 2, nx, SPINE_Y + SPINE_H / 2)

    # Raw data enters the spine at Stage 1, left of the gate bus.
    arrow(8, 44, 8, SPINE_Y + SPINE_H)

    # ---- Dictionary bus: named semantics feed three stages -----------------
    box(24, 2, 52, 11, "Data dictionary",
        "3,924 F-codes mapped to named banking variables — what makes\n"
        "leak classification by meaning, named features and\n"
        "plain-English SHAP reasons possible", MUTED)
    arrow(50, 13, 50, BUS_Y - 0.4, MUTED, dashed=True)
    ax.plot([centres[0], centres[-1]], [BUS_Y, BUS_Y], color=MUTED,
            linewidth=0.9, linestyle="--", zorder=2)
    for cx in (centres[0], centres[1], centres[3]):
        arrow(cx, BUS_Y, cx, SPINE_Y, MUTED, lw=0.9, dashed=True)

    _save(fig, "fig1_architecture.png")


# --------------------------------------------------------------------------
def fig_integrity() -> None:
    """The falsification tests against the random baseline — log scale."""
    a = _load("06_integrity_audit.json")
    if not a:
        return
    prev = a["prevalence"]
    items = [
        ("Missingness only\n(no values at all)", a["test_A_missingness_only"]["auprc"], WARN),
        ("250 columns each\n|corr| < 0.05", a["test_B_individually_useless"]["auprc"], WARN),
        ("Shuffled labels\n(sanity floor)", a["test_C_shuffled_labels"]["auprc"], MUTED),
        ("Random guess\n(prevalence)", prev, MUTED),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ys = np.arange(len(items))[::-1]
    ax.barh(ys, [v for _, v, _ in items],
            color=[c for _, _, c in items], height=0.62, alpha=0.85)
    for y, (_, v, _) in zip(ys, items):
        ax.text(v * 1.15, y, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(ys); ax.set_yticklabels([n for n, _, _ in items])
    ax.set_xscale("log"); ax.set_xlim(0.005, 3.0)
    ax.set_xlabel("AUPRC (log scale)")
    ax.axvline(prev, color=INK, linestyle=":", linewidth=0.9)
    ax.set_title("Information that cannot identify a mule\nstill separates the classes",
                 fontsize=8)
    _save(fig, "fig2_integrity.png")


# --------------------------------------------------------------------------
def fig_shap() -> None:
    """Top features by mean |SHAP|, labelled with real banking variable names.

    Drawn at exactly one column wide so LaTeX places it 1:1. Downscaling a wider
    figure into \\columnwidth shrinks the tick labels below legibility, which is
    what happened in the first draft.
    """
    s = _load("05_shap_top_features.json")
    if not s:
        return
    top = s["top_features_by_mean_abs_shap"][:12][::-1]
    names = [t["variable"] for t in top]
    vals = [t["mean_abs_shap"] for t in top]
    # Engineered features get the accent colour to separate them from the bank's
    # own supplied variables.
    colors = [ACCENT if n.startswith("mg_") else MUTED for n in names]

    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    ys = np.arange(len(names))
    ax.barh(ys, vals, color=colors, height=0.7, alpha=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=5.6)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("mean |SHAP| contribution", fontsize=7)
    ax.set_title("Teal = engineered in this work", fontsize=7)
    _save(fig, "fig3_shap.png")


# --------------------------------------------------------------------------
def fig_bands() -> None:
    """Risk-band triage: how many accounts, how many mules, what action."""
    sc = _load("05_scoring_report.json")
    if not sc:
        return
    bands = ["LOW", "MEDIUM", "HIGH"]
    stats = sc["band_stats"]
    counts = [stats[b]["accounts"] for b in bands]
    mules = [stats[b]["true_mules"] for b in bands]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W * 0.62, 2.3))
    x = np.arange(3)
    ax1.bar(x, counts, color=[MUTED, "#e0a458", WARN], alpha=0.85)
    ax1.set_yscale("log"); ax1.set_xticks(x); ax1.set_xticklabels(bands)
    ax1.set_ylabel("accounts (log)"); ax1.set_title("Queue size", fontsize=8)
    for i, v in enumerate(counts):
        ax1.text(i, v * 1.2, f"{v:,}", ha="center", fontsize=6.5)

    prec = [stats[b]["precision"] for b in bands]
    ax2.bar(x, prec, color=[MUTED, "#e0a458", WARN], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(bands)
    ax2.set_ylabel("share that are mules"); ax2.set_ylim(0, 1.1)
    ax2.set_title("Band precision", fontsize=8)
    for i, (p, m) in enumerate(zip(prec, mules)):
        ax2.text(i, p + 0.05, f"{p:.2f}\n({m})", ha="center", fontsize=6.5)
    fig.tight_layout()
    _save(fig, "fig4_bands.png")


# --------------------------------------------------------------------------
def fig_ablation() -> None:
    """Per-model AUPRC with error bars — including the Isolation Forest result."""
    m = _load("03_metrics.json")
    if not m:
        return
    order = ["iso", "lgbm", "xgb"]
    label = {"iso": "Isolation\nForest", "lgbm": "LightGBM", "xgb": "XGBoost"}
    pm = m["per_model"]
    names = [label[k] for k in order if k in pm] + ["Calibrated\nensemble"]
    means = [pm[k]["auprc"]["mean"] for k in order if k in pm]
    stds = [pm[k]["auprc"]["std"] for k in order if k in pm]
    e = m["ensemble_precision_first"]["auprc"]
    means.append(e["mean"]); stds.append(e["std"])
    colors = [MUTED] * (len(means) - 1) + [ACCENT]

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.9,
           error_kw={"linewidth": 0.8})
    ax.axhline(m["auprc_baseline_random"], color=WARN, linestyle=":", linewidth=0.9)
    ax.text(len(names) - 0.5, m["auprc_baseline_random"] * 1.6, "random baseline",
            fontsize=6, color=WARN, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylabel("AUPRC"); ax.set_ylim(0, 1.05)
    ax.set_title("Base models vs ensemble (mean +/- std)", fontsize=8)
    _save(fig, "fig5_ablation.png")


def main() -> None:
    fig_architecture()
    fig_integrity()
    fig_shap()
    fig_bands()
    fig_ablation()
    log(f"All paper figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
