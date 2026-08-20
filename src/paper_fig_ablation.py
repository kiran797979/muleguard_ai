"""
Figure 6 — the feature ablation, drawn for the paper.

This is the result the rest of the paper's numbers have to be read against: the
raw columns barely outperform a model given no values at all, so most of the
headline is extract provenance rather than detection. Drawing it beside the
blank-pattern baseline is what makes that legible in one glance.

Run:  python src/paper_fig_ablation.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from utils import log

FIG_DIR = C.REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEAL = "#128a7d"
RED = "#b3261e"
GREY = "#6b7280"


def main() -> None:
    ab_path = C.REPORTS_DIR / "08_feature_ablation.json"
    ig_path = C.REPORTS_DIR / "06_integrity_audit.json"
    if not ab_path.exists():
        log("No 08_feature_ablation.json — run src/08_feature_ablation.py first.")
        return

    ab = json.loads(ab_path.read_text(encoding="utf-8"))
    get = lambda k: next(r for r in ab["results"] if r["condition"].startswith(k))
    full, raw, typ = get("FULL"), get("RAW"), get("TYPOLOGY")

    artefact = 0.8236
    baseline = 0.0089
    if ig_path.exists():
        ig = json.loads(ig_path.read_text(encoding="utf-8"))
        a = ig.get("test_A_missingness_only", {})
        if a.get("auprc"):
            artefact = a["auprc"]
        baseline = ig.get("auprc_random_baseline", baseline)

    labels = [
        f"All features\n({full['n_features']:,})",
        f"Raw columns only\n({raw['n_features']:,})",
        f"Behavioural only\n({typ['n_features']})",
    ]
    means = [full["auprc"]["mean"], raw["auprc"]["mean"], typ["auprc"]["mean"]]
    errs = [full["auprc"]["std"], raw["auprc"]["std"], typ["auprc"]["std"]]
    colours = [GREY, RED, TEAL]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.barh(labels, means, xerr=errs, color=colours, height=.58,
                   error_kw=dict(ecolor="#333", lw=1.1, capsize=3))

    ax.axvline(artefact, color=RED, ls="--", lw=1.6,
               label=f"Blank patterns only, no values ({artefact:.3f})")
    ax.axvline(baseline, color="#999", ls=":", lw=1.3,
               label=f"Random baseline ({baseline:.4f})")

    # Value labels sit in a clear gutter to the right of every bar, so they never
    # collide with the reference lines the argument depends on.
    for b, m, e in zip(bars, means, errs):
        ax.text(1.03, b.get_y() + b.get_height() / 2, f"{m:.3f}",
                va="center", ha="left", fontsize=10.5, fontweight="bold")

    # The gap that carries the argument. Stated as text in the empty region to
    # the right of the shortest bar: an arrow between two near-identical values
    # collapses to an unreadable blob at this scale.
    gap = raw["auprc"]["mean"] - artefact
    ax.text(0.40, 2.00,
            "Raw columns beat a\n"
            "values-free model by\n"
            f"only {gap:+.3f} AUPRC.",
            fontsize=8.5, color=RED, fontweight="bold", va="center",
            linespacing=1.55)

    ax.set_xlabel("AUPRC (5-fold, identical folds across conditions)")
    ax.set_xlim(0, 1.16)
    ax.set_ylim(2.6, -0.6)
    ax.grid(axis="x", alpha=.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=8.5, frameon=False)
    ax.set_title("Removing the behavioural features costs nothing.\n"
                 "The raw columns barely beat a model with no values at all.",
                 fontsize=10.5, loc="left", pad=12)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig6_feature_ablation.{ext}", dpi=300)
    plt.close(fig)
    log(f"wrote fig6_feature_ablation.png + .pdf  "
        f"(raw {raw['auprc']['mean']:.4f} vs artefact {artefact:.4f})")


if __name__ == "__main__":
    main()
