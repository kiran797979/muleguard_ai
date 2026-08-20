"""
Reporting plots — PR curve, ROC curve, calibration curve.
Reads out-of-fold predictions and writes PNGs to reports/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from utils import log


def main() -> None:
    path = C.DATA_DIR / "oof_predictions.csv"
    if not path.exists():
        log("No oof_predictions.csv — run Stage 4/5 first. Skipping plots.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            precision_recall_curve, roc_curve, average_precision_score, roc_auc_score,
        )
        from sklearn.calibration import calibration_curve
    except Exception as e:  # noqa: BLE001
        log(f"Plotting libs unavailable ({e}); skipping plots.")
        return

    df = pd.read_csv(path)
    y = df["y_true"].values
    p = df["p_ensemble_calibrated"].values

    # PR curve
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#128a7d")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AUPRC={average_precision_score(y, p):.3f})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C.REPORTS_DIR / "pr_curve.png", dpi=130); plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f5fa8"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUROC={roc_auc_score(y, p):.4f})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C.REPORTS_DIR / "roc_curve.png", dpi=130); plt.close()

    # Calibration curve.
    #
    # NOT quantile-binned. With 0.89% prevalence, 94% of accounts score below
    # p=0.002, so ten equal-count bins put NINE of them in that sliver: they
    # stack on top of each other at the origin and the plot shows nothing at all
    # about the range an analyst actually acts on. The previous version of this
    # function did exactly that and produced a chart with two visible points.
    #
    # Fixed probability bands instead, each annotated with how many accounts and
    # how many mules fall in it, so a sparsely populated bin is visibly sparse
    # rather than quietly averaged away.
    edges = [0.0, 0.001, 0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0001]
    xs, ys, ns, ks = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
        ks.append(int(y[m].sum()))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="perfectly calibrated")
    ax.plot(xs, ys, "-", color="#128a7d", linewidth=1.6, zorder=2)
    # Marker area carries the bin population, so small bins cannot masquerade
    # as strong evidence.
    ax.scatter(xs, ys, s=[max(28, min(320, 26 * (n ** 0.5))) for n in ns],
               color="#128a7d", zorder=3, alpha=.85, edgecolors="white", linewidths=.8)
    # The four lowest bands sit almost on top of each other near the origin, so
    # their labels are fanned upward with leader lines rather than all landing
    # in the same place and becoming an unreadable pile.
    low_offsets = [(18, 0), (18, 16), (18, 32), (18, 48)]
    low_i = 0
    for x, yv, n, k in zip(xs, ys, ns, ks):
        crowded = x < 0.10
        if crowded:
            off = low_offsets[min(low_i, len(low_offsets) - 1)]
            low_i += 1
        else:
            off = (11, -15)
        ax.annotate(f"n={n:,} · {k} mule{'s' if k != 1 else ''}",
                    (x, yv), textcoords="offset points", xytext=off,
                    fontsize=7.5, color="#33474a",
                    arrowprops=(dict(arrowstyle="-", color="#9bb0b2", lw=.6,
                                     shrinkA=0, shrinkB=3) if crowded else None))
    ax.set_xlabel("Mean predicted probability of being a mule")
    ax.set_ylabel("Observed fraction that really are mules")
    ax.set_title("Calibration, binned by probability band\n"
                 "(quantile bins collapse at 0.89% prevalence)", fontsize=11)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "calibration_curve.png", dpi=130); plt.close(fig)

    # Keep the raw bin table alongside the picture so the numbers can be checked
    # without re-deriving them from the predictions.
    from utils import save_json
    save_json({
        "note": "Calibration binned by fixed probability bands. Quantile binning "
                "places 9 of 10 bins below p=0.002 at this prevalence and hides "
                "the operating range entirely.",
        "bins": [{"mean_predicted": round(x, 5), "observed_rate": round(v, 5),
                  "accounts": n, "mules": k}
                 for x, v, n, k in zip(xs, ys, ns, ks)],
    }, C.REPORTS_DIR / "07_calibration_bins.json")

    log("Wrote pr_curve.png, roc_curve.png, calibration_curve.png to reports/")


if __name__ == "__main__":
    main()
