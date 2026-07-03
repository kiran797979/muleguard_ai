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

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, "o-", color="#128a7d"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction of mules")
    plt.title("Calibration (reliability) curve")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C.REPORTS_DIR / "calibration_curve.png", dpi=130); plt.close()

    log("Wrote pr_curve.png, roc_curve.png, calibration_curve.png to reports/")


if __name__ == "__main__":
    main()
