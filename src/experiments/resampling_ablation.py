"""
Experiment C — resampling ablation (retires the double-correction bug).

The current pipeline corrects class imbalance TWICE: it fits SMOTE-Tomek to a
~50/50 rebalanced training set AND passes scale_pos_weight ~= 111 (computed from
the ORIGINAL 9001/81 ratio) on top. Those two corrections compound rather than
being alternatives, distorting the effective class ratio the trees train on.

This ablation measures four arms on honest, threshold-free metrics + calibration:

  (a) smote_only : SMOTE-Tomek in-fold, scale_pos_weight = 1
  (b) spw_only   : no SMOTE, scale_pos_weight = n_neg/n_pos   (cleanest)
  (c) both       : SMOTE-Tomek + scale_pos_weight ~= 111       (current default)
  (d) none       : no SMOTE, scale_pos_weight = 1              (true baseline)

Selection metric = mean AUPRC over repeated CV (threshold-free, so it does NOT
couple with the separate operating-point-optimism issue). Brier score + ECE are
reported too because the product bins probabilities into 0-1000 risk bands, so
calibration quality matters — and SMOTE/both distort the base rate, which usually
degrades calibration.

Run:  .venv/bin/python src/experiments/resampling_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "experiments"))

import config as C  # noqa: E402
from utils import log, save_json  # noqa: E402
import _harness as H  # noqa: E402


ARMS = {
    "smote_only": {"use_smote": True, "use_spw": False},
    "spw_only": {"use_smote": False, "use_spw": True},
    "both": {"use_smote": True, "use_spw": True},
    "none": {"use_smote": False, "use_spw": False},
}


def main() -> None:
    data = H.load_features()
    X, y = data.X, data.y
    log(f"Resampling ablation on {X.shape[0]:,} x {X.shape[1]:,}, positives={int(y.sum())}")

    n_rep = 5
    report: dict = {"n_repeats": n_rep, "arms": {}}

    for name, cfg in ARMS.items():
        log(f"Arm '{name}' ({cfg}) — {n_rep} repeats ...")
        oofs = H.repeated_oof(X, y, n_repeats=n_rep, **cfg)
        per = [H.ranking_metrics(y, p) for p in oofs]
        agg = {}
        for k in per[0]:
            vals = [d[k] for d in per]
            agg[k] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals)), 4)}
        # Also report recall@P>=0.90 as a SECONDARY descriptor (not the selector).
        r_at_p = []
        for p in oofs:
            thr, met = H.pick_precision_threshold(y, p, C.PRECISION_TARGET)
            r_at_p.append(H.metrics_at(y, p, thr)["recall"] if met else 0.0)
        agg["recall_at_p90_secondary"] = {"mean": round(float(np.mean(r_at_p)), 4),
                                          "std": round(float(np.std(r_at_p)), 4)}
        report["arms"][name] = agg
        log(f"    AUPRC={agg['auprc']['mean']}±{agg['auprc']['std']}  "
            f"Brier={agg['brier']['mean']}  ECE={agg['ece']['mean']}  "
            f"R@P90={agg['recall_at_p90_secondary']['mean']}")

    # Winner by mean AUPRC (primary), tie-break by lower ECE.
    winner = max(report["arms"].items(),
                 key=lambda kv: (kv[1]["auprc"]["mean"], -kv[1]["ece"]["mean"]))[0]
    report["winner"] = winner
    report["selection_rule"] = ("max mean AUPRC over repeated CV, tie-break lower ECE; "
                                "recall@P>=0.90 reported as secondary only")
    log(f"WINNER: {winner}")

    save_json(report, C.REPORTS_DIR / "07_resampling_ablation.json")
    log("Wrote reports/07_resampling_ablation.json")


if __name__ == "__main__":
    main()
