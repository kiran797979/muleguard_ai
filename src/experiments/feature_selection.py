"""
Experiment B — in-fold feature-selection sensitivity.

There are ~1614 features but only 81 positives (~20:1). This experiment measures
whether cutting dimensionality by XGBoost-gain top-K helps, hurts, or is neutral
on honest AUPRC. Selection is done STRICTLY inside each training fold (a prelim
XGB is fit on the training slice, its gain importances rank features, top-K kept),
so it cannot leak the validation labels.

Crucially, K itself is a hyperparameter: reporting the OOF-AUPRC of the K that
maximises OOF-AUPRC would be selection optimism on K. So each K is evaluated with
its own honest OOF and reported as a SENSITIVITY curve — the pipeline's default K
is then chosen by this curve (documented), not tuned against the final headline.

Run:  .venv/bin/python src/experiments/feature_selection.py
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


def gain_selector(k: int):
    """Return a fold-local selector: fit a prelim XGB on (Xtr,ytr), keep top-k by gain."""
    def _select(Xtr, ytr):
        spw = (ytr == 0).sum() / max(ytr.sum(), 1)
        m = H.make_xgb(spw=spw, seed=C.RANDOM_STATE)
        m.fit(Xtr, ytr)
        imp = m.feature_importances_
        if k >= Xtr.shape[1]:
            return np.arange(Xtr.shape[1])
        return np.argsort(imp)[::-1][:k]
    return _select


def main() -> None:
    data = H.load_features()
    X, y = data.X, data.y
    p_total = X.shape[1]
    log(f"Feature-selection sensitivity on {X.shape[0]:,} x {p_total:,}, positives={int(y.sum())}")

    n_rep = 3
    ks = [k for k in (100, 200, 400) if k < p_total] + [p_total]
    report: dict = {"n_repeats": n_rep, "n_features_total": p_total, "sweep": {}}

    # Use the cleaner spw-only config for the sweep (SMOTE decision handled by C).
    for k in ks:
        label = "all" if k >= p_total else str(k)
        sel = None if k >= p_total else gain_selector(k)
        log(f"K={label} — {n_rep} repeats ...")
        oofs = H.repeated_oof(X, y, n_repeats=n_rep, use_smote=False, use_spw=True,
                              feature_selector=sel)
        auprc = [H.ranking_metrics(y, p)["auprc"] for p in oofs]
        auroc = [H.ranking_metrics(y, p)["auroc"] for p in oofs]
        report["sweep"][label] = {
            "auprc_mean": round(float(np.mean(auprc)), 4),
            "auprc_std": round(float(np.std(auprc)), 4),
            "auroc_mean": round(float(np.mean(auroc)), 4),
        }
        log(f"    K={label}: AUPRC={report['sweep'][label]['auprc_mean']}"
            f"±{report['sweep'][label]['auprc_std']}  "
            f"AUROC={report['sweep'][label]['auroc_mean']}")

    best = max(report["sweep"].items(), key=lambda kv: kv[1]["auprc_mean"])[0]
    all_auprc = report["sweep"]["all"]["auprc_mean"]
    best_auprc = report["sweep"][best]["auprc_mean"]
    # Only prefer selection if it clears a meaningful margin over "all".
    report["recommended_k"] = best if (best_auprc - all_auprc) > 0.005 else "all"
    report["note"] = ("K chosen by this sensitivity curve, not tuned against the headline. "
                      "Selection preferred over 'all' only if AUPRC gain > 0.005.")
    log(f"Recommended K: {report['recommended_k']} "
        f"(best={best}@{best_auprc}, all@{all_auprc})")

    save_json(report, C.REPORTS_DIR / "08_feature_selection.json")
    log("Wrote reports/08_feature_selection.json")


if __name__ == "__main__":
    main()
