"""
Stage 4/5 — Imbalance-aware, calibrated XGBoost with an HONEST operating point.

What changed from the earlier build (and why it is more defensible):

  * NESTED cross-validation for the operating point. The isotonic calibrator AND
    the precision-first threshold are fit on INNER out-of-fold predictions of each
    outer-training split, then applied to the UNTOUCHED outer-test fold. Pooling the
    outer-test decisions gives an out-of-sample precision/recall/F1 — not the old
    number where the threshold was picked on the very same pooled vector it was then
    scored on. The single-level pooled number is still reported, but explicitly
    LABELLED as an "optimistic ceiling (in-sample operating point)".

  * The threshold-free ranking metrics (AUPRC / AUROC) are the honest HEADLINE. They
    come from single-level out-of-fold probabilities and are unaffected by the
    monotonic isotonic calibration, so they carry no operating-point optimism.

  * Repeated nested CV (N_CV_REPEATS) + stratified bootstrap CIs quantify the (large)
    uncertainty at only 81 positives, and the spread of per-fold thresholds is
    reported so a reader can see whether the operating point is reliably estimable.

  * No DOUBLE imbalance correction. The resampling policy (USE_SMOTE / USE_SPW) is
    set from the measured ablation; the default is clean spw-only (no SMOTE), so the
    ~111 scale_pos_weight is not stacked on a ~50/50 resampled set.

  * No vestigial stacking layer. With a single base model the old logistic
    "meta-learner" only monotonically rescaled XGBoost; it is gone. The path is
    XGBoost -> isotonic calibration -> threshold.

Outputs:
  models/muleguard_models.joblib — final model refit on all data (+ calibrator) for scoring
  reports/03_metrics.json        — honest operating point, optimistic ceiling, CIs, threshold spread
  data/oof_predictions.csv       — single-level out-of-fold probabilities (used by later stages)

Run:  .venv/bin/python src/03_train.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / "experiments"))

import config as C
from utils import log, save_json
import _harness as H

warnings.filterwarnings("ignore")

HAVE_XGB = H.HAVE_XGB
HAVE_IMB = H.HAVE_IMB

# Re-export threshold/metric helpers (moved to the shared harness) so existing
# callers and tests can still reference them as train.pick_precision_threshold etc.
pick_precision_threshold = H.pick_precision_threshold
metrics_at = H.metrics_at


def _feature_selector():
    """Fold-local top-K by XGB gain, or None to keep all features."""
    k = C.FEATURE_SELECT_K
    if isinstance(k, int):
        def _select(Xtr, ytr):
            spw = (ytr == 0).sum() / max(ytr.sum(), 1)
            m = H.make_xgb(spw=spw, seed=C.RANDOM_STATE)
            m.fit(Xtr, ytr)
            if k >= Xtr.shape[1]:
                return np.arange(Xtr.shape[1])
            return np.argsort(m.feature_importances_)[::-1][:k]
        return _select
    return None


# --------------------------------------------------------------------------
# Nested-CV honest operating point.
# --------------------------------------------------------------------------
def nested_operating_point(X, y, *, seed):
    """One repeat of nested CV.

    For each outer fold: build inner OOF probs on the outer-training rows, fit
    isotonic + pick the precision-first threshold on those inner OOF probs, then
    train a fresh model on ALL outer-training rows and apply the inner calibrator +
    inner threshold to the untouched outer-test fold.

    Returns pooled outer-test (y, decision, calibrated_prob) and the per-fold
    thresholds (to report their spread).
    """
    outer = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
    sel = _feature_selector()

    pooled_y = np.zeros(len(y))
    pooled_decision = np.zeros(len(y))
    pooled_prob = np.zeros(len(y))
    fold_thresholds = []

    for otr, ote in outer.split(X, y):
        Xtr, Xte, ytr, yte = X[otr], X[ote], y[otr], y[ote]

        # Inner OOF probabilities on the outer-training rows only.
        inner_oof = H.oof_probabilities(
            Xtr, ytr, use_smote=C.USE_SMOTE, use_spw=C.USE_SPW,
            n_folds=C.INNER_FOLDS, seed=seed, feature_selector=sel,
        )
        # Calibrate + pick threshold on inner OOF (never sees outer-test).
        inner_cal = H.calibrate(inner_oof, ytr, inner_oof)
        thr, _met = H.pick_precision_threshold(ytr, inner_cal, C.PRECISION_TARGET)
        fold_thresholds.append(thr)

        # Train the final outer model on ALL outer-training rows.
        cols = sel(Xtr, ytr) if sel is not None else None
        Xtr_f = Xtr[:, cols] if cols is not None else Xtr
        Xte_f = Xte[:, cols] if cols is not None else Xte
        n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
        spw = (n_neg / max(n_pos, 1)) if C.USE_SPW else 1.0
        Xtr_r, ytr_r = H._resample(Xtr_f, ytr, C.USE_SMOTE, seed)
        model = H.make_xgb(spw=spw, seed=seed).fit(Xtr_r, ytr_r)

        raw_te = model.predict_proba(Xte_f)[:, 1]
        # Apply the inner calibrator to outer-test raw probs.
        cal_te = H.calibrate(inner_oof, ytr, raw_te)

        pooled_y[ote] = yte
        pooled_prob[ote] = cal_te
        pooled_decision[ote] = (cal_te >= thr).astype(int)

    return pooled_y, pooled_decision, pooled_prob, fold_thresholds


def _pooled_decision_metrics(y, decision):
    """Precision/recall/F1/FPR from pooled out-of-sample DECISIONS (not a threshold)."""
    yhat = decision.astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "fpr": round(fp / max(fp + tn, 1), 4), "tp": tp, "fp": fp, "fn": fn,
    }


def main() -> None:
    data = H.load_features()
    X, y, feat_names = data.X, data.y, data.feat_names
    log(f"Training matrix: {X.shape[0]:,} x {X.shape[1]:,}   positives={int(y.sum())}")
    if not HAVE_XGB:
        log("WARNING: xgboost not installed — the primary engine is missing.")
    log(f"Policy: USE_SMOTE={C.USE_SMOTE}, USE_SPW={C.USE_SPW}, "
        f"FEATURE_SELECT_K={C.FEATURE_SELECT_K}, repeats={C.N_CV_REPEATS}")

    # --- Honest headline: threshold-free ranking from single-level OOF ---
    # AUPRC/AUROC are computed on the RAW out-of-fold probabilities, not the
    # in-sample-calibrated ones: isotonic is only weakly monotonic, so fitting it
    # in-sample on the same labels it is scored against would nudge AUROC up via the
    # ROC convex hull. Raw OOF carries zero calibration optimism. Calibration quality
    # (Brier/ECE) is instead measured OUT-OF-SAMPLE from the nested-CV pooled probs
    # below — never on rows used to fit the calibrator.
    from sklearn.metrics import average_precision_score, roc_auc_score

    sel = _feature_selector()
    oof_single = H.oof_probabilities(X, y, use_smote=C.USE_SMOTE, use_spw=C.USE_SPW,
                                     feature_selector=sel)
    ranking = {
        "auprc": round(float(average_precision_score(y, oof_single)), 4),
        "auroc": round(float(roc_auc_score(y, oof_single)), 4),
    }
    log(f"Honest ranking (raw single-level OOF): AUPRC={ranking['auprc']}  "
        f"AUROC={ranking['auroc']}")

    # --- Honest operating point: repeated nested CV ---
    log(f"Nested CV for honest operating point ({C.N_CV_REPEATS} repeats)...")
    rep_metrics, all_thresholds = [], []
    oos_cal_first = None   # pooled OUT-OF-SAMPLE calibrated probs (first repeat) for honest Brier/ECE
    for r in range(C.N_CV_REPEATS):
        py, pd_dec, pp, thrs = nested_operating_point(X, y, seed=C.RANDOM_STATE + r)
        m = _pooled_decision_metrics(py, pd_dec)
        rep_metrics.append(m)
        all_thresholds.extend(thrs)
        if oos_cal_first is None:
            oos_cal_first = pp   # calibrator fit on inner-OOF, applied to untouched outer-test
        if r == 0 or (r + 1) % 5 == 0:
            log(f"  repeat {r+1}/{C.N_CV_REPEATS}: P={m['precision']} R={m['recall']} "
                f"F1={m['f1']} (TP={m['tp']} FP={m['fp']})")

    # Out-of-sample calibration quality (Brier/ECE on nested-CV pooled probs — the
    # calibrator never saw these rows, unlike an in-sample fit).
    from sklearn.metrics import brier_score_loss
    ranking["brier_oos"] = round(float(brier_score_loss(y, oos_cal_first)), 6)
    ranking["ece_oos"] = round(float(H.expected_calibration_error(y, oos_cal_first)), 4)
    log(f"Out-of-sample calibration: Brier={ranking['brier_oos']}  ECE={ranking['ece_oos']}")

    def _agg(key):
        vals = [m[key] for m in rep_metrics]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    honest_op = {k: _agg(k) for k in ("precision", "recall", "f1", "fpr")}
    honest_op["mean_tp"] = round(float(np.mean([m["tp"] for m in rep_metrics])), 1)
    honest_op["mean_fp"] = round(float(np.mean([m["fp"] for m in rep_metrics])), 1)
    honest_op["threshold_spread"] = {
        "median": round(float(np.median(all_thresholds)), 4),
        "min": round(float(np.min(all_thresholds)), 4),
        "max": round(float(np.max(all_thresholds)), 4),
        "std": round(float(np.std(all_thresholds)), 4),
    }
    log(f"HONEST operating point: P={honest_op['precision']['mean']}±"
        f"{honest_op['precision']['std']}  R={honest_op['recall']['mean']}±"
        f"{honest_op['recall']['std']}  F1={honest_op['f1']['mean']}")

    # --- Stratified bootstrap CIs on the honest ranking (RAW OOF — rank-honest) ---
    auprc_ci = H.stratified_bootstrap_ci(y, oof_single, average_precision_score,
                                         n_boot=C.BOOTSTRAP_N)
    auroc_ci = H.stratified_bootstrap_ci(y, oof_single, roc_auc_score, n_boot=C.BOOTSTRAP_N)
    # Recall CI: the across-repeats std (honest_op['recall']['std']) is the primary
    # uncertainty; this Wilson interval is a binomial-only sanity floor over the 81
    # positives and is labelled as such (it ignores fold/model variance).
    mean_recall = honest_op["recall"]["mean"]
    recall_ci = H.wilson_ci(int(round(mean_recall * int(y.sum()))), int(y.sum()))

    # --- Optimistic ceiling: the OLD in-sample operating point (labelled as such) ---
    # This one INTENTIONALLY uses in-sample calibration + threshold on the same OOF
    # it scores — that is precisely what makes it the "optimistic ceiling" we contrast
    # the honest nested-CV number against.
    oof_cal_insample = H.calibrate(oof_single, y, oof_single)
    thr_pool, met_pool = H.pick_precision_threshold(y, oof_cal_insample, C.PRECISION_TARGET)
    ceiling = H.metrics_at(y, oof_cal_insample, thr_pool)

    # --- Assemble report ---
    results = {
        "n_accounts": int(len(y)),
        "n_mules": int(y.sum()),
        "prevalence_pct": round(100 * y.mean(), 4),
        "n_features": len(feat_names),
        "policy": {"use_smote": C.USE_SMOTE, "use_spw": C.USE_SPW,
                   "feature_select_k": C.FEATURE_SELECT_K, "n_cv_repeats": C.N_CV_REPEATS},
        "precision_target": C.PRECISION_TARGET,
        "headline_ranking": ranking,
        "headline_ranking_ci": {"auprc": auprc_ci, "auroc": auroc_ci},
        "honest_operating_point": honest_op,
        "honest_operating_point_recall_ci": recall_ci,
        "optimistic_ceiling": {
            "note": ("in-sample operating point — threshold picked on the same pooled OOF "
                     "it is scored on; reported for comparison only, NOT the headline"),
            "precision_target_met": bool(met_pool),
            **ceiling,
        },
        "provenance": ("Ranking (AUPRC/AUROC) from single-level OOF (honest, calibration is "
                       "monotonic). Operating point from repeated nested CV: calibrator+threshold "
                       "fit on inner OOF, applied to untouched outer-test. See "
                       "reports/06_leak_audit.json for the calendar/cohort leak audit."),
    }

    log("=== HEADLINE (honest) ===")
    log(f"    AUPRC={ranking['auprc']} (95% CI {auprc_ci['lo']}–{auprc_ci['hi']})")
    log(f"    AUROC={ranking['auroc']} (95% CI {auroc_ci['lo']}–{auroc_ci['hi']})")
    log(f"    Operating point P={honest_op['precision']['mean']} "
        f"R={honest_op['recall']['mean']} (nested-CV, out-of-sample)")
    log(f"    Optimistic ceiling (in-sample): P={ceiling['precision']} R={ceiling['recall']}")

    save_json(results, C.REPORTS_DIR / "03_metrics.json")

    # Persist single-level OOF for scoring/plot stages. p_ensemble_calibrated is the
    # in-sample-calibrated OOF prob used for RISK SCORING (0-1000 bands) and the PR/
    # ROC/calibration plots — not for the headline ranking metrics, which use raw OOF.
    pd.DataFrame({
        "y_true": y,
        "p_xgb": oof_single,
        "p_ensemble_calibrated": oof_cal_insample,
    }).to_csv(C.DATA_DIR / "oof_predictions.csv", index=False)
    log(f"Wrote {C.DATA_DIR / 'oof_predictions.csv'}")

    # Refit final model on ALL data + persist calibrator for inference.
    _refit_and_save(X, y, feat_names, oof_single)


def _refit_and_save(X, y, feat_names, oof_single) -> None:
    """Refit XGBoost on all data and persist model + isotonic calibrator for inference."""
    import joblib
    from sklearn.isotonic import IsotonicRegression

    sel = _feature_selector()
    cols = sel(X, y) if sel is not None else None
    Xf = X[:, cols] if cols is not None else X
    sub_names = [feat_names[i] for i in cols] if cols is not None else feat_names

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    spw = (n_neg / max(n_pos, 1)) if C.USE_SPW else 1.0
    Xr, yr = H._resample(Xf, y, C.USE_SMOTE, C.RANDOM_STATE)
    model = H.make_xgb(spw=spw, seed=C.RANDOM_STATE).fit(Xr, yr)

    # Calibrator fit on single-level OOF (honest mapping raw->calibrated prob).
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_single, y)

    bundle = {
        "feat_names": sub_names,
        "iso_calibrator": iso,
        "scale_pos_weight": spw,
        "xgb": model,
        "policy": {"use_smote": C.USE_SMOTE, "use_spw": C.USE_SPW,
                   "feature_select_k": C.FEATURE_SELECT_K},
    }
    joblib.dump(bundle, C.MODELS_DIR / "muleguard_models.joblib")
    log(f"Saved final model -> {C.MODELS_DIR / 'muleguard_models.joblib'}")


if __name__ == "__main__":
    main()
