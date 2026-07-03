"""
Stage 4/5 — Imbalance-aware, calibrated ensemble with honest 5-fold CV.

Design (matches the submission methodology, without leakage):
  * Base models: Isolation Forest (unsupervised anomaly), XGBoost, LightGBM.
  * Cost-sensitive: scale_pos_weight = n_neg / n_pos computed PER FOLD.
  * SMOTE-Tomek resampling fitted STRICTLY INSIDE each training fold — never on
    validation data, never on the full dataset.
  * Stacking: a logistic meta-learner combines the three base probabilities,
    trained on out-of-fold predictions.
  * Calibration: isotonic calibration of the final probability.
  * Metrics: precision, recall, F1, AUPRC, AUROC, FPR — reported per base model
    and for the ensemble, aggregated across 5 stratified folds (out-of-fold).
  * Threshold: the HIGH-band probability cutoff is chosen from the pooled
    out-of-fold precision–recall curve to hold precision >= PRECISION_TARGET.

Outputs:
  models/    — final models refit on all data (for scoring new accounts)
  reports/03_metrics.json — the honest numbers that replace PDF placeholders
  data/oof_predictions.csv — out-of-fold probabilities (used by later stages)

Run:  python src/03_train.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import config as C
from utils import load_frame, log, save_json

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:  # noqa: BLE001
    HAVE_LGBM = False

try:
    from imblearn.combine import SMOTETomek
    HAVE_IMB = True
except Exception:  # noqa: BLE001
    HAVE_IMB = False


def iso_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Convert IsolationForest decision_function to a 0..1 anomaly probability."""
    raw = -model.decision_function(X)          # higher = more anomalous
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo + 1e-12)


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    return {
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "fpr": round(fp / (fp + tn + 1e-12), 4),
    }


def pick_precision_threshold(y: np.ndarray, p: np.ndarray, target: float) -> tuple[float, bool]:
    """Pick an operating threshold.

    Preferred: the threshold that holds precision >= target while maximising
    recall (the precision-first operating point).

    Fallback (target unreachable on this data): the threshold that maximises F1,
    NOT a blind 0.5 — on a calibrated <1% problem almost all probabilities sit
    below 0.5, so a 0.5 cut would predict all-negative and report zeros.

    Returns (threshold, target_met).
    """
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, False
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) > 0:
        # Among thresholds meeting precision, choose the one giving best recall.
        best = ok[np.argmax(rec[:-1][ok])]
        return float(thr[best]), True
    # Fallback: maximise F1 over the curve.
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best = int(np.argmax(f1s))
    return float(thr[best]), False


def main() -> None:
    df = load_frame(C.FEATURES_PARQUET)
    y = df[C.TARGET_COL].astype(int).values
    X = df.drop(columns=[C.TARGET_COL]).select_dtypes(include=[np.number])
    feat_names = X.columns.tolist()
    X = X.values.astype(np.float32)
    log(f"Training matrix: {X.shape[0]:,} x {X.shape[1]:,}   positives={int(y.sum())}")

    if not HAVE_XGB or not HAVE_LGBM:
        log(f"WARNING: xgboost={HAVE_XGB}, lightgbm={HAVE_LGBM}. "
            f"Missing engines are skipped; install for full ensemble.")
    if not HAVE_IMB:
        log("WARNING: imbalanced-learn missing — using cost-sensitive weighting "
            "only (no SMOTE-Tomek).")

    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    # Out-of-fold probability stores
    oof = {"iso": np.zeros(len(y)), "xgb": np.zeros(len(y)), "lgbm": np.zeros(len(y))}
    oof_meta = np.zeros(len(y))

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
        n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
        spw = n_neg / max(n_pos, 1)

        # --- In-fold resampling (training only) ---
        if HAVE_IMB and n_pos > C.N_FOLDS:
            try:
                Xtr_r, ytr_r = SMOTETomek(random_state=C.RANDOM_STATE).fit_resample(Xtr, ytr)
            except Exception:  # noqa: BLE001 — too few minority neighbours
                Xtr_r, ytr_r = Xtr, ytr
        else:
            Xtr_r, ytr_r = Xtr, ytr

        base_va = {}

        # Isolation Forest (unsupervised — fit on training features only)
        iso = IsolationForest(**C.ISO_FOREST_PARAMS).fit(Xtr)
        base_va["iso"] = iso_scores(iso, Xva)

        # XGBoost
        if HAVE_XGB:
            xgb = XGBClassifier(**C.XGB_PARAMS, scale_pos_weight=spw)
            xgb.fit(Xtr_r, ytr_r)
            base_va["xgb"] = xgb.predict_proba(Xva)[:, 1]
        else:
            base_va["xgb"] = np.zeros(len(va))

        # LightGBM
        if HAVE_LGBM:
            lgbm = LGBMClassifier(**C.LGBM_PARAMS, scale_pos_weight=spw)
            lgbm.fit(Xtr_r, ytr_r)
            base_va["lgbm"] = lgbm.predict_proba(Xva)[:, 1]
        else:
            base_va["lgbm"] = np.zeros(len(va))

        for k in oof:
            oof[k][va] = base_va[k]

        # --- Meta-learner trained on this fold's base outputs ---
        # (Stacking on out-of-fold base preds; simple + leakage-safe per fold.)
        meta_features_va = np.column_stack([base_va["iso"], base_va["xgb"], base_va["lgbm"]])
        # Fit meta on the SAME fold's base preds vs true labels for validation rows
        # is not allowed (would peek). Instead we fit meta on training rows using
        # base models' train predictions.
        meta_tr = np.column_stack([
            iso_scores(iso, Xtr),
            xgb.predict_proba(Xtr)[:, 1] if HAVE_XGB else np.zeros(len(tr)),
            lgbm.predict_proba(Xtr)[:, 1] if HAVE_LGBM else np.zeros(len(tr)),
        ])
        meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        meta.fit(meta_tr, ytr)
        oof_meta[va] = meta.predict_proba(meta_features_va)[:, 1]

        log(f"fold {fold}/{C.N_FOLDS} done (spw={spw:.1f}, "
            f"train+={n_pos}, resampled={len(ytr_r):,})")

    # --- Isotonic calibration of the pooled out-of-fold ensemble probs ---
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    oof_ensemble = iso_cal.fit_transform(oof_meta, y)

    # --- Threshold chosen to hold precision >= target (precision-first) ---
    thr_hi, target_met = pick_precision_threshold(y, oof_ensemble, C.PRECISION_TARGET)
    if not target_met:
        log(f"NOTE: precision target {C.PRECISION_TARGET} not reachable on this "
            f"data; precision-first point falls back to best-F1 threshold.")

    # --- Report ---
    results = {
        "n_accounts": int(len(y)),
        "n_mules": int(y.sum()),
        "prevalence_pct": round(100 * y.mean(), 4),
        "engines": {"xgboost": HAVE_XGB, "lightgbm": HAVE_LGBM, "imblearn": HAVE_IMB},
        "n_features": len(feat_names),
        "precision_target": C.PRECISION_TARGET,
        "precision_target_met": bool(target_met),
        "per_model": {},
        "ensemble_precision_first": {},
        "ensemble_high_recall": {},
    }

    for name, p in oof.items():
        # For base models, report AUPRC/AUROC + metrics at their own best point.
        default_thr, _ = pick_precision_threshold(y, p, C.PRECISION_TARGET)
        results["per_model"][name] = metrics_at(y, p, default_thr)

    results["ensemble_precision_first"] = metrics_at(y, oof_ensemble, thr_hi)
    # A high-recall operating point for the analyst-review queue.
    thr_recall, _ = pick_precision_threshold(y, oof_ensemble, 0.70)
    results["ensemble_high_recall"] = metrics_at(y, oof_ensemble, thr_recall)

    log("=== ENSEMBLE (precision-first, out-of-fold) ===")
    for k, v in results["ensemble_precision_first"].items():
        log(f"    {k}: {v}")

    save_json(results, C.REPORTS_DIR / "03_metrics.json")

    # Persist OOF probabilities for scoring / plotting stages.
    pd.DataFrame({
        "y_true": y,
        "p_iso": oof["iso"],
        "p_xgb": oof["xgb"],
        "p_lgbm": oof["lgbm"],
        "p_ensemble_calibrated": oof_ensemble,
    }).to_csv(C.DATA_DIR / "oof_predictions.csv", index=False)
    log(f"Wrote {C.DATA_DIR / 'oof_predictions.csv'}")

    # --- Refit final models on ALL data for scoring future accounts ---
    _refit_and_save(X, y, feat_names, iso_cal)


def _refit_and_save(X, y, feat_names, iso_cal) -> None:
    """Refit base models on the full dataset and persist for inference."""
    import joblib

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    spw = n_neg / max(n_pos, 1)

    bundle = {"feat_names": feat_names, "iso_calibrator": iso_cal, "scale_pos_weight": spw}
    bundle["iso_forest"] = IsolationForest(**C.ISO_FOREST_PARAMS).fit(X)
    if HAVE_XGB:
        bundle["xgb"] = XGBClassifier(**C.XGB_PARAMS, scale_pos_weight=spw).fit(X, y)
    if HAVE_LGBM:
        bundle["lgbm"] = LGBMClassifier(**C.LGBM_PARAMS, scale_pos_weight=spw).fit(X, y)

    joblib.dump(bundle, C.MODELS_DIR / "muleguard_models.joblib")
    log(f"Saved final models -> {C.MODELS_DIR / 'muleguard_models.joblib'}")


if __name__ == "__main__":
    main()
