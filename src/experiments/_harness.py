"""
Shared experiment harness for MuleGuard's honest-rebuild experiments (E, C, B, D, A).

Everything here is deliberately leakage-safe and reused across experiments so the
numbers stay comparable:

  * single-level stratified out-of-fold (OOF) probabilities (honest ranking metrics),
  * repeated stratified CV (variance reduction — the single biggest lever at N=81),
  * in-fold SMOTE-Tomek + per-fold scale_pos_weight (both optional, so the
    resampling ablation can turn each on/off independently),
  * stratified bootstrap confidence intervals (resample positives and negatives
    separately so prevalence and >=1 positive are preserved),
  * a precision-first threshold picker (holds precision >= target, maximises recall),
  * the calibrated-probability -> metrics helpers.

None of these fit anything on validation rows. The XGBoost model, SMOTE, and the
per-fold scale_pos_weight are all fit strictly inside the training slice of a fold.

This module lives under src/experiments/ with an importable name (no numeric prefix)
so both the experiment scripts and the rewritten Stage 4/5 can import it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import config as C

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    HAVE_XGB = False

try:
    from imblearn.combine import SMOTETomek
    HAVE_IMB = True
except Exception:  # noqa: BLE001
    HAVE_IMB = False


# --------------------------------------------------------------------------
# Model factory — one place to build the base learner so every experiment
# trains the *same* estimator (only the knobs an experiment varies change).
# --------------------------------------------------------------------------
def make_xgb(params: dict | None = None, spw: float = 1.0, seed: int | None = None):
    """Build an XGBClassifier from config defaults with optional overrides."""
    p = dict(C.XGB_PARAMS)
    if params:
        p.update(params)
    if seed is not None:
        p["random_state"] = seed
    return XGBClassifier(**p, scale_pos_weight=spw)


def _resample(Xtr, ytr, use_smote: bool, seed: int):
    """In-fold SMOTE-Tomek (training rows only). Falls back to raw if it can't run."""
    if not (use_smote and HAVE_IMB):
        return Xtr, ytr
    n_pos = int(ytr.sum())
    if n_pos <= C.N_FOLDS:
        return Xtr, ytr
    try:
        return SMOTETomek(random_state=seed).fit_resample(Xtr, ytr)
    except Exception:  # noqa: BLE001 — too few minority neighbours
        return Xtr, ytr


# --------------------------------------------------------------------------
# Single-level out-of-fold probabilities.
# --------------------------------------------------------------------------
def oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    *,
    use_smote: bool = False,
    use_spw: bool = True,
    params: dict | None = None,
    n_folds: int | None = None,
    seed: int = C.RANDOM_STATE,
    feature_selector=None,
) -> np.ndarray:
    """Return honest OOF probabilities: each row scored by a model that never saw it.

    * use_smote  — fit SMOTE-Tomek inside each training fold.
    * use_spw    — pass scale_pos_weight = n_neg/n_pos (from the ORIGINAL fold ratio).
                   When SMOTE is on and use_spw is off, spw stays 1 (no double count).
    * feature_selector(Xtr, ytr) -> column index array; fit inside the fold only.
    """
    n_folds = n_folds or C.N_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)

    for tr, va in skf.split(X, y):
        Xtr, Xva, ytr = X[tr], X[va], y[tr]
        cols = None
        if feature_selector is not None:
            cols = feature_selector(Xtr, ytr)
            Xtr, Xva = Xtr[:, cols], Xva[:, cols]

        n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
        spw = (n_neg / max(n_pos, 1)) if use_spw else 1.0

        Xtr_r, ytr_r = _resample(Xtr, ytr, use_smote, seed)
        model = make_xgb(params, spw=spw, seed=seed)
        model.fit(Xtr_r, ytr_r)
        oof[va] = model.predict_proba(Xva)[:, 1]

    return oof


def repeated_oof(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    base_seed: int = C.RANDOM_STATE,
    **kwargs,
) -> list[np.ndarray]:
    """OOF probabilities over several seeds (different fold assignments)."""
    return [
        oof_probabilities(X, y, seed=base_seed + r, **kwargs)
        for r in range(n_repeats)
    ]


# --------------------------------------------------------------------------
# Thresholding + metrics.
# --------------------------------------------------------------------------
def pick_precision_threshold(y: np.ndarray, p: np.ndarray, target: float) -> tuple[float, bool]:
    """Threshold that holds precision >= target while maximising recall.

    Fallback (target unreachable): the best-F1 threshold — never a blind 0.5 that
    would predict all-negative at <1% prevalence. Returns (threshold, target_met).
    """
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, False
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) > 0:
        best = ok[np.argmax(rec[:-1][ok])]
        return float(thr[best]), True
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1s))]), False


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Precision/recall/F1/AUPRC/AUROC/FPR at a fixed threshold."""
    yhat = (p >= thr).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    tp = int(((yhat == 1) & (y == 1)).sum())
    return {
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "fpr": round(fp / (fp + tn + 1e-12), 4),
        "tp": tp,
        "fp": fp,
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free metrics + calibration error (isotonic-agnostic)."""
    return {
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "ece": round(float(expected_calibration_error(y, p)), 4),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Quantile-binned expected calibration error."""
    p = np.clip(p, 0, 1)
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        conf = p_s[b].mean()
        acc = y_s[b].mean()
        ece += (len(b) / len(p)) * abs(conf - acc)
    return ece


def calibrate(p_train: np.ndarray, y_train: np.ndarray, p_apply: np.ndarray) -> np.ndarray:
    """Fit isotonic on (p_train, y_train), apply to p_apply. Monotonic -> rank-safe."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    return iso.transform(p_apply)


# --------------------------------------------------------------------------
# Stratified bootstrap CIs — resample positives and negatives separately so
# prevalence is preserved and every resample has >=1 positive.
# --------------------------------------------------------------------------
def stratified_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 2000,
    seed: int = C.RANDOM_STATE,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for metric_fn(y_resampled, p_resampled) over stratified resamples."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            vals.append(float(metric_fn(y[idx], p[idx])))
        except Exception:  # noqa: BLE001 — degenerate resample
            continue
    vals = np.array(vals)
    return {
        "median": round(float(np.median(vals)), 4),
        "lo": round(float(np.quantile(vals, alpha / 2)), 4),
        "hi": round(float(np.quantile(vals, 1 - alpha / 2)), 4),
        "n_boot": int(len(vals)),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binomial proportion k/n (e.g. recall = TP/P)."""
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0}
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return {
        "point": round(phat, 4),
        "lo": round(max(0.0, centre - half), 4),
        "hi": round(min(1.0, centre + half), 4),
    }


@dataclass
class Data:
    """Loaded feature matrix + label, with column names, ready for experiments."""

    X: np.ndarray
    y: np.ndarray
    feat_names: list[str] = field(default_factory=list)


def load_features() -> Data:
    """Load the Stage-2 feature matrix (numeric only) as arrays + names."""
    from utils import load_frame

    df = load_frame(C.FEATURES_PARQUET)
    y = df[C.TARGET_COL].astype(int).values
    Xdf = df.drop(columns=[C.TARGET_COL]).select_dtypes(include=[np.number])
    return Data(
        X=Xdf.values.astype(np.float32),
        y=y,
        feat_names=Xdf.columns.tolist(),
    )
