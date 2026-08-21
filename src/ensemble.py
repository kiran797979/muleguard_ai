"""
The MuleGuard ensemble, and the threshold/metric helpers it depends on.

This lives in its own module rather than inside `03_train.py` for a concrete
reason: a class defined in a script that runs as `__main__` gets pickled with
module name `__main__`, and then fails to unpickle in any other process, where
`__main__` is a different file. Stage 7 loads the saved bundle, so the class must
have a stable importable home.

Everything here is fitted on training rows only. See `03_train.py` for the
cross-validation harness that uses it.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import config as C

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


# --------------------------------------------------------------------------
# Metrics and operating points
# --------------------------------------------------------------------------
def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Metrics at a FIXED threshold that was decided without seeing `y`."""
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prevalence = float(y.mean()) if len(y) else 0.0
    scorable = 0 < y.sum() < len(y)
    return {
        "threshold": round(float(thr), 6),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4) if scorable else float("nan"),
        "auroc": round(float(roc_auc_score(y, p)), 4) if scorable else float("nan"),
        "fpr": round(fp / max(fp + tn, 1), 4),
        # Lift over the base rate is what an AML desk actually acts on:
        # "our queue is N times richer in mules than a random sample".
        "lift_over_prevalence": round((tp / max(tp + fp, 1)) / prevalence, 2)
        if prevalence > 0 and (tp + fp) > 0 else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def pick_threshold(y: np.ndarray, p: np.ndarray, target: float) -> tuple[float, bool]:
    """Choose an operating threshold. Must only ever be called on INNER data.

    Preferred: the highest-recall threshold whose precision still clears the
    target. Fallback when the target is unreachable: the best-F1 point — never a
    blind 0.5, which on a calibrated sub-1% problem would predict all-negative
    and silently report zeros.
    """
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5, False
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, False
    ok = np.where(prec[:-1] >= target)[0]
    if len(ok) > 0:
        best = ok[int(np.argmax(rec[:-1][ok]))]
        return float(thr[best]), True
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1s))]), False


# --------------------------------------------------------------------------
# The ensemble
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
# Isotonic regression is the usual recommendation and it is the wrong choice at
# this sample size. Measured on the supplied data, fitting on ~40 positives and
# evaluating on the rest over 40 repeats:
#
#     calibrator     Brier      log loss           ECE   distinct p
#     none         0.00227        0.0102       0.00196        4,541
#     isotonic     0.00210        0.0126       0.00093           15
#     Platt        0.00199        0.0098       0.00099        4,380
#
# Two things are wrong with the isotonic row. It is *worse than no calibration
# at all* on log loss and carries the highest variance, which is what
# overfitting looks like. And it collapses the output to fifteen distinct
# values: isotonic is a step function, and with this few positives it has very
# few steps. Ranking inside a step is impossible, which directly damages
# Precision@K and band assignment — the two things this project reports.
#
# The real pipeline fits calibration on ~13 positives per inner fold, fewer
# than the measurement above, so the effect is understated there.
#
# Platt (a sigmoid, two parameters) cannot overfit the same way and preserves
# the ranking. We therefore choose by positive count rather than by preference,
# and record which was used so a report can state it.
MIN_POSITIVES_FOR_ISOTONIC = 100


def _fit_calibrator(p: np.ndarray, y: np.ndarray):
    """Return (fitted calibrator, method name), chosen by how many positives exist."""
    n_pos = int(np.sum(y))
    if n_pos >= MIN_POSITIVES_FOR_ISOTONIC:
        return IsotonicRegression(out_of_bounds="clip").fit(p, y), "isotonic"
    lg = _logit(p).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=1000).fit(lg, y), "platt"


def _apply_calibrator(cal, method: str, p: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return cal.predict(p)
    return cal.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


class MuleEnsemble:
    """Feature selection + 3 base models + stacking + calibration + threshold.

    Every component is fitted in `fit()` from the rows handed to it and nothing
    else. The inner cross-validation produces honest out-of-fold base
    predictions, and those — not training-set predictions, where a 400-tree
    booster is near-perfect — are what the meta-learner and the calibrator learn
    from.
    """

    def __init__(self, seed: int = C.RANDOM_STATE, top_k: int = C.TOP_K_FEATURES):
        self.seed = seed
        self.top_k = top_k
        self.sel_idx: np.ndarray | None = None
        self.medians: np.ndarray | None = None
        self.iso_lo = 0.0
        self.iso_hi = 1.0

    # -- imputation --------------------------------------------------------
    def _prep(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Replace non-finite cells with the TRAINING median of their column.

        Previously Stage 1 median-imputed the whole matrix before any split, so
        validation rows helped choose the value used to fill training rows and
        vice versa. That is a transductive leak: mild, but it contradicts the
        claim that every fitted component lives inside the fold. The medians are
        now learned here, on training rows only, and applied frozen to whatever
        is scored later — including a single live account, which has no
        distribution of its own to impute from.
        """
        X = np.asarray(X, dtype=np.float32)
        bad = ~np.isfinite(X)
        if fit:
            Xn = np.where(bad, np.nan, X)
            with np.errstate(all="ignore"):
                med = np.nanmedian(Xn, axis=0)
            self.medians = np.nan_to_num(med, nan=0.0, posinf=0.0,
                                         neginf=0.0).astype(np.float32)
        if self.medians is None:
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not bad.any():
            return X
        X = X.copy()
        X[bad] = np.take(self.medians, np.where(bad)[1])
        return X

    # -- feature selection -------------------------------------------------
    def _select(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Rank features by gradient-boosted gain, on training rows only.

        With ~1,600 candidate columns and 81 positives, handing the model
        everything invites it to fit noise. Refitting this ranker per fold is
        what keeps the choice of features from seeing validation rows.
        """
        if not HAVE_XGB or X.shape[1] <= self.top_k:
            return np.arange(X.shape[1])
        spw = (y == 0).sum() / max(y.sum(), 1)
        ranker = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.5, tree_method="hist", eval_metric="aucpr",
            scale_pos_weight=spw, random_state=self.seed, n_jobs=-1,
        ).fit(X, y)
        return np.argsort(ranker.feature_importances_)[::-1][:self.top_k]

    # -- base models -------------------------------------------------------
    def _fit_base(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Cost-sensitive base models.

        `scale_pos_weight` handles the 111:1 imbalance directly. We prefer it to
        SMOTE-style oversampling here: interpolating synthetic minority points
        across hundreds of dimensions from 65 training positives invents
        neighbourhoods that do not exist, and Tomek-link cleaning is O(n^2) on
        9k rows. Reweighting achieves the same goal without fabricating rows.
        """
        spw = (y == 0).sum() / max(y.sum(), 1)
        models: dict = {"iso": IsolationForest(**C.ISO_FOREST_PARAMS, n_jobs=-1).fit(X)}
        if HAVE_XGB:
            models["xgb"] = XGBClassifier(
                **C.XGB_PARAMS, scale_pos_weight=spw, n_jobs=-1).fit(X, y)
        if HAVE_LGBM:
            models["lgbm"] = LGBMClassifier(
                **C.LGBM_PARAMS, scale_pos_weight=spw, n_jobs=-1).fit(X, y)
        return models

    def _iso_score(self, model, X: np.ndarray, fit_range: bool = False) -> np.ndarray:
        """Anomaly score scaled to 0..1 using the TRAINING range.

        Scaling against the scored rows' own extremes would let the output depend
        on which rows happen to be scored together, so the range is learned once
        on training data and then frozen.
        """
        raw = -model.decision_function(X)
        if fit_range:
            self.iso_lo, self.iso_hi = float(raw.min()), float(raw.max())
        return np.clip((raw - self.iso_lo) / (self.iso_hi - self.iso_lo + 1e-12), 0, 1)

    def _base_matrix(self, models: dict, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([
            self._iso_score(models["iso"], X),
            models["xgb"].predict_proba(X)[:, 1] if "xgb" in models else np.zeros(n),
            models["lgbm"].predict_proba(X)[:, 1] if "lgbm" in models else np.zeros(n),
        ])

    # -- fit ---------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MuleEnsemble":
        X = self._prep(X, fit=True)
        self.sel_idx = self._select(X, y)
        Xs = X[:, self.sel_idx]

        # Inner CV -> honest base predictions for the stack and the calibrator.
        inner = StratifiedKFold(n_splits=C.INNER_FOLDS, shuffle=True,
                                random_state=self.seed)
        inner_oof = np.zeros((len(y), 3))
        for tr, va in inner.split(Xs, y):
            m = self._fit_base(Xs[tr], y[tr])
            self._iso_score(m["iso"], Xs[tr], fit_range=True)
            inner_oof[va] = self._base_matrix(m, Xs[va])

        self.meta = LogisticRegression(
            max_iter=2000, class_weight="balanced").fit(inner_oof, y)
        # Recorded so the reports can state what each base model contributed.
        # The isolation forest scores BELOW random on this data (AUROC ~0.31),
        # so its coefficient comes out negative: the stack learns to read it
        # upside-down. That is worth publishing, not hiding behind the word
        # "ensemble of three models".
        self.meta_coef = dict(zip(("iso", "xgb", "lgbm"),
                                  self.meta.coef_[0].round(4).tolist()))
        meta_p = self.meta.predict_proba(inner_oof)[:, 1]

        self.calibrator, self.calibration_method = _fit_calibrator(meta_p, y)
        cal_p = _apply_calibrator(self.calibrator, self.calibration_method, meta_p)

        self.thr_precision, self.target_met = pick_threshold(
            y, cal_p, C.PRECISION_TARGET)
        self.thr_recall, _ = pick_threshold(y, cal_p, C.RECALL_QUEUE_PRECISION)

        # Refit base models on ALL training rows for actual scoring.
        self.models = self._fit_base(Xs, y)
        self._iso_score(self.models["iso"], Xs, fit_range=True)
        return self

    # -- predict -----------------------------------------------------------
    def base_probs(self, X: np.ndarray) -> np.ndarray:
        return self._base_matrix(self.models, self._prep(X)[:, self.sel_idx])

    def selected_matrix(self, X: np.ndarray) -> np.ndarray:
        """Imputed, selection-restricted matrix — what the tree models see.

        Stage 7/8 needs exactly this to compute SHAP against the same input the
        model scored, rather than re-deriving it and hoping the two agree.
        """
        return self._prep(X)[:, self.sel_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = self.base_probs(X)
        meta_p = self.meta.predict_proba(B)[:, 1]
        # Must go through the same helper the fit path used. Calling
        # `.predict()` directly works for isotonic and returns *class labels*
        # for a Platt sigmoid, which would silently turn every score into 0 or 1.
        return _apply_calibrator(self.calibrator, self.calibration_method, meta_p)
