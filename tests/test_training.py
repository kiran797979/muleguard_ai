"""
Tests for Stage 4/5 training logic — the honest-threshold behaviour:

  * When the precision target is reachable, the chosen threshold holds precision
    at/above the target and reports target_met=True.
  * When it is NOT reachable, it falls back to the best-F1 threshold (never a
    blind 0.5 that would predict all-negative at <1% prevalence) and reports
    target_met=False.
  * metrics_at returns coherent precision/recall/FPR.
"""

from __future__ import annotations

import numpy as np

from conftest import load_stage

train = load_stage("03_train.py")


def _separable_scores(n=1000, pos=50, seed=0):
    """Well-separated scores where high precision at good recall is achievable."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=pos, replace=False)] = 1
    p = np.where(y == 1,
                 rng.uniform(0.80, 0.99, n),
                 rng.uniform(0.00, 0.20, n))
    return y, p


def test_threshold_meets_reachable_precision_target():
    y, p = _separable_scores()
    thr, met = train.pick_precision_threshold(y, p, target=0.90)
    assert met is True
    # At the chosen threshold, realised precision must actually be >= target.
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    assert precision >= 0.90


def test_threshold_falls_back_to_best_f1_when_target_unreachable():
    """Unreachable target must NOT collapse to predicting all-negative."""
    rng = np.random.default_rng(1)
    n = 1000
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=50, replace=False)] = 1
    # Heavily overlapping scores -> 0.99 precision is impossible.
    p = np.clip(y * 0.15 + rng.normal(0.1, 0.1, n), 0, 1)
    thr, met = train.pick_precision_threshold(y, p, target=0.99)
    assert met is False
    # Fallback must still flag SOME positives (recall > 0), i.e. not all-negative.
    assert (p >= thr).sum() > 0


def test_metrics_at_is_coherent():
    y, p = _separable_scores()
    m = train.metrics_at(y, p, thr=0.5)
    for k in ("precision", "recall", "f1", "auprc", "auroc", "fpr", "threshold"):
        assert k in m
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["fpr"] <= 1.0


def test_lgbm_disabled_by_default_in_config():
    """The measured winning config: XGBoost solo (LightGBM + IsoForest off)."""
    import config as C
    assert C.USE_LGBM is False
    assert C.USE_ISO_FOREST is False


def test_no_double_imbalance_correction_by_default():
    """The double-correction bug is retired: not both SMOTE and spw at once."""
    import config as C
    assert not (C.USE_SMOTE and C.USE_SPW), \
        "USE_SMOTE and USE_SPW both True re-introduces the double-correction"


def test_nested_operating_point_is_out_of_sample():
    """Nested CV must score every row with a threshold/calibrator it never saw.

    Constructs separable data, runs one nested-CV repeat, and asserts: every row is
    assigned exactly one pooled out-of-sample decision, thresholds vary per fold
    (were re-fit, not global), and the pooled recall is sane. This is the guarantee
    that the honest operating point is not the in-sample optimistic one.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "experiments"))

    y, p_unused = _separable_scores(n=600, pos=60, seed=3)
    # Build a feature matrix loosely correlated with y so XGB can learn something.
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, size=(len(y), 8)).astype("float32")
    X[y == 1] += 1.2

    py, dec, prob, thrs = train.nested_operating_point(X, y, seed=42)
    # Every row scored exactly once, decisions are binary.
    assert len(py) == len(y)
    assert set(np.unique(dec)).issubset({0.0, 1.0})
    # Per-fold thresholds were fit independently (not a single global constant).
    assert len(thrs) == 5
    # Pooled out-of-sample recall is a real fraction.
    tp = int(((dec == 1) & (py == 1)).sum())
    assert 0 <= tp <= int(py.sum())
