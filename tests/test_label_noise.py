"""Tests for label-noise detection and the row-order integrity test.

The property that matters most here is restraint: this module must not flag
every borderline account as a bad label, because a review queue nobody can
finish is the same as no review queue. So the tests check both that it finds a
planted flipped label and that it stays quiet on clean data.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import label_noise as LN  # noqa: E402
import schema as S  # noqa: E402


def _clean(n_neg=500, n_pos=25, seed=0):
    """A well-separated problem with no label errors."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]
    p = np.r_[rng.uniform(0.0, 0.05, n_neg), rng.uniform(0.85, 0.99, n_pos)]
    return y, p


# ==========================================================================
# Restraint
# ==========================================================================
def test_clean_labels_produce_no_candidates():
    y, p = _clean()
    assert LN.suspect_labels(y, p).empty


def test_clean_labels_estimate_near_zero_noise():
    y, p = _clean()
    e = LN.estimate_noise(y, p)
    assert e["labelled_mule_flagged"] == 0
    assert e["labelled_legitimate_flagged"] == 0


def test_borderline_accounts_are_not_flagged():
    """Disagreement alone is not evidence; it must clear the other threshold."""
    y, p = _clean()
    p[10] = 0.30          # a negative the model is unsure about
    p[-1] = 0.60          # a positive the model is unsure about
    sus = LN.suspect_labels(y, p)
    flagged = set(sus["account_idx"]) if not sus.empty else set()
    assert 10 not in flagged and (len(y) - 1) not in flagged


# ==========================================================================
# Detection
# ==========================================================================
def test_flipped_positive_is_caught():
    """A 'mule' that behaves exactly like a normal customer."""
    y, p = _clean()
    y = y.copy()
    p = p.copy()
    p[-1] = 0.001                       # labelled mule, scored as clean as it gets
    sus = LN.suspect_labels(y, p)
    assert not sus.empty
    top = sus.iloc[0]
    assert top["account_idx"] == len(y) - 1
    assert top["direction"] == "LABELLED_MULE_SCORED_LOW"


def test_flipped_negative_is_caught():
    y, p = _clean()
    p = p.copy()
    p[0] = 0.99                         # labelled legitimate, scored as a mule
    sus = LN.suspect_labels(y, p)
    assert not sus.empty
    assert sus.iloc[0]["account_idx"] == 0
    assert sus.iloc[0]["direction"] == "LABELLED_LEGITIMATE_SCORED_HIGH"


def test_candidates_are_ranked_by_margin():
    y, p = _clean()
    p = p.copy()
    p[-1] = 0.001
    p[-2] = 0.010
    sus = LN.suspect_labels(y, p)
    assert list(sus["margin"]) == sorted(sus["margin"], reverse=True)


def test_every_candidate_carries_a_reading():
    y, p = _clean()
    p = p.copy()
    p[-1] = 0.001
    for row in LN.suspect_labels(y, p).to_dict("records"):
        assert row["reading"], "a candidate with no explanation is not actionable"


# ==========================================================================
# The rank check is calibration-free
# ==========================================================================
def test_rank_check_survives_a_monotone_rescaling():
    """Squashing every probability must not change which labels look wrong."""
    y, p = _clean()
    p = p.copy()
    p[0] = 0.99
    a = LN.rank_anomalies(y, p)
    b = LN.rank_anomalies(y, p ** 3)        # monotone, ranks preserved
    assert set(a["account_idx"]) == set(b["account_idx"])


def test_audit_reports_what_it_is_not():
    y, p = _clean()
    res = LN.audit(y, p)
    assert "does not assert" in res["what_this_is_not"]
    assert res["uses_out_of_fold_scores"] is True


# ==========================================================================
# Row order
# ==========================================================================
def test_sorted_file_is_detected():
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    r = S.row_order_leak(y)
    assert r["position_auroc"] == pytest.approx(1.0)
    assert r["sorted_by_label"] is True
    assert r["positives_are_contiguous"] is True


def test_reverse_sorted_file_is_also_detected():
    """Positives first is the same finding, and must not score as clean."""
    y = np.r_[np.ones(100, int), np.zeros(900, int)]
    r = S.row_order_leak(y)
    assert r["position_auroc"] == pytest.approx(0.0)
    assert r["sorted_by_label"] is True


def test_shuffled_file_is_clean():
    rng = np.random.default_rng(0)
    y = rng.permutation(np.r_[np.zeros(900, int), np.ones(100, int)])
    r = S.row_order_leak(y)
    assert r["sorted_by_label"] is False
    assert r["separation"] < 0.2


def test_interleaved_file_is_clean():
    y = np.tile([0, 0, 0, 0, 1], 200)
    r = S.row_order_leak(y)
    assert r["sorted_by_label"] is False


def test_single_class_is_not_applicable():
    r = S.row_order_leak(np.zeros(100, int))
    assert r["applicable"] is False


def test_position_auroc_matches_sklearn():
    """The rank-sum shortcut must agree with the standard implementation."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    y = rng.permutation(np.r_[np.zeros(400, int), np.ones(60, int)])
    ours = S.row_order_leak(y)["position_auroc"]
    theirs = roc_auc_score(y, np.arange(len(y)))
    # row_order_leak rounds to 6 dp on purpose, so the tolerance matches
    # that rather than float precision.
    assert ours == pytest.approx(theirs, abs=1e-6)
