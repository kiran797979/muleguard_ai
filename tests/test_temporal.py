"""Tests for temporal localisation and the temporal IoU metric.

The properties worth pinning are the ones that decide whether the submission's
window columns are worth anything: that the metric is right, that a planted
episode is recovered rather than the whole history, and that the module refuses
to invent a date when there are no transactions to read one from.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import temporal as T  # noqa: E402

START = pd.Timestamp("2023-01-01")


def _txn(acct, day_offset, amount, direction):
    return {"account_id": acct,
            "timestamp": START + pd.Timedelta(days=day_offset),
            "amount": amount, "direction": direction}


# ==========================================================================
# The metric
# ==========================================================================
def test_iou_of_identical_windows_is_one():
    a, b = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-01")
    assert T.temporal_iou(a, b, a, b) == pytest.approx(1.0)


def test_iou_of_disjoint_windows_is_zero():
    assert T.temporal_iou("2023-01-01", "2023-01-31",
                          "2023-06-01", "2023-06-30") == 0.0


def test_iou_of_half_overlap():
    # pred covers days 1-20, truth 11-30; intersect 10 days, union 30.
    iou = T.temporal_iou("2023-01-01", "2023-01-20", "2023-01-11", "2023-01-30")
    assert iou == pytest.approx(10 / 30, abs=0.02)


def test_iou_rewards_predicting_nothing_when_there_is_nothing():
    assert T.temporal_iou(None, None, None, None) == 1.0


def test_iou_punishes_a_window_where_there_is_none():
    assert T.temporal_iou("2023-01-01", "2023-02-01", None, None) == 0.0
    assert T.temporal_iou(None, None, "2023-01-01", "2023-02-01") == 0.0


def test_iou_is_symmetric_to_reversed_bounds():
    """A caller handing the interval end-first must not silently score zero."""
    fwd = T.temporal_iou("2023-01-01", "2023-01-20", "2023-01-11", "2023-01-30")
    rev = T.temporal_iou("2023-01-20", "2023-01-01", "2023-01-30", "2023-01-11")
    assert fwd == pytest.approx(rev)


# ==========================================================================
# Otsu baseline
# ==========================================================================
def test_otsu_separates_a_bimodal_series():
    quiet = np.full(90, 0.05)
    hot = np.full(10, 0.80)
    thr, sep = T.otsu_threshold(np.concatenate([quiet, hot]))
    assert 0.05 < thr < 0.80
    assert sep > 0.01


def test_otsu_separation_is_near_zero_for_a_flat_series():
    """A history with no episode in it must not look separable."""
    _, sep = T.otsu_threshold(np.full(200, 0.12))
    assert sep < 1e-6


def test_median_baseline_swallows_the_history_and_otsu_does_not():
    """The bug this module was rewritten around, pinned so it cannot return.

    Kadane maximises a sum, so with a median baseline roughly half the days
    carry positive excess and the window grows to cover everything.
    """
    # The quiet days must be right-skewed, which is what a real activity series
    # looks like: mostly flat with occasional ordinary spikes. That puts the
    # median below the mean, so `score - median` is positive on average and
    # Kadane keeps extending. A uniform quiet period gives exactly zero excess
    # and hides the bug.
    quiet = np.where(np.arange(180) % 5 == 0, 0.25, 0.05)
    scores = np.concatenate([quiet, np.full(20, 0.85), quiet])
    days = pd.Series(pd.date_range(START, periods=len(scores), freq="D"))

    s_med, e_med, _ = T.best_window(scores, days, baseline=float(np.median(scores)))
    s_ots, e_ots, _ = T.best_window(scores, days)          # Otsu by default

    span_med = (e_med - s_med).days + 1
    span_ots = (e_ots - s_ots).days + 1
    assert span_med > 300, "median baseline should over-extend; fixture is wrong"
    assert span_ots <= 30, f"Otsu window should be tight, got {span_ots} days"


# ==========================================================================
# Window extraction
# ==========================================================================
def test_kadane_absorbs_a_dip_inside_one_episode():
    """A quiet day mid-episode must not split the window into two."""
    scores = np.array([0.05] * 40 + [0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9] + [0.05] * 40)
    days = pd.Series(pd.date_range(START, periods=len(scores), freq="D"))
    s, e, _ = T.best_window(scores, days)
    assert (e - s).days + 1 >= 7, "the dip fragmented the episode"


def test_window_is_widened_to_a_minimum_span():
    scores = np.array([0.02] * 50 + [0.99] + [0.02] * 50)
    days = pd.Series(pd.date_range(START, periods=len(scores), freq="D"))
    s, e, _ = T.best_window(scores, days)
    assert (e - s).days + 1 >= T.MIN_WINDOW_DAYS


def test_empty_series_yields_no_window():
    s, e, strength = T.best_window(np.array([]), pd.Series([], dtype="datetime64[ns]"))
    assert s is None and e is None and strength == 0.0


# ==========================================================================
# Schema discipline
# ==========================================================================
def test_missing_columns_raise_rather_than_guess():
    bad = pd.DataFrame({"acct": ["A"], "when": [START], "value": [1.0]})
    with pytest.raises(KeyError):
        T.daily_frame(bad)


def test_reversals_are_measured_by_magnitude():
    """A negative amount is a reversal, not a negative day of activity."""
    txns = pd.DataFrame([_txn("A", 0, -5000.0, "D"), _txn("A", 0, 5000.0, "C")])
    d = T.daily_frame(txns)
    assert d["debit"].iloc[0] == pytest.approx(5000.0)


def test_structuring_just_under_a_threshold_is_flagged():
    txns = pd.DataFrame([_txn("A", 0, 48_000.0, "C"), _txn("A", 0, 12_000.0, "C")])
    d = T.daily_frame(txns)
    assert d["n_near_threshold"].iloc[0] == 1


# ==========================================================================
# End to end
# ==========================================================================
def _account_with_episode(acct, quiet_days, episode_days):
    rows = []
    for d in range(quiet_days):
        rows.append(_txn(acct, d, 40_000.0, "C"))
        rows.append(_txn(acct, d, 900.0, "D"))
    for d in range(quiet_days, quiet_days + episode_days):
        for _ in range(4):
            rows.append(_txn(acct, d, 48_500.0, "C"))
            rows.append(_txn(acct, d, 48_000.0, "D"))
    for d in range(quiet_days + episode_days, quiet_days + episode_days + quiet_days):
        rows.append(_txn(acct, d, 40_000.0, "C"))
        rows.append(_txn(acct, d, 900.0, "D"))
    return rows


def test_planted_episode_is_recovered_with_high_iou():
    quiet, episode = 120, 40
    txns = pd.DataFrame(_account_with_episode("A", quiet, episode))
    w = T.detect_windows(txns)
    row = w.iloc[0]
    iou = T.temporal_iou(row.suspicious_start, row.suspicious_end,
                         START + pd.Timedelta(days=quiet),
                         START + pd.Timedelta(days=quiet + episode - 1))
    assert iou > 0.6, f"IoU {iou:.3f} — window {row.suspicious_start}..{row.suspicious_end}"


def test_window_confidence_is_higher_for_an_episode_than_a_flat_history():
    with_ep = pd.DataFrame(_account_with_episode("A", 100, 30))
    flat = pd.DataFrame([r for d in range(230)
                         for r in (_txn("B", d, 40_000.0, "C"), _txn("B", d, 900.0, "D"))])
    a = T.detect_windows(with_ep).iloc[0]["window_confidence"]
    b = T.detect_windows(flat).iloc[0]["window_confidence"]
    assert a > b


def test_low_risk_accounts_get_no_window():
    txns = pd.DataFrame(_account_with_episode("A", 60, 20))
    w = T.detect_windows(txns, risk={"A": 0.01}, min_risk=0.5)
    assert pd.isna(w.iloc[0]["suspicious_start"])


# ==========================================================================
# Submission format
# ==========================================================================
def test_submission_leaves_the_window_empty_for_predicted_legitimate():
    risk = pd.DataFrame({"account_id": ["A", "B"], "is_mule": [0.91, 0.03]})
    windows = pd.DataFrame({
        "account_id": ["A", "B"],
        "suspicious_start": [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-01")],
        "suspicious_end": [pd.Timestamp("2023-04-01"), pd.Timestamp("2023-06-01")],
    })
    sub = T.to_submission(risk, windows, threshold=0.5)
    a = sub[sub.account_id == "A"].iloc[0]
    b = sub[sub.account_id == "B"].iloc[0]
    assert a.suspicious_start == "2023-03-01T00:00:00"
    assert b.suspicious_start == "" and b.suspicious_end == ""


def test_submission_has_exactly_the_required_columns_in_order():
    risk = pd.DataFrame({"account_id": ["A"], "is_mule": [0.7]})
    windows = pd.DataFrame({"account_id": ["A"],
                            "suspicious_start": [pd.Timestamp("2023-03-01")],
                            "suspicious_end": [pd.Timestamp("2023-04-01")]})
    sub = T.to_submission(risk, windows)
    assert list(sub.columns) == ["account_id", "is_mule",
                                 "suspicious_start", "suspicious_end"]


def test_score_windows_aggregates_iou():
    pred = pd.DataFrame({"account_id": ["A", "B"],
                         "suspicious_start": [pd.Timestamp("2023-01-01"), None],
                         "suspicious_end": [pd.Timestamp("2023-01-31"), None]})
    truth = pd.DataFrame({"account_id": ["A", "B"],
                          "suspicious_start": [pd.Timestamp("2023-01-01"), None],
                          "suspicious_end": [pd.Timestamp("2023-01-31"), None]})
    out = T.score_windows(pred, truth)
    assert out["n_scored"] == 2
    assert out["mean_temporal_iou"] == pytest.approx(1.0)
