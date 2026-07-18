"""
Tests for Stage 1 cleaning — the honesty guarantees that must never regress:

  * The categorical leak guard drops a column that (almost) perfectly determines
    the label — this is what removes the real F2230 sampling-month leak.
  * A genuine low-cardinality categorical is one-hot encoded, not dropped.
  * A date column becomes a numeric "vintage" feature.
  * Identifier columns (R's "Unnamed: 0") are removed.
  * The correlation leak scan flags a near-perfectly-correlated numeric column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import load_stage

clean = load_stage("01_clean.py")
import config as C  # noqa: E402


# --------------------------------------------------------------------------
# Categorical leak guard (the F2230 defence)
# --------------------------------------------------------------------------
def test_perfect_categorical_leak_is_detected():
    """A category that perfectly separates the classes is flagged as a leak."""
    y = pd.Series([0] * 90 + [1] * 10)
    # 'month': every normal is 'Oct', every mule is a different month — the F2230 pattern.
    leaky = pd.Series(["Oct"] * 90 + ["Nov"] * 5 + ["Dec"] * 5)
    assert clean._is_categorical_leak(leaky, y) is True


def test_benign_categorical_is_not_a_leak():
    """A category evenly spread across classes is NOT flagged."""
    y = pd.Series([0] * 90 + [1] * 10)
    rng = np.random.default_rng(0)
    benign = pd.Series(rng.choice(["A", "B", "C"], size=100))
    assert clean._is_categorical_leak(benign, y) is False


def test_encode_drops_leak_column_before_onehot():
    """encode_categoricals removes a perfect-separation column entirely."""
    y = [0] * 90 + [1] * 10
    df = pd.DataFrame({
        "month": ["Oct"] * 90 + ["Nov"] * 10,   # perfect leak
        C.TARGET_COL: y,
    })
    out, info = clean.encode_categoricals(df.copy(), protect=[C.TARGET_COL])
    # No column derived from 'month' should survive.
    assert not any(col.startswith("month") for col in out.columns)
    assert "month" in [d["col"] for d in info["dropped_leak"]]


# --------------------------------------------------------------------------
# Legitimate categorical recovery
# --------------------------------------------------------------------------
def test_low_cardinality_categorical_is_one_hot_encoded():
    y = ([0, 1] * 50)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "occupation": rng.choice(["salaried", "student", "retired"], size=100),
        C.TARGET_COL: y,
    })
    out, info = clean.encode_categoricals(df.copy(), protect=[C.TARGET_COL])
    assert "occupation" in info["one_hot"]
    # Dummy columns exist and the raw string column is gone.
    assert any(c.startswith("occupation_") for c in out.columns)
    assert "occupation" not in out.columns


def test_date_column_becomes_numeric_vintage():
    y = ([0, 1] * 50)
    dates = pd.date_range("2010-01-01", periods=100, freq="D").strftime("%m-%d-%Y")
    df = pd.DataFrame({"opened": dates, C.TARGET_COL: y})
    out, info = clean.encode_categoricals(df.copy(), protect=[C.TARGET_COL])
    assert "opened" in info["date_vintage"]
    assert pd.api.types.is_numeric_dtype(out["opened"])
    assert (out["opened"] >= 0).all()   # days-before-latest is non-negative


# --------------------------------------------------------------------------
# ID column removal & correlation leak scan
# --------------------------------------------------------------------------
def test_id_column_prefix_is_configured():
    """The Unnamed: prefix must remain in the drop list (the R row-id defence)."""
    assert any("Unnamed" in p for p in C.ID_COL_PREFIXES)


def test_correlation_scan_flags_near_perfect_feature():
    y = pd.Series([0] * 80 + [1] * 20)
    df = pd.DataFrame({
        "leak": y.astype(float) + np.random.default_rng(2).normal(0, 0.01, 100),
        "noise": np.random.default_rng(3).normal(0, 1, 100),
        C.TARGET_COL: y,
    })
    leaks, top = clean.detect_leaks(df, C.TARGET_COL)
    assert "leak" in leaks
    assert "noise" not in leaks
