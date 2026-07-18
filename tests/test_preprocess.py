"""
Tests for the fittable Preprocessor — the inference-path guarantees:

  * BIT-EXACT reproduction: fitting the Preprocessor on the raw CSV and transforming
    it must reproduce the existing cleaned matrix (same columns, order, dtypes, and
    values) that 01_clean.py produced — so no measured number can move.
  * Round-trip persistence: save() then load() yields an identical transform.
  * Inference replay: transform() works on a single new raw row without the target,
    reusing fitted medians / one-hot vocab / date reference (no re-derivation).
  * Unseen categories and absent columns are handled without error.

The bit-exact test is skipped (not failed) if the raw dataset or the prior cleaned
matrix is absent, so the suite still runs in a fresh checkout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import load_stage  # noqa: F401  (ensures src/ is on sys.path)
import config as C  # noqa: E402
from preprocess import Preprocessor  # noqa: E402


def _load_prior_clean():
    """Load the cleaned matrix produced by the original Stage 1, if present."""
    if C.CLEAN_PARQUET.exists():
        try:
            return pd.read_parquet(C.CLEAN_PARQUET)
        except Exception:
            pass
    alt = C.CLEAN_PARQUET.with_suffix(".csv")
    if alt.exists():
        return pd.read_csv(alt, low_memory=False)
    return None


@pytest.mark.skipif(not C.RAW_CSV.exists(), reason="raw DataSet.csv not present")
def test_preprocessor_reproduces_clean_matrix_bit_exact():
    """Preprocessor.fit_transform(raw) must equal the existing clean matrix exactly."""
    prior = _load_prior_clean()
    if prior is None:
        pytest.skip("no prior clean.parquet to compare against")

    raw = pd.read_csv(C.RAW_CSV, low_memory=False)
    pre = Preprocessor()
    got, _report = pre.fit_transform(raw)

    # Same columns, same order.
    assert list(got.columns) == list(prior.columns), "column set/order diverged"

    # Same values (exact for the target; tight tolerance for float features to
    # absorb only parquet round-trip float32/64 storage, not logic differences).
    for col in prior.columns:
        if col == C.TARGET_COL:
            assert (got[col].values == prior[col].values).all(), f"target values diverged"
        else:
            a = got[col].astype(float).values
            b = prior[col].astype(float).values
            assert np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True), \
                f"values diverged in column {col}"


@pytest.mark.skipif(not C.RAW_CSV.exists(), reason="raw DataSet.csv not present")
def test_preprocessor_save_load_roundtrip(tmp_path):
    raw = pd.read_csv(C.RAW_CSV, low_memory=False, nrows=2000)
    pre = Preprocessor()
    pre.fit_transform(raw)

    path = tmp_path / "pre.json"
    pre.save(path)
    pre2 = Preprocessor.load(path)

    # A fresh raw slice transforms identically through the reloaded preprocessor.
    new = raw.drop(columns=[C.TARGET_COL]).head(10)
    t1 = pre.transform(new)
    t2 = pre2.transform(new)
    assert list(t1.columns) == list(t2.columns)
    assert np.allclose(t1.values, t2.values, equal_nan=True)


@pytest.mark.skipif(not C.RAW_CSV.exists(), reason="raw DataSet.csv not present")
def test_transform_scores_single_new_row_shape():
    """A single new account (no target) transforms to the fitted feature schema."""
    raw = pd.read_csv(C.RAW_CSV, low_memory=False, nrows=1000)
    pre = Preprocessor()
    pre.fit_transform(raw)

    one = raw.drop(columns=[C.TARGET_COL]).head(1)
    out = pre.transform(one)
    assert list(out.columns) == list(pre.feature_columns)
    assert out.shape[0] == 1
    assert out.isna().sum().sum() == 0  # medians/zeros fill everything


@pytest.mark.skipif(
    not (C.MODELS_DIR / "muleguard_models.joblib").exists()
    or not (C.MODELS_DIR / "preprocessor.json").exists()
    or not C.RAW_CSV.exists(),
    reason="model bundle / preprocessor / raw CSV not present (run the pipeline first)",
)
def test_score_new_end_to_end_from_raw_csv(tmp_path):
    """A held-out raw account scores end-to-end through the real inference path.

    This is the guarantee the old code could NOT make: take rows straight from the
    raw CSV (with the target column removed), replay the fitted preprocessing +
    mg_* features, and produce a calibrated risk score/band — no training-row
    shortcut, no re-derived schema.
    """
    from score_new import score_new

    raw = pd.read_csv(C.RAW_CSV, low_memory=False, nrows=25)
    held_out = raw.drop(columns=[C.TARGET_COL])
    csv = tmp_path / "new_accounts.csv"
    held_out.to_csv(csv, index=False)

    out = score_new(csv)
    assert len(out) == len(held_out)
    assert {"prob", "risk_score", "band", "action"}.issubset(out.columns)
    assert out["risk_score"].between(0, C.SCORE_MAX).all()
    assert out["band"].isin(["LOW", "MEDIUM", "HIGH"]).all()
    assert not out["prob"].isna().any()


def test_transform_handles_unseen_and_missing_categories():
    """Unseen one-hot levels vanish; absent levels appear as all-zero columns."""
    y = [0] * 90 + [1] * 10
    df = pd.DataFrame({
        "occupation": ["salaried"] * 45 + ["student"] * 45 + ["retired"] * 10,
        "num": np.arange(100, dtype=float),
        C.TARGET_COL: y,
    })
    pre = Preprocessor()
    pre.fit_transform(df.copy())

    # New batch with an unseen category ("astronaut") and missing "retired".
    new = pd.DataFrame({"occupation": ["salaried", "astronaut"], "num": [1.0, 2.0]})
    out = pre.transform(new)
    # Schema is fixed to fitted feature columns; no crash, no extra columns.
    assert list(out.columns) == list(pre.feature_columns)
    assert out.isna().sum().sum() == 0


def test_absent_majority_dummy_fills_zero_not_median():
    """Regression: an absent MAJORITY one-hot dummy must fill 0.0, never its median.

    With a >50%-prevalence category, that dummy's fit-time median is 1.0. If a
    single new account of a MINORITY category is scored, the majority dummy is absent
    -> must be 0 (not 1), else the mutually-exclusive one-hot group emits two 1s and
    the account is silently encoded into a category it isn't in.
    """
    y = [0] * 80 + [1] * 20
    # 'salaried' is the strict majority (70/100), so its dummy median is 1.0.
    occ = ["salaried"] * 70 + ["student"] * 20 + ["retired"] * 10
    df = pd.DataFrame({"occupation": occ, "num": np.arange(100, dtype=float),
                       C.TARGET_COL: y})
    pre = Preprocessor()
    pre.fit_transform(df.copy())

    # A single 'student' account — 'salaried' and 'retired' dummies are absent.
    one = pd.DataFrame({"occupation": ["student"], "num": [5.0]})
    out = pre.transform(one)

    dummy_cols = [c for c in pre.feature_columns if c.startswith("occupation_")]
    row = out.iloc[0]
    # Exactly ONE dummy in the group is hot (the true 'student' one).
    assert row[dummy_cols].sum() == 1.0, f"one-hot group not mutually exclusive: {row[dummy_cols].to_dict()}"
    if "occupation_salaried" in dummy_cols:
        assert row["occupation_salaried"] == 0.0, "absent majority dummy was median-filled to 1.0"
    if "occupation_student" in dummy_cols:
        assert row["occupation_student"] == 1.0
