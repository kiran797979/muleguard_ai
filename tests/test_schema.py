"""
Tests for the dataset adaptation layer.

These are the regression tests for "it works on any dataset". Each one plants a
single trap and asserts it is found from structure alone, so a failure says
exactly which inference broke.

Runs with pytest, or standalone with no test dependency at all:

    python tests/test_schema.py
    pytest tests/test_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import schema as S  # noqa: E402

RNG = np.random.default_rng(0)


def _frame(n=400, prevalence=0.05, **extra) -> pd.DataFrame:
    y = np.zeros(n, dtype=int)
    y[RNG.choice(n, size=max(int(n * prevalence), 10), replace=False)] = 1
    d = {f"metric_{i}": RNG.normal(0, 1, n) for i in range(8)}
    d.update(extra)
    df = pd.DataFrame(d)
    df["is_mule"] = y
    return df


# ---------------------------------------------------------------- normalise
def test_norm_ignores_punctuation_and_case():
    keys = {S.norm(x) for x in
            ["TOT_TXNAMT_CR_L7D", "tot.txnamt.cr.l7d", "Tot Txnamt Cr L7D",
             "tot-txnamt-cr-l7d"]}
    assert keys == {"TOTTXNAMTCRL7D"}, keys


# ------------------------------------------------------------------- target
def test_target_found_by_name_not_position():
    df = _frame()
    df["trailing_column"] = RNG.normal(0, 1, len(df))   # target is NOT last
    col, how = S.resolve_target(df)
    assert col == "is_mule", (col, how)


def test_configured_target_wins_when_present():
    df = _frame()
    df["F3924"] = df["is_mule"]
    col, _ = S.resolve_target(df, "F3924")
    assert col == "F3924"


def test_target_falls_back_to_only_binary_column():
    df = _frame()
    df = df.rename(columns={"is_mule": "outcome_flag"})
    col, how = S.resolve_target(df)
    assert col == "outcome_flag", (col, how)


def test_no_binary_column_raises_rather_than_guessing():
    df = pd.DataFrame({"a": RNG.normal(0, 1, 50), "b": RNG.normal(0, 1, 50)})
    try:
        S.resolve_target(df)
    except KeyError as exc:
        assert "MULEGUARD_TARGET" in str(exc)
    else:
        raise AssertionError("guessed a target instead of stopping")


# --------------------------------------------------------------- identifiers
def test_identifier_found_but_floats_are_never_identifiers():
    """The bug this guards: every continuous column is ~100% unique.

    Applying the near-unique rule to floats classifies the whole feature matrix
    as identifiers and deletes the dataset.
    """
    df = _frame(n=400)
    df["account_number"] = [f"AC{i:06d}" for i in range(len(df))]
    ids = S.identifier_columns(df, "is_mule")
    assert ids == ["account_number"], ids


def test_identifier_rule_backs_off_when_it_would_eat_the_dataset():
    n = 200
    df = pd.DataFrame({f"id_like_{i}": np.arange(n) for i in range(20)})
    df["is_mule"] = 0
    df.loc[:9, "is_mule"] = 1
    ids = S.identifier_columns(df, "is_mule")
    # 20 near-unique int columns is more than the safety valve allows; nothing
    # matches by name, so nothing is dropped.
    assert ids == [], ids


# --------------------------------------------------------------- categorical
def test_categorical_detection_and_ordinal_vocabulary():
    df = _frame()
    df["segment"] = RNG.choice(["retail", "sme", "corporate"], len(df))
    df["account_age"] = RNG.choice(["L7D", "L31D", "L365D", "G365D"], len(df))
    cats = S.categorical_columns(df, "is_mule")
    assert set(cats) == {"segment", "account_age"}, cats
    assert S.ordinal_mapping(df["account_age"].unique()) is not None
    assert S.ordinal_mapping(df["segment"].unique()) is None


# ---------------------------------------------------------------- partitions
def test_partition_column_detected_from_shape():
    """The generalisation of MNTH: classes in disjoint value sets."""
    df = _frame(n=600, prevalence=0.05)
    month = np.array(["2025-10"] * len(df), dtype=object)
    month[df["is_mule"] == 1] = "2025-09"
    df["data_month"] = month
    found = S.partition_columns(df, "is_mule")
    assert [f["column"] for f in found] == ["data_month"], found
    assert found[0]["purity"] == 1.0
    assert found[0]["values_containing_both_classes"] == 0


def test_genuine_categorical_is_not_flagged_as_a_partition():
    """The false-positive guard — occupation must survive."""
    df = _frame(n=600, prevalence=0.1)
    df["occupation"] = RNG.choice(["student", "salaried", "retired"], len(df))
    found = S.partition_columns(df, "is_mule")
    assert [f["column"] for f in found] == [], found


def test_high_cardinality_id_is_not_mistaken_for_a_partition():
    df = _frame(n=600)
    df["account_number"] = [f"AC{i}" for i in range(len(df))]
    found = S.partition_columns(df, "is_mule")
    assert "account_number" not in [f["column"] for f in found]


# --------------------------------------------------------------------- leaks
def test_post_outcome_and_structural_patterns():
    cols = ["case_resolution", "min_resolve_days", "fraud_suspected",
            "false_positive", "data_month", "snapshot_id",
            "tot_txnamt_cr_l7d", "avg_bal_7days"]
    post = set(S.post_outcome_matches(cols))
    struct = set(S.structural_matches(cols))
    assert post == {"case_resolution", "min_resolve_days", "fraud_suspected",
                    "false_positive"}, post
    assert struct == {"data_month", "snapshot_id"}, struct
    assert not (post | struct) & {"tot_txnamt_cr_l7d", "avg_bal_7days"}


# --------------------------------------------------------------------- runner
def _main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
