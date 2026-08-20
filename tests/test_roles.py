"""
Tests for role based column resolution.

These guard the claim on deck slide 21: that the 29 behavioural features build on
a schema nobody has seen, by matching what a column MEANS rather than what it is
called. Each test pins one inference, so a failure says exactly which one broke.

The two regression tests at the bottom are the bugs this module actually shipped
with and had to have fixed. They are here so they cannot come back.

    python tests/test_roles.py
    pytest tests/test_roles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import roles as R  # noqa: E402


# ---------------------------------------------------------------- parsing
def test_parses_the_hackathon_naming_style():
    r = R.parse_role("TOT_TXNAMT_CR_L7D")
    assert (r.stat, r.measure, r.direction, r.window) == ("TOT", "amount", "credit", 7), r


def test_amount_beats_count_in_a_compound_token():
    """TXNAMT contains both a count word and an amount word. It is an amount."""
    assert R.parse_role("TOT_TXNAMT_CR_L7D").measure == "amount"
    assert R.parse_role("TOT_TXNS_L7D").measure == "count"


def test_recognises_foreign_naming_styles():
    cases = {
        "InwardAmount7Day": ("amount", "credit", 7),
        "credit_value_week": ("amount", "credit", 7),
        "outflow.amt.31d": ("amount", "debit", 31),
        "mean balance 31 days": ("balance", None, 31),
        "cash_withdrawal_amt_7d": ("amount", "debit", 7),
    }
    for name, (measure, direction, window) in cases.items():
        r = R.parse_role(name)
        assert (r.measure, r.direction, r.window) == (measure, direction, window), (name, r)


def test_channel_detection():
    assert R.parse_role("upi_inflow_value_week").channel == "UPI"
    assert R.parse_role("atm_debit_amount_7d").channel == "ATM"
    assert R.parse_role("NEFT_AMT_DB_L7D").channel == "ELEC_XFER"
    assert R.parse_role("credit_value_week").channel is None


def test_month_is_never_read_as_a_window():
    """The confound guard.

    MNTH is the extract-month column. Reading it as "a 31 day window" would let
    a behavioural feature resolve onto it and quietly reinstate the exact leak
    the pipeline exists to remove.
    """
    for name in ("MNTH", "MONTH", "data_month", "extract_month"):
        assert R.parse_role(name).window is None, name


def test_identifiers_and_dates_carry_no_measure():
    for name in ("customer_ref", "account_number", "acct_opened_on", "ref_key"):
        assert R.parse_role(name).measure is None, name


# ---------------------------------------------------------------- matching
def _index():
    return R.RoleIndex([
        "InwardAmount7Day", "OutwardAmount7Day", "total value 7 days",
        "mean.balance.7d", "transaction count week", "cash_withdrawal_amt_7d",
        "upi_inflow_value_week", "atm_debit_amount_7d",
        "total inward amount 31 days", "total outward amount 31 days",
        "customer_ref", "MNTH", "client_category",
    ])


def test_resolves_across_naming_conventions():
    idx = _index()
    wanted = {
        "TOT_TXNAMT_CR_L7D": "InwardAmount7Day",
        "TOT_TXNAMT_DB_L7D": "OutwardAmount7Day",
        "AVG_BAL_7DAYS": "mean.balance.7d",
        "CASH_AMT_DB_L7D": "cash_withdrawal_amt_7d",
        "UPI_AMT_CR_L7D": "upi_inflow_value_week",
        "TOT_TXNAMT_CR_L31D": "total inward amount 31 days",
    }
    for req, expect in wanted.items():
        got = idx.find(R.parse_role(req))
        assert got == expect, f"{req} resolved to {got!r}, expected {expect!r}"


def test_direction_is_never_silently_swapped():
    """A credit total resolving onto a debit column would invert pass-through."""
    idx = _index()
    assert idx.find(R.parse_role("TOT_TXNAMT_CR_L7D")) != "OutwardAmount7Day"
    assert idx.find(R.parse_role("TOT_TXNAMT_DB_L7D")) != "InwardAmount7Day"


def test_undirected_request_does_not_grab_a_one_sided_column():
    idx = _index()
    got = idx.find(R.parse_role("TOT_TXNAMT_L7D"))
    assert got == "total value 7 days", got


def test_channel_request_does_not_match_a_channelless_column():
    idx = R.RoleIndex(["InwardAmount7Day"])          # no channel anywhere
    assert idx.find(R.parse_role("UPI_AMT_CR_L7D")) is None


def test_missing_quantity_returns_none_rather_than_a_wrong_guess():
    idx = _index()
    assert idx.find(R.parse_role("CHQ_AMT_DB_L7D")) is None


# ------------------------------------------------------------- regressions
def test_regression_digit_uppercase_must_not_split_a_window_token():
    """The camelCase splitter originally used ([a-z0-9])([A-Z]).

    That turns AVG_BAL_7DAYS into `7` + `DAYS` and 7D into `7` + `D`, so every
    window became invisible and half the features stopped resolving.
    """
    assert R.parse_role("AVG_BAL_7DAYS").window == 7
    assert R.parse_role("TOT_TXNAMT_CR_L7D").window == 7
    assert R.parse_role("OutwardAmount7Day").window == 7


def test_regression_separator_between_number_and_unit():
    """"total inward amount 31 days" splits the number away from its unit."""
    assert R.parse_role("total inward amount 31 days").window == 31
    assert R.parse_role("credit_amt_7_d").window == 7


# ---------------------------------------------------------------- runner
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
