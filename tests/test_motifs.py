"""Tests for AML typology motif detection.

The whole value of this module is the discrimination between a laundering shape
and its innocent look-alike — SAML-D ships `Normal_Fan_Out` alongside
`Layered_Fan_Out` precisely because counting edges cannot tell them apart. These
tests pin the evidence that does: amount uniformity, value conservation, and
relay retention.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import motifs as M  # noqa: E402

D0 = pd.Timestamp("2023-01-02")


def _tx(src, dst, amount, day=0):
    return {"src": src, "dst": dst, "amount": amount,
            "date": D0 + pd.Timedelta(days=day)}


def _frame(rows):
    return pd.DataFrame(rows)


# ==========================================================================
# Schema discipline
# ==========================================================================
def test_missing_columns_raise_rather_than_guess():
    bad = _frame([{"from": "A", "to": "B", "value": 1.0, "when": D0}])
    with pytest.raises(KeyError):
        M.detect_motifs(bad)


def test_negative_amounts_are_treated_as_magnitude():
    t = M._prep(_frame([_tx("A", "B", -500.0)]))
    assert t["amount"].iloc[0] == 500.0


# ==========================================================================
# Fan motifs
# ==========================================================================
def test_uniform_fan_out_is_detected():
    rows = [_tx("HUB", f"L{i}", 49_000.0, day=i % 3) for i in range(6)]
    fo = M.fan_out(M._prep(_frame(rows)))
    assert len(fo) == 1
    assert fo.iloc[0]["legs"] == 6
    assert fo.iloc[0]["uniformity"] > 0.95


def test_fan_below_the_leg_minimum_is_ignored():
    rows = [_tx("HUB", f"L{i}", 49_000.0) for i in range(M.MIN_LEGS - 1)]
    assert M.fan_out(M._prep(_frame(rows))).empty


def test_uniform_and_varied_fans_are_separated_by_uniformity():
    """A payroll run and a split sum both fan out. Only one has equal legs."""
    split = [_tx("S", f"L{i}", 50_000.0) for i in range(8)]
    payroll = [_tx("P", f"E{i}", 12_000.0 * (i + 1)) for i in range(8)]
    fo = M.fan_out(M._prep(_frame(split + payroll))).set_index("account")
    assert fo.loc["S", "uniformity"] > fo.loc["P", "uniformity"] + 0.3


def test_threshold_hugging_is_measured():
    rows = [_tx("HUB", f"L{i}", 48_000.0) for i in range(6)]      # just under 50k
    fo = M.fan_out(M._prep(_frame(rows)))
    assert fo.iloc[0]["threshold_hugging"] == pytest.approx(1.0)


def test_fan_in_is_the_mirror_of_fan_out():
    rows = [_tx(f"L{i}", "HUB", 49_000.0) for i in range(6)]
    fi = M.fan_in(M._prep(_frame(rows)))
    assert len(fi) == 1 and fi.iloc[0]["account"] == "HUB"
    assert M.fan_out(M._prep(_frame(rows))).empty


def test_fans_are_split_by_time_window():
    """Legs spread across separate windows are separate events, not one fan."""
    near = [_tx("HUB", f"L{i}", 49_000.0, day=0) for i in range(5)]
    far = [_tx("HUB", f"F{i}", 49_000.0, day=M.WINDOW_DAYS * 3) for i in range(5)]
    fo = M.fan_out(M._prep(_frame(near + far)))
    assert len(fo) == 2, "legs in different windows collapsed into one fan"


# ==========================================================================
# Gather-scatter: the merchant discrimination
# ==========================================================================
def test_conduit_that_passes_value_through_is_flagged():
    rows = [_tx(f"S{i}", "HUB", 50_000.0) for i in range(5)]
    rows += [_tx("HUB", f"D{i}", 49_000.0) for i in range(5)]
    gs = M.gather_scatter(M._prep(_frame(rows)))
    assert not gs.empty
    assert gs.iloc[0]["conservation"] > 0.9


def test_merchant_that_retains_value_is_not_flagged():
    """Fans in exactly like a conduit, but keeps the money. Not a mule."""
    rows = [_tx(f"S{i}", "SHOP", 50_000.0) for i in range(5)]
    rows += [_tx("SHOP", "SUPPLIER", 5_000.0)]
    gs = M.gather_scatter(M._prep(_frame(rows)))
    assert gs.empty or "SHOP" not in set(gs["account"])


# ==========================================================================
# Chains
# ==========================================================================
def test_relay_forwarding_nearly_everything_is_a_chain():
    rows = [_tx("A", "B", 100_000.0, day=0), _tx("B", "C", 98_000.0, day=1)]
    ch = M.chains(M._prep(_frame(rows)))
    assert len(ch) == 1
    assert ch.iloc[0]["relay_retention"] < 0.05


def test_account_that_keeps_most_of_it_is_not_a_relay():
    rows = [_tx("A", "B", 100_000.0, day=0), _tx("B", "C", 5_000.0, day=1)]
    assert M.chains(M._prep(_frame(rows))).empty


def test_chain_respects_direction_of_time():
    """B paying C before A paid B is not a relay of A's money."""
    rows = [_tx("A", "B", 100_000.0, day=5), _tx("B", "C", 98_000.0, day=0)]
    assert M.chains(M._prep(_frame(rows))).empty


def test_chain_does_not_count_money_going_straight_back():
    rows = [_tx("A", "B", 100_000.0, day=0), _tx("B", "A", 98_000.0, day=1)]
    assert M.chains(M._prep(_frame(rows))).empty


# ==========================================================================
# End to end
# ==========================================================================
def test_detect_motifs_scores_the_hub_above_its_legs():
    rows = [_tx(f"S{i}", "HUB", 50_000.0) for i in range(6)]
    rows += [_tx("HUB", f"D{i}", 49_000.0) for i in range(6)]
    res = M.detect_motifs(_frame(rows))
    sc = res["scores"]
    assert sc["HUB"] > sc["S0"], "the hub must outrank its counterparties"
    assert 0.0 <= sc["HUB"] <= 1.0


def test_every_scored_account_carries_named_evidence():
    rows = [_tx(f"S{i}", "HUB", 50_000.0) for i in range(6)]
    rows += [_tx("HUB", f"D{i}", 49_000.0) for i in range(6)]
    res = M.detect_motifs(_frame(rows))
    for acct in res["scores"]:
        assert res["evidence"].get(acct), f"{acct} scored with no evidence"


def test_ordinary_traffic_produces_no_motifs():
    """Unrelated pairs paying each other once should trip nothing."""
    rows = [_tx(f"A{i}", f"B{i}", 1_000.0 + 37 * i, day=i % 20) for i in range(60)]
    res = M.detect_motifs(_frame(rows))
    assert res["counts"]["fan_out"] == 0
    assert res["counts"]["fan_in"] == 0
    assert not res["scores"]
