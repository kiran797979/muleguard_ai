"""Tests for ledger-derived features and the unified scorer.

The unified scorer is the piece that makes this one system rather than five
scripts, so the properties worth pinning are the joins: that an account is
aggregated from both sides of the ledger, that network evidence arrives as
ordinary columns the model can weigh, and that the submission it emits is the
shape the challenge asks for.
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

import ledger_features as LF  # noqa: E402
import score_unified as SU  # noqa: E402

T0 = pd.Timestamp("2023-01-02 10:00")


def tx(src, dst, amount, day=0, hour=10):
    return {"src": src, "dst": dst, "amount": amount,
            "timestamp": T0 + pd.Timedelta(days=day, hours=hour - 10)}


def frame(rows):
    return pd.DataFrame(rows)


# ==========================================================================
# Schema discipline
# ==========================================================================
def test_missing_columns_raise_rather_than_guess():
    with pytest.raises(KeyError):
        LF.build(frame([{"from": "A", "to": "B", "value": 1.0, "when": T0}]))


def test_reversals_are_measured_by_magnitude():
    f = LF.build(frame([tx("A", "B", -500.0), tx("C", "A", 500.0)]))
    assert f.loc["A", "sum_out"] == pytest.approx(500.0)


# ==========================================================================
# An account is both sides of the ledger
# ==========================================================================
def test_account_is_aggregated_from_both_sides():
    """A appears as receiver once and sender once; both must land on one row."""
    f = LF.build(frame([tx("X", "A", 1000.0), tx("A", "Y", 990.0)]))
    assert f.loc["A", "n_in"] == 1 and f.loc["A", "n_out"] == 1
    assert f.loc["A", "sum_in"] == pytest.approx(1000.0)
    assert f.loc["A", "sum_out"] == pytest.approx(990.0)


def test_every_account_in_the_ledger_gets_a_row():
    f = LF.build(frame([tx("A", "B", 100.0), tx("B", "C", 90.0)]))
    assert set(f.index) == {"A", "B", "C"}


# ==========================================================================
# The conduit signature
# ==========================================================================
def test_passthrough_is_one_when_everything_forwards():
    f = LF.build(frame([tx("X", "A", 1000.0), tx("A", "Y", 1000.0)]))
    assert f.loc["A", "passthrough"] == pytest.approx(1.0)
    assert f.loc["A", "retention"] == pytest.approx(0.0, abs=1e-6)


def test_accumulator_has_low_passthrough_and_high_retention():
    f = LF.build(frame([tx("X", "A", 1000.0), tx("A", "Y", 50.0)]))
    assert f.loc["A", "passthrough"] < 0.1
    assert f.loc["A", "retention"] > 0.9


def test_net_flow_is_near_zero_for_a_conduit():
    f = LF.build(frame([tx("X", "A", 1000.0), tx("A", "Y", 995.0)]))
    assert abs(f.loc["A", "net_flow"]) < 0.01


# ==========================================================================
# Shape of the traffic
# ==========================================================================
def test_burstiness_is_high_when_activity_is_concentrated():
    """Two accounts, same transaction count, different spread."""
    burst = [tx("S", "A", 500.0, day=d) for d in range(3)] * 4
    spread = [tx("S", "B", 500.0, day=d * 30) for d in range(12)]
    f = LF.build(frame(burst + spread))
    assert f.loc["A", "burstiness"] > f.loc["B", "burstiness"]


def test_threshold_hugging_is_measured():
    rows = [tx("S", "A", 48_000.0) for _ in range(4)]      # just under 50k
    rows += [tx("S", "B", 12_000.0) for _ in range(4)]
    f = LF.build(frame(rows))
    assert f.loc["A", "threshold_share"] == pytest.approx(1.0)
    assert f.loc["B", "threshold_share"] == pytest.approx(0.0)


def test_round_amounts_are_measured():
    f = LF.build(frame([tx("S", "A", 50_000.0), tx("S", "B", 47_321.0)]))
    assert f.loc["A", "round_share"] == pytest.approx(1.0)
    assert f.loc["B", "round_share"] == pytest.approx(0.0)


def test_night_share_uses_the_hour():
    f = LF.build(frame([tx("S", "A", 100.0, hour=3), tx("S", "B", 100.0, hour=14)]))
    assert f.loc["A", "night_share"] == pytest.approx(1.0)
    assert f.loc["B", "night_share"] == pytest.approx(0.0)


def test_fan_wide_and_shallow_differs_from_repeat_relationships():
    wide = [tx("A", f"D{i}", 1000.0) for i in range(20)]
    deep = [tx("B", "SUPPLIER", 1000.0) for _ in range(20)]
    f = LF.build(frame(wide + deep))
    assert f.loc["B", "txn_per_counterparty"] > f.loc["A", "txn_per_counterparty"]


def test_features_contain_no_infinities_or_nans():
    """Division by zero is everywhere in ratios; none of it may escape."""
    f = LF.build(frame([tx("A", "B", 0.0), tx("C", "D", 100.0)]))
    assert np.isfinite(f.to_numpy()).all()


# ==========================================================================
# The unified scorer
# ==========================================================================
def _mixed_ledger():
    rows = []
    for i in range(60):
        a = f"N{i:03d}"
        for m in range(6):
            rows.append(tx("EMP", a, 40_000.0, day=30 * m))
            rows.append(tx(a, f"BILL{i % 5}", 900.0, day=30 * m + 3))
    for i in range(12):
        a = f"M{i:03d}"
        for d in range(20):
            for _ in range(3):
                rows.append(tx(f"S{d%7}", a, 48_500.0, day=100 + d))
                rows.append(tx(a, f"O{d%7}", 48_000.0, day=100 + d))
    return frame(rows)


def test_account_view_stacks_both_directions():
    v = SU.to_account_view(frame([tx("A", "B", 100.0)]))
    assert len(v) == 2
    assert set(v["direction"]) == {"D", "C"}
    assert set(v["account_id"]) == {"A", "B"}


def test_network_evidence_arrives_as_ordinary_columns():
    feats, meta = SU.build_features(_mixed_ledger(), with_network=True)
    for col in ("motif_score", "ring_score", "role_relay"):
        assert col in feats.columns, f"{col} missing; the model cannot weigh it"
    assert meta["network_used"] is True


def test_network_can_be_switched_off():
    feats, meta = SU.build_features(_mixed_ledger(), with_network=False)
    assert not any(c.startswith(("motif_", "ring_", "role_")) for c in feats.columns)
    assert meta["network_used"] is False


def test_string_account_ids_survive_the_whole_pipeline():
    """Real ledgers carry ids like ACCT_000003, not row indices."""
    feats, _ = SU.build_features(_mixed_ledger(), with_network=True)
    assert all(isinstance(i, str) for i in feats.index[:5])


def test_unsupervised_fallback_is_used_when_labels_are_too_few():
    led = _mixed_ledger()
    y = pd.Series(0, index=sorted(set(led.src) | set(led.dst)))
    y.iloc[0] = 1                                    # one positive: not fittable
    res = SU.run(led, labels=y, threshold=0.9)
    assert res["meta"]["mode"] == "unfitted"
    assert "too few to fit" in res["meta"]["note"]


def test_supervised_mode_ranks_planted_mules_first():
    led = _mixed_ledger()
    accounts = sorted(set(led.src) | set(led.dst))
    y = pd.Series([1 if a.startswith("M") else 0 for a in accounts], index=accounts)
    res = SU.run(led, labels=y, threshold=0.5)
    assert res["meta"]["mode"] == "supervised"
    r = res["risk"].set_index("account_id")["is_mule"]
    top = r.sort_values(ascending=False).head(12).index
    assert sum(a.startswith("M") for a in top) >= 10


def test_submission_has_the_required_shape():
    led = _mixed_ledger()
    accounts = sorted(set(led.src) | set(led.dst))
    y = pd.Series([1 if a.startswith("M") else 0 for a in accounts], index=accounts)
    sub = SU.run(led, labels=y, threshold=0.5)["submission"]
    assert list(sub.columns) == ["account_id", "is_mule",
                                 "suspicious_start", "suspicious_end"]
    assert ((sub.is_mule >= 0) & (sub.is_mule <= 1)).all()
    # A window only where the account is predicted a mule.
    flagged = sub[sub.is_mule >= 0.5]
    quiet = sub[sub.is_mule < 0.5]
    assert (quiet.suspicious_start == "").all()
    assert len(flagged) == 0 or (flagged.suspicious_start != "").any()
