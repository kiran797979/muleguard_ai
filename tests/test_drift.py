"""Tests for drift detection and the re-selection policy.

Two properties matter more than the rest, and both are about restraint:

  * unsupervised drift must never move a precision-targeted threshold, because
    re-fitting a cutoff to make band populations look normal would conceal the
    degradation the monitoring exists to find;
  * one noisy batch must never trigger a retrain.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import drift as D  # noqa: E402

TARGET = 0.90


def _report(wpsi=0.0, n_drifted=0):
    return {"weighted_psi": wpsi, "n_drifted": n_drifted, "n_features": 100}


def _scores(psi=0.0):
    return {"psi": psi, "ks": 0.0, "reference_mean": 0.1, "current_mean": 0.1}


def _precision(p=None, reviewed=0):
    return {"precision": p, "reviewed": reviewed,
            "sufficient": reviewed >= D.MIN_REVIEWED_FOR_PRECISION}


def _settle(fr, sr, pr, state, times=D.CONSECUTIVE_BREACHES):
    """Apply the same window repeatedly so hysteresis has a chance to trip."""
    out = None
    for _ in range(times):
        out = D.assess(fr, sr, pr, TARGET, state_path=state)
    return out


# ==========================================================================
# PSI
# ==========================================================================
def test_identical_samples_have_near_zero_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert D.psi(x, x.copy()) < 0.01


def test_shifted_distribution_has_high_psi():
    rng = np.random.default_rng(0)
    assert D.psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > D.PSI_SIGNIFICANT


def test_psi_uses_reference_bins_not_current():
    """Binning on the current data would make every distribution look stable."""
    rng = np.random.default_rng(1)
    ref, cur = rng.normal(0, 1, 5000), rng.normal(3, 1, 5000)
    assert D.psi(ref, cur) > 0.5


def test_empty_bin_does_not_produce_infinity():
    ref = np.concatenate([np.zeros(100), np.ones(100)])
    cur = np.zeros(100)
    v = D.psi(ref, cur)
    assert np.isfinite(v)


def test_psi_handles_empty_input():
    assert D.psi(np.array([]), np.array([1.0, 2.0])) == 0.0


# ==========================================================================
# The rule that matters most
# ==========================================================================
def test_unsupervised_drift_never_refits_thresholds():
    """Feature drift alone may retrain or halt. It must not move a cutoff."""
    state = None
    for wpsi in (0.3, 0.6, 0.9):
        d = D.assess(_report(wpsi, 40), _scores(0.4), _precision(None, 0), TARGET, state)
        assert d.action != "REFIT_THRESHOLDS", (
            f"unsupervised PSI {wpsi} moved a precision-targeted threshold")


def test_missing_precision_is_reported_not_assumed_good():
    d = D.assess(_report(), _scores(), _precision(None, 0), TARGET)
    assert any("no usable realised precision" in r for r in d.reasons)


def test_thin_review_volume_does_not_license_a_refit(tmp_path):
    """Precision from a handful of reviews is noise, not evidence."""
    thin = _precision(0.20, D.MIN_REVIEWED_FOR_PRECISION - 1)
    d = _settle(_report(), _scores(), thin, tmp_path / "s.json")
    assert d.action != "REFIT_THRESHOLDS"


# ==========================================================================
# The ladder
# ==========================================================================
def test_quiet_data_stays_on_monitor():
    d = D.assess(_report(0.02), _scores(0.01), _precision(0.95, 100), TARGET)
    assert d.action == "MONITOR"


def test_score_shift_alone_triggers_recalibration(tmp_path):
    d = _settle(_report(0.02), _scores(0.20), _precision(0.95, 100), tmp_path / "s.json")
    assert d.action == "RECALIBRATE"


def test_material_feature_drift_triggers_retrain(tmp_path):
    d = _settle(_report(0.30, 25), _scores(0.05), _precision(0.95, 100), tmp_path / "s.json")
    assert d.action == "RETRAIN"


def test_fallen_precision_triggers_threshold_refit(tmp_path):
    d = _settle(_report(0.02), _scores(0.02), _precision(0.60, 200), tmp_path / "s.json")
    assert d.action == "REFIT_THRESHOLDS"


def test_severe_drift_halts_automation_and_demands_signoff(tmp_path):
    d = _settle(_report(0.80, 90), _scores(0.5), _precision(0.95, 100), tmp_path / "s.json")
    assert d.action == "HALT_AUTOMATION"
    assert d.requires_human_signoff is True


def test_severe_drift_outranks_a_precision_fall(tmp_path):
    """When the population itself has moved, stop before adjusting a dial."""
    d = _settle(_report(0.80, 90), _scores(0.5), _precision(0.10, 500), tmp_path / "s.json")
    assert d.action == "HALT_AUTOMATION"


def test_only_halt_requires_signoff(tmp_path):
    for fr, sr, pr in ((_report(0.30, 25), _scores(0.05), _precision(0.95, 100)),
                       (_report(0.02), _scores(0.20), _precision(0.95, 100)),
                       (_report(0.02), _scores(0.02), _precision(0.60, 200))):
        d = _settle(fr, sr, pr, tmp_path / f"s{hash(str(fr))}.json")
        assert d.requires_human_signoff is False


# ==========================================================================
# Hysteresis
# ==========================================================================
def test_a_single_bad_batch_does_not_act(tmp_path):
    state = tmp_path / "state.json"
    d = D.assess(_report(0.30, 25), _scores(0.05), _precision(0.95, 100), TARGET, state)
    assert d.action == "MONITOR", "one window was enough to trigger a retrain"


def test_a_persistent_breach_does_act(tmp_path):
    state = tmp_path / "state.json"
    fr, sr, pr = _report(0.30, 25), _scores(0.05), _precision(0.95, 100)
    first = D.assess(fr, sr, pr, TARGET, state)
    second = D.assess(fr, sr, pr, TARGET, state)
    assert first.action == "MONITOR" and second.action == "RETRAIN"


def test_a_clean_window_resets_the_streak(tmp_path):
    state = tmp_path / "state.json"
    bad, good = _report(0.30, 25), _report(0.01)
    D.assess(bad, _scores(), _precision(0.95, 100), TARGET, state)
    D.assess(good, _scores(), _precision(0.95, 100), TARGET, state)   # recovery
    d = D.assess(bad, _scores(), _precision(0.95, 100), TARGET, state)
    assert d.action == "MONITOR", "the streak survived a clean window"


def test_state_is_persisted_between_calls(tmp_path):
    state = tmp_path / "state.json"
    D.assess(_report(0.30, 25), _scores(), _precision(0.95, 100), TARGET, state)
    assert state.exists()
    assert "streaks" in json.loads(state.read_text(encoding="utf-8"))


def test_missing_state_file_is_not_an_error(tmp_path):
    d = D.assess(_report(), _scores(), _precision(), TARGET, tmp_path / "nope" / "s.json")
    assert d.action == "MONITOR"


def test_corrupt_state_file_is_survived(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{not json", encoding="utf-8")
    d = D.assess(_report(), _scores(), _precision(), TARGET, state)
    assert d.action == "MONITOR"


# ==========================================================================
# Feature drift weighting and realised precision
# ==========================================================================
def test_drift_in_an_unused_feature_is_discounted():
    """A column the model ignores must not raise the alarm."""
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"used": rng.normal(0, 1, 2000), "ignored": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"used": rng.normal(0, 1, 2000), "ignored": rng.normal(5, 1, 2000)})
    weighted = D.feature_drift(ref, cur, importance={"used": 1.0, "ignored": 0.0})
    unweighted = D.feature_drift(ref, cur)
    assert weighted["weighted_psi"] < unweighted["weighted_psi"]


def test_realised_precision_counts_only_closed_decisions():
    dec = pd.DataFrame({"outcome": ["CONFIRMED", "CONFIRMED", "DISMISSED", "OPEN"]})
    r = D.realised_precision(dec)
    assert r["reviewed"] == 3
    assert r["precision"] == pytest.approx(2 / 3, abs=1e-4)   # reported to 4 dp


def test_realised_precision_states_its_own_bias():
    dec = pd.DataFrame({"outcome": ["CONFIRMED", "DISMISSED"]})
    assert "not a random sample" in D.realised_precision(dec)["note"]


def test_no_decisions_yields_no_precision():
    assert D.realised_precision(pd.DataFrame()) ["precision"] is None
