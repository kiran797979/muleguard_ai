"""
Drift detection and the operating-point re-selection policy.

The problem
-----------
Every cutoff this system uses was chosen inside cross-validation on a snapshot.
The high band is the threshold that held precision at or above a target; the
medium band is the review-queue threshold. Both are properties of the data as it
looked when they were fitted.

Data moves. A new product launches, a channel changes its settlement pattern, a
fraud ring switches technique, an upstream extract starts populating a field it
used to leave blank. When it moves, a cutoff chosen for 0.90 precision quietly
stops delivering 0.90 precision, and nothing in the system notices — the queue
still fills, the alerts still look the same, and the first sign of trouble is an
investigator saying the alerts have gone bad.

Detecting drift is the easy half and most projects stop there. The hard half is
having decided, in advance and in writing, what you do about it.

The distinction this module is built around
-------------------------------------------
There are two kinds of signal and they license completely different actions:

  UNSUPERVISED   feature distributions, score distributions, band populations.
                 Available immediately on every batch, with no labels. These can
                 tell you something changed. They CANNOT tell you the model got
                 worse, and they must never be used to move a precision-targeted
                 threshold — re-fitting a cutoff to make the band populations
                 look normal again is fitting to noise, and it would hide exactly
                 the degradation you are trying to catch.

  SUPERVISED     realised precision from investigator decisions. Arrives late and
                 only for accounts somebody actually reviewed, which is a biased
                 sample by construction. This is the only signal that licenses
                 moving a threshold.

So: unsupervised drift raises the alarm and can trigger recalibration or a
retrain. Only confirmed outcomes can re-select an operating point.

The ladder
----------
    MONITOR             within tolerance; record and continue
    RECALIBRATE         scores shifted but ranking looks intact; refit the
                        calibrator, leave the models alone
    REFIT_THRESHOLDS    realised precision has fallen below target on reviewed
                        accounts; re-derive cutoffs from recent labels
    RETRAIN             input features have materially moved; full refit
    HALT_AUTOMATION     drift severe enough that automated freezing stops until
                        a human signs off

That last rung exists because this system freezes people's money. There has to
be a written condition under which it stops doing that by itself.

Hysteresis
----------
A single noisy batch must not trigger a retrain. Every rung above MONITOR
requires the condition to hold for `CONSECUTIVE_BREACHES` consecutive windows,
tracked in a small state file, so the policy is stable rather than twitchy.

Run:  python src/drift.py --reference data/features.parquet --current new.parquet
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Population Stability Index bands. These are the long-standing credit-risk
# conventions, not numbers we fitted: below 0.10 stable, 0.10-0.25 moderate,
# above 0.25 significant. Using an industry convention rather than our own
# threshold matters here, because a bank's model-risk function already knows
# what a PSI of 0.25 means.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
PSI_SEVERE = 0.50

KS_ALARM = 0.15               # per-feature Kolmogorov-Smirnov distance
SCORE_PSI_ALARM = 0.10        # the model's own output distribution
PRECISION_TOLERANCE = 0.10    # relative fall from target before acting
CONSECUTIVE_BREACHES = 2      # windows a condition must hold before it acts
MIN_REVIEWED_FOR_PRECISION = 30   # below this, realised precision is noise


@dataclass
class DriftDecision:
    action: str
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    requires_human_signoff: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reasons": self.reasons,
            "evidence": self.evidence,
            "requires_human_signoff": self.requires_human_signoff,
        }


# ==========================================================================
# Distance measures
# ==========================================================================
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples of one variable.

    Bin edges come from the *reference* quantiles, because the reference is the
    distribution the model was fitted on. Re-binning on the current data would
    measure nothing: any distribution is stable against bins drawn from itself.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    r = np.histogram(ref, bins=edges)[0].astype(float)
    c = np.histogram(cur, bins=edges)[0].astype(float)
    # Laplace smoothing: an empty bin gives an infinite PSI otherwise, which
    # would make one absent value look like total collapse.
    r = (r + 0.5) / (r.sum() + 0.5 * len(r))
    c = (c + 0.5) / (c.sum() + 0.5 * len(c))
    return float(np.sum((c - r) * np.log(c / r)))


def ks_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, without the p-value.

    The p-value is deliberately not used. With a large enough batch every
    trivial difference becomes significant, so a p-value answers "is there any
    difference at all" when the question is "is the difference big enough to
    act on". The statistic is an effect size; the p-value is not.
    """
    from scipy.stats import ks_2samp
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    return float(ks_2samp(ref, cur).statistic)


# ==========================================================================
# What moved
# ==========================================================================
def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  features: list[str] | None = None,
                  importance: dict[str, float] | None = None) -> dict:
    """Per-feature drift, weighted by how much the model actually uses it.

    Drift in a column the model ignores is not a problem, and reporting it as
    one trains everybody to ignore the alarm. When SHAP importances are supplied
    the summary is weighted by them.
    """
    cols = features or [c for c in reference.columns if c in current.columns]
    cols = [c for c in cols
            if pd.api.types.is_numeric_dtype(reference[c])
            and pd.api.types.is_numeric_dtype(current[c])]

    rows = []
    for c in cols:
        rows.append({
            "feature": c,
            "psi": round(psi(reference[c].to_numpy(), current[c].to_numpy()), 4),
            "ks": round(ks_distance(reference[c].to_numpy(), current[c].to_numpy()), 4),
            "importance": float((importance or {}).get(c, 0.0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_features": 0, "drifted": [], "weighted_psi": 0.0}

    df = df.sort_values("psi", ascending=False)
    w = df["importance"].to_numpy()
    weighted = (float(np.average(df["psi"], weights=w)) if w.sum() > 0
                else float(df["psi"].mean()))
    drifted = df[(df.psi >= PSI_SIGNIFICANT) | (df.ks >= KS_ALARM)]
    return {
        "n_features": int(len(df)),
        "weighted_psi": round(weighted, 4),
        "max_psi": float(df["psi"].max()),
        "n_drifted": int(len(drifted)),
        "share_drifted": round(len(drifted) / max(len(df), 1), 4),
        "drifted": drifted.head(20).to_dict("records"),
    }


def score_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> dict:
    """Drift in the model's own output, which is the most direct early warning.

    Feature drift may or may not reach the prediction. A shift in the score
    distribution definitely has.
    """
    return {
        "psi": round(psi(reference_scores, current_scores), 4),
        "ks": round(ks_distance(reference_scores, current_scores), 4),
        "reference_mean": round(float(np.mean(reference_scores)), 6),
        "current_mean": round(float(np.mean(current_scores)), 6),
    }


def realised_precision(decisions: pd.DataFrame) -> dict:
    """Precision actually achieved on accounts an investigator closed.

    The only signal that licenses moving a threshold, and a biased one: we only
    ever learn the outcome of accounts somebody chose to review, which is not a
    random sample of the book. Reported with its count so a reader can weigh it.
    """
    if decisions is None or decisions.empty:
        return {"reviewed": 0, "precision": None,
                "note": "no closed investigator decisions yet"}
    closed = decisions[decisions["outcome"].isin(["CONFIRMED", "DISMISSED"])]
    n = len(closed)
    if n == 0:
        return {"reviewed": 0, "precision": None, "note": "no closed decisions"}
    confirmed = int((closed["outcome"] == "CONFIRMED").sum())
    return {
        "reviewed": int(n),
        "confirmed": confirmed,
        "precision": round(confirmed / n, 4),
        "sufficient": bool(n >= MIN_REVIEWED_FOR_PRECISION),
        "note": ("Reviewed accounts are not a random sample of the book; this is "
                 "precision among alerts a human chose to close."),
    }


# ==========================================================================
# The policy
# ==========================================================================
def _load_state(path: pathlib.Path) -> dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def assess(feature_report: dict, score_report: dict, precision_report: dict,
           target_precision: float, state_path: pathlib.Path | None = None,
           ) -> DriftDecision:
    """Decide what to do, applying hysteresis so one noisy batch cannot act."""
    state = _load_state(state_path) if state_path else {}
    streak = dict(state.get("streaks", {}))

    raw: dict[str, bool] = {}

    def breach(name: str, condition: bool) -> bool:
        """True only once a condition has held for enough consecutive windows."""
        raw[name] = bool(condition)
        streak[name] = streak.get(name, 0) + 1 if condition else 0
        return streak[name] >= CONSECUTIVE_BREACHES

    wpsi = float(feature_report.get("weighted_psi", 0.0))
    spsi = float(score_report.get("psi", 0.0))
    prec = precision_report.get("precision")
    enough = bool(precision_report.get("sufficient"))

    severe = breach("severe_features", wpsi >= PSI_SEVERE)
    material = breach("material_features", wpsi >= PSI_SIGNIFICANT)
    scores_moved = breach("score_shift", spsi >= SCORE_PSI_ALARM)
    precision_fell = breach(
        "precision_fell",
        bool(enough and prec is not None
             and prec < target_precision * (1 - PRECISION_TOLERANCE)))

    reasons: list[str] = []
    action = "MONITOR"
    signoff = False

    # Ordered most severe first; the first match wins.
    if severe:
        action, signoff = "HALT_AUTOMATION", True
        reasons.append(
            f"weighted feature PSI {wpsi:.3f} is at or above the severe band "
            f"({PSI_SEVERE}); the population the model was fitted on is no longer "
            f"the population being scored, so automated freezing stops until a "
            f"human signs off")
    elif precision_fell:
        action = "REFIT_THRESHOLDS"
        reasons.append(
            f"realised precision {prec:.3f} on {precision_report['reviewed']} "
            f"reviewed accounts is more than {PRECISION_TOLERANCE:.0%} below the "
            f"{target_precision:.2f} target; cutoffs re-derived from recent "
            f"confirmed outcomes")
    elif material:
        action = "RETRAIN"
        reasons.append(
            f"weighted feature PSI {wpsi:.3f} is at or above the significant band "
            f"({PSI_SIGNIFICANT}) across {feature_report.get('n_drifted', 0)} "
            f"features; the inputs have moved enough to warrant a refit")
    elif scores_moved:
        action = "RECALIBRATE"
        reasons.append(
            f"score PSI {spsi:.3f} exceeds {SCORE_PSI_ALARM} while feature drift "
            f"stays inside tolerance; the ranking looks intact, so refit the "
            f"calibrator and leave the models alone")
    else:
        # "Not yet confirmed" is not "fine". A batch can sit far outside
        # tolerance and still take no action, because hysteresis requires the
        # condition to hold twice. Saying "all quantities inside tolerance"
        # while the weighted PSI reads 2.90 is simply false, and it is the
        # sentence an operator would quote back when asking why nobody acted.
        pending = [n for n, hit in raw.items() if hit]
        if pending:
            reasons.append(
                "; ".join(
                    f"{n.replace('_', ' ')} is OUTSIDE tolerance but has held "
                    f"for {streak.get(n, 0)} of {CONSECUTIVE_BREACHES} required "
                    f"consecutive windows"
                    for n in pending)
                + " — measured, not yet actioned")
        else:
            reasons.append("all monitored quantities are inside tolerance")

    # Say plainly when the deciding signal is missing, rather than reading
    # silence as good news.
    if prec is None or not enough:
        reasons.append(
            f"no usable realised precision yet "
            f"({precision_report.get('reviewed', 0)} reviewed, "
            f"{MIN_REVIEWED_FOR_PRECISION} needed); threshold re-selection is "
            f"unavailable and unsupervised drift alone must never move a "
            f"precision-targeted cutoff")

    decision = DriftDecision(
        action=action, reasons=reasons, requires_human_signoff=signoff,
        evidence={"weighted_feature_psi": wpsi, "score_psi": spsi,
                  "realised_precision": prec,
                  "reviewed": precision_report.get("reviewed", 0),
                  "target_precision": target_precision,
                  "streaks": streak,
                  "conditions_true_now": raw,
                  "consecutive_breaches_required": CONSECUTIVE_BREACHES})

    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(
            {"streaks": streak, "last_action": action}, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    import config as C
    from utils import load_frame, log, save_json

    ap = argparse.ArgumentParser(description="Drift assessment and re-selection policy")
    ap.add_argument("--reference", default=None, help="parquet/csv the model was fitted on")
    ap.add_argument("--current", required=True, help="the new batch")
    ap.add_argument("--target-precision", type=float, default=C.PRECISION_TARGET)
    args = ap.parse_args()

    ref_path = pathlib.Path(args.reference) if args.reference else C.FEATURES_PARQUET
    if not ref_path.exists():
        log(f"No reference at {ref_path}; run the pipeline first.")
        return
    ref = load_frame(ref_path)
    cur = load_frame(pathlib.Path(args.current))

    imp = {}
    shap_path = C.REPORTS_DIR / "05_shap_top_features.json"
    if shap_path.exists():
        d = json.loads(shap_path.read_text(encoding="utf-8"))
        for r in d.get("top_features_by_mean_abs_shap", []):
            imp[r.get("feature")] = float(r.get("mean_abs_shap", 0.0))

    fr = feature_drift(ref, cur, importance=imp)
    sr = {"psi": 0.0, "ks": 0.0, "reference_mean": 0.0, "current_mean": 0.0}
    pr = realised_precision(None)

    decision = assess(fr, sr, pr, args.target_precision,
                      state_path=C.REPORTS_DIR / "drift_state.json")

    log(f"Drift decision: {decision.action}")
    for r in decision.reasons:
        log(f"  - {r}")
    save_json({"feature_drift": fr, "score_drift": sr,
               "realised_precision": pr, "decision": decision.to_dict()},
              C.REPORTS_DIR / "12_drift_policy.json")
    log(f"Wrote {C.REPORTS_DIR / '12_drift_policy.json'}")


if __name__ == "__main__":
    main()
