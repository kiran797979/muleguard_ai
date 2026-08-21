"""
Label-noise detection — which training labels should a human look at again?

Why this exists
---------------
Every other defence in this project protects against bad *features*: leaked
columns, assembly artefacts, post-outcome fields. None of them protects against
a bad *label*. The challenge brief is explicit that this is a live risk:

    "Labels may contain noise/red-herrings. Not all labels are guaranteed to be
     correct."

A model trained on noisy labels inherits the noise, and worse, the operating
threshold is chosen against those same labels — so a systematically mislabelled
subset shifts the cutoff for everyone.

The method
----------
Confident learning (Northcutt et al.). The idea is simple and does not require
knowing the noise rate in advance:

  1. Score every account **out of fold**, so no account is judged by a model
     that trained on it. Using in-fold predictions here would be circular: a
     model memorising a bad label would then confirm it.
  2. For each class, take the average confidence the model assigns to accounts
     carrying that label. That average is the class's self-confidence threshold.
  3. An account whose label disagrees with the model *and* which clears the
     other class's threshold is a candidate. Two conditions, not one, because
     "the model disagrees" on its own flags every borderline case.

Two directions, and they mean different things
----------------------------------------------
    LABELLED LEGITIMATE, SCORED HIGH   either a mule nobody caught, or a false
                                       positive. Both are worth an analyst's
                                       time, and for a bank the first is the
                                       more valuable outcome of the whole system.

    LABELLED MULE, SCORED LOW          either a planted red-herring label, or a
                                       real mule whose behaviour this data
                                       cannot show. The brief says the first
                                       exists on purpose.

What this deliberately does NOT claim
-------------------------------------
It does not say a label is wrong. It cannot: a confidently-scored negative is
genuinely ambiguous between "bad label" and "good label, wrong model", and no
amount of arithmetic separates those without going and looking. The output is a
**ranked review queue with the evidence attached**, which is the honest form of
the answer and also the operationally useful one.

A rank-based check runs alongside the probability one, because probabilities
depend on the calibrator and ranks do not. If a supposedly-legitimate account
outranks 99% of confirmed mules, that is suspicious whatever the calibration is
doing.

Run:  python src/label_noise.py
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

# An account must beat the other class's self-confidence by this margin before
# it is worth an analyst's attention. Round number, not fitted: the point of the
# output is a ranked queue, and where a reviewer stops reading is their call.
MIN_MARGIN = 0.0
# Rank evidence: share of the opposite class an account must outrank.
RANK_ALARM = 0.95


def confident_thresholds(y: np.ndarray, p: np.ndarray) -> dict[int, float]:
    """Average predicted probability of each class, among accounts so labelled.

    This is the self-confidence of the label set. It adapts to however
    well-separated the problem happens to be, which is why confident learning
    needs no prior estimate of the noise rate.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        1: float(p[y == 1].mean()) if (y == 1).any() else 1.0,
        0: float((1.0 - p)[y == 0].mean()) if (y == 0).any() else 1.0,
    }


def _rank_position(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each account: the share of positives, and of negatives, it outranks."""
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    over_pos = np.searchsorted(pos, p, side="left") / max(len(pos), 1)
    over_neg = np.searchsorted(neg, p, side="left") / max(len(neg), 1)
    return over_pos, over_neg


def suspect_labels(y, p, index=None, min_margin: float = MIN_MARGIN) -> pd.DataFrame:
    """Rank the labels most worth re-checking, with the evidence for each."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    idx = np.arange(len(y)) if index is None else np.asarray(index)
    thr = confident_thresholds(y, p)
    over_pos, over_neg = _rank_position(p, y)

    rows = []

    # Labelled legitimate, but the model is confident it is a mule.
    neg_mask = (y == 0) & (p >= thr[1])
    for i in np.flatnonzero(neg_mask):
        rows.append({
            "account_idx": idx[i],
            "given_label": 0,
            "probability": float(p[i]),
            "direction": "LABELLED_LEGITIMATE_SCORED_HIGH",
            "margin": float(p[i] - thr[1]),
            "outranks_share_of_mules": float(over_pos[i]),
            "reading": ("a mule nobody caught, or a false positive; "
                        "both are worth a look"),
        })

    # Labelled mule, but the model is confident it is legitimate.
    pos_mask = (y == 1) & ((1.0 - p) >= thr[0])
    for i in np.flatnonzero(pos_mask):
        rows.append({
            "account_idx": idx[i],
            "given_label": 1,
            "probability": float(p[i]),
            "direction": "LABELLED_MULE_SCORED_LOW",
            "margin": float((1.0 - p[i]) - thr[0]),
            "outranks_share_of_normals": float(over_neg[i]),
            "reading": ("a planted red-herring label, or a real mule this data "
                        "cannot show"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["margin"] >= min_margin]
    return df.sort_values("margin", ascending=False).reset_index(drop=True)


def rank_anomalies(y, p, index=None, alarm: float = RANK_ALARM) -> pd.DataFrame:
    """Calibration-free check: labels that sit on the wrong side of the ranking.

    Probabilities move when the calibrator changes; ranks do not. A negative
    that outranks nearly every confirmed mule is worth reviewing whatever the
    probability says.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    idx = np.arange(len(y)) if index is None else np.asarray(index)
    over_pos, over_neg = _rank_position(p, y)

    rows = []
    for i in np.flatnonzero((y == 0) & (over_pos >= alarm)):
        rows.append({"account_idx": idx[i], "given_label": 0,
                     "probability": float(p[i]),
                     "outranks_share_of_mules": float(over_pos[i]),
                     "direction": "LABELLED_LEGITIMATE_SCORED_HIGH"})
    for i in np.flatnonzero((y == 1) & (over_neg <= 1.0 - alarm)):
        rows.append({"account_idx": idx[i], "given_label": 1,
                     "probability": float(p[i]),
                     "outranks_share_of_normals": float(over_neg[i]),
                     "direction": "LABELLED_MULE_SCORED_LOW"})
    df = pd.DataFrame(rows)
    return df if df.empty else df.reset_index(drop=True)


def estimate_noise(y, p) -> dict:
    """A rough share of labels worth re-checking, per class.

    Deliberately called an estimate. Confident learning bounds how many labels
    *disagree confidently* with an out-of-fold model, which is an upper bound on
    genuine mislabelling and a lower bound on nothing at all — a label can be
    wrong in a way the model also gets wrong, and that is invisible here.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    thr = confident_thresholds(y, p)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    sus_neg = int(((y == 0) & (p >= thr[1])).sum())
    sus_pos = int(((y == 1) & ((1.0 - p) >= thr[0])).sum())
    return {
        "self_confidence_mule": round(thr[1], 6),
        "self_confidence_legitimate": round(thr[0], 6),
        "labelled_legitimate": n_neg,
        "labelled_legitimate_flagged": sus_neg,
        "labelled_legitimate_flagged_pct": round(100 * sus_neg / max(n_neg, 1), 4),
        "labelled_mule": n_pos,
        "labelled_mule_flagged": sus_pos,
        "labelled_mule_flagged_pct": round(100 * sus_pos / max(n_pos, 1), 2),
        "caveat": ("An upper bound on confidently-disagreeing labels, not a "
                   "measurement of mislabelling. A label the model gets wrong in "
                   "the same direction is invisible to this test."),
    }


def audit(y, p, index=None) -> dict:
    """Everything, assembled for a report."""
    sus = suspect_labels(y, p, index)
    rank = rank_anomalies(y, p, index)
    return {
        "method": ("confident learning on out-of-fold predictions "
                   "(Northcutt et al.), plus a calibration-free rank check"),
        "uses_out_of_fold_scores": True,
        "estimate": estimate_noise(y, p),
        "n_flagged": int(len(sus)),
        "n_rank_anomalies": int(len(rank)),
        "by_direction": (sus["direction"].value_counts().to_dict()
                         if not sus.empty else {}),
        "top_candidates": (sus.head(25).to_dict(orient="records")
                           if not sus.empty else []),
        "what_this_is_not": ("This does not assert that any label is wrong. A "
                             "confidently-scored negative is ambiguous between a "
                             "bad label and a good label with a wrong model, and "
                             "nothing here separates those without investigation. "
                             "It is a ranked review queue."),
    }


def main() -> None:
    import json

    import config as C
    from utils import log, save_json

    ap = argparse.ArgumentParser(description="Label-noise audit")
    ap.add_argument("--oof", default=None, help="CSV with y_true and a probability column")
    ap.add_argument("--prob-col", default="p_ensemble_calibrated")
    args = ap.parse_args()

    path = pathlib.Path(args.oof) if args.oof else C.DATA_DIR / "oof_predictions.csv"
    if not path.exists():
        log(f"No out-of-fold predictions at {path}. Run Stage 4/5 first.")
        return

    d = pd.read_csv(path)
    res = audit(d["y_true"].to_numpy(), d[args.prob_col].to_numpy(), d.index.to_numpy())

    e = res["estimate"]
    log("Label-noise audit (confident learning, out-of-fold)")
    log(f"  self-confidence  mule {e['self_confidence_mule']:.4f}   "
        f"legitimate {e['self_confidence_legitimate']:.4f}")
    log(f"  labelled legitimate : {e['labelled_legitimate_flagged']:,} of "
        f"{e['labelled_legitimate']:,} flagged ({e['labelled_legitimate_flagged_pct']}%)")
    log(f"  labelled mule       : {e['labelled_mule_flagged']:,} of "
        f"{e['labelled_mule']:,} flagged ({e['labelled_mule_flagged_pct']}%)")
    log(f"  rank anomalies      : {res['n_rank_anomalies']:,}")

    save_json(res, C.REPORTS_DIR / "11_label_noise.json")
    log(f"Wrote {C.REPORTS_DIR / '11_label_noise.json'}")


if __name__ == "__main__":
    main()
