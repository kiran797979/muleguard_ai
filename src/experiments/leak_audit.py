"""
Experiment E — calendar/cohort leak audit (the centerpiece).

The dataset is a case-control assembly artifact: every one of the 9,001 normal
accounts is sampled from an October-2025 snapshot, while all 81 confirmed mules
come from Sep/Nov/Dec-2025 investigations. The explicit month column (F2230) is
already dropped as a categorical leak — but that does NOT remove the *design*
confound. Any calendar-anchored feature (account vintage, recency buckets) can act
as a partial proxy for "which cohort/month this record came from", i.e. for the
label, without carrying generalizable fraud signal.

This audit answers the one question that decides whether the ~0.90 AUPRC headline
is a real behavioural ceiling or a cohort artifact:

  (1) HONEST OOF AUPRC/AUROC with vs WITHOUT the calendar/recency features.
        - barely moves  -> signal is not (only) cohort-driven  [reassuring]
        - collapses     -> the model leans on the cohort clock [ceiling is lower]

Plus multivariate concentration diagnostics (concentrated leak vs distributed signal):
  (2) permutation importance on the OOF model,
  (3) drop-top-k AUPRC degradation curve,
  (4) AUROC of a model trained on ONLY the top-5 features
        (~0.99 => a tiny cluster reconstructs the label = concentrated leak;
         ~0.85 => signal is spread across many moderate features = healthier).

It also records the hard structural finding that a FORWARD TEMPORAL SPLIT is
infeasible by construction (no month contains both classes), so temporal
validation cannot be fabricated — only disclosed.

Run:  .venv/bin/python src/experiments/leak_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "experiments"))

import config as C  # noqa: E402
from utils import log, save_json  # noqa: E402
import _harness as H  # noqa: E402


# Calendar / recency / cohort-clock feature name patterns. F3888 = account-open
# date -> vintage; F3889_* = recency buckets (days-since-last-activity). These are
# the features whose value is literally anchored to the calendar and therefore the
# prime suspects for encoding the sampling month/cohort.
CALENDAR_PREFIXES = ("F3888", "F3889")


def calendar_columns(feat_names: list[str]) -> list[str]:
    return [c for c in feat_names
            if any(c == p or c.startswith(p + "_") or c == p for p in CALENDAR_PREFIXES)]


def _mean_ranking(oofs: list[np.ndarray], y: np.ndarray) -> dict:
    """Average threshold-free metrics across repeated-OOF vectors."""
    per = [H.ranking_metrics(y, p) for p in oofs]
    keys = per[0].keys()
    return {k: round(float(np.mean([d[k] for d in per])), 4) for k in keys}


def main() -> None:
    data = H.load_features()
    X, y, names = data.X, data.y, data.feat_names
    log(f"Leak audit on {X.shape[0]:,} x {X.shape[1]:,} features, positives={int(y.sum())}")

    cal_cols = calendar_columns(names)
    cal_idx = [i for i, c in enumerate(names) if c in cal_cols]
    keep_idx = [i for i in range(len(names)) if i not in set(cal_idx)]
    log(f"Calendar/recency features ({len(cal_cols)}): {cal_cols}")

    report: dict = {
        "n_accounts": int(len(y)),
        "n_mules": int(y.sum()),
        "calendar_features": cal_cols,
        "n_calendar_features": len(cal_cols),
    }

    # --- (1) with vs without calendar features (repeated honest OOF) ---
    # spw-only, no SMOTE: the cleanest ranking-honest configuration.
    n_rep = 5
    log(f"(1) Honest OOF AUPRC/AUROC, {n_rep} repeats, with vs without calendar features ...")
    oof_all = H.repeated_oof(X, y, n_repeats=n_rep, use_smote=False, use_spw=True)
    with_cal = _mean_ranking(oof_all, y)

    oof_nocal = H.repeated_oof(X[:, keep_idx], y, n_repeats=n_rep, use_smote=False, use_spw=True)
    without_cal = _mean_ranking(oof_nocal, y)

    report["with_calendar"] = with_cal
    report["without_calendar"] = without_cal
    report["auprc_drop_when_removed"] = round(with_cal["auprc"] - without_cal["auprc"], 4)
    report["auroc_drop_when_removed"] = round(with_cal["auroc"] - without_cal["auroc"], 4)
    log(f"    AUPRC with={with_cal['auprc']}  without={without_cal['auprc']}  "
        f"drop={report['auprc_drop_when_removed']}")
    log(f"    AUROC with={with_cal['auroc']}  without={without_cal['auroc']}  "
        f"drop={report['auroc_drop_when_removed']}")

    # Verdict heuristic on the calendar dependence.
    drop = report["auprc_drop_when_removed"]
    if drop <= 0.03:
        report["calendar_verdict"] = ("ROBUST — removing calendar features barely changes AUPRC; "
                                      "the signal is not primarily cohort-driven.")
    elif drop <= 0.10:
        report["calendar_verdict"] = ("MODERATE — calendar features contribute meaningfully; part of "
                                      "the headline may be cohort timing, report both numbers.")
    else:
        report["calendar_verdict"] = ("FRAGILE — AUPRC collapses without calendar features; the "
                                      "headline is substantially a cohort/month artifact.")
    log(f"    VERDICT: {report['calendar_verdict']}")

    # --- (2) permutation importance on a single honest OOF model view ---
    # Fit one model on all data (for importance ranking only) and permute each
    # feature's column on a held-out-style OOF probability proxy.
    log("(2) Permutation importance (top 15 by AUPRC drop) ...")
    perm = _permutation_importance(X, y, names, n_repeats=1)
    report["top_permutation_importance"] = perm[:15]

    # --- (3) drop-top-k AUPRC degradation curve ---
    log("(3) Drop-top-k AUPRC degradation curve ...")
    ranked = [p["feature"] for p in perm]
    ranked_idx = [names.index(f) for f in ranked]
    curve = []
    for k in (0, 1, 3, 5, 10, 25):
        drop_set = set(ranked_idx[:k])
        cols = [i for i in range(len(names)) if i not in drop_set]
        oof_k = H.oof_probabilities(X[:, cols], y, use_smote=False, use_spw=True)
        m = H.ranking_metrics(y, oof_k)
        curve.append({"dropped_top_k": k, "auprc": m["auprc"], "auroc": m["auroc"]})
        log(f"    drop top {k:>2}: AUPRC={m['auprc']}  AUROC={m['auroc']}")
    report["drop_top_k_curve"] = curve

    # --- (4) top-5-only model ---
    log("(4) Model trained on top-5 features only ...")
    top5_idx = ranked_idx[:5]
    oof_top5 = H.oof_probabilities(X[:, top5_idx], y, use_smote=False, use_spw=True)
    m5 = H.ranking_metrics(y, oof_top5)
    report["top5_only"] = {"features": ranked[:5], **m5}
    if m5["auroc"] >= 0.97:
        report["concentration_verdict"] = ("CONCENTRATED — a tiny feature cluster reconstructs the "
                                            "label (AUROC>=0.97); inspect these for residual leakage.")
    elif m5["auroc"] >= 0.90:
        report["concentration_verdict"] = ("MIXED — top-5 alone is strong but not decisive.")
    else:
        report["concentration_verdict"] = ("DISTRIBUTED — no small cluster reconstructs the label; "
                                            "the 0.99 full-model AUROC is genuine multivariate signal.")
    log(f"    top-5 AUROC={m5['auroc']}  -> {report['concentration_verdict']}")

    # --- structural finding: temporal split infeasible ---
    report["temporal_split"] = {
        "feasible": False,
        "reason": ("Case-control assembly: all normals are Oct25, all mules are Sep/Nov/Dec25. "
                   "No month contains both classes, so no forward temporal split has both a "
                   "positive and a negative on each side. Temporal validation cannot be built on "
                   "this data and is therefore disclosed, not fabricated."),
    }

    save_json(report, C.REPORTS_DIR / "06_leak_audit.json")
    log("Wrote reports/06_leak_audit.json")


def _permutation_importance(X, y, names, n_repeats=1):
    """OOF-model permutation importance by AUPRC drop.

    Train on all data once (importance ranking, not a performance claim); for each
    feature, permute its column and measure the AUPRC decrease. Higher = the model
    relies on it more. Deterministic RNG seeded from config.
    """
    from sklearn.metrics import average_precision_score

    spw = (y == 0).sum() / max(y.sum(), 1)
    model = H.make_xgb(spw=spw, seed=C.RANDOM_STATE)
    model.fit(X, y)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])

    rng = np.random.default_rng(C.RANDOM_STATE)
    # Only permute features the model actually splits on (importances > 0) for speed.
    booster_imp = model.feature_importances_
    candidate = np.where(booster_imp > 0)[0]
    drops = []
    Xp = X.copy()
    for j in candidate:
        col = Xp[:, j].copy()
        acc = 0.0
        for _ in range(n_repeats):
            rng.shuffle(Xp[:, j])
            acc += base - average_precision_score(y, model.predict_proba(Xp)[:, 1])
            Xp[:, j] = col  # restore
        drops.append((names[j], acc / n_repeats))
    drops.sort(key=lambda t: t[1], reverse=True)
    return [{"feature": f, "auprc_drop": round(float(d), 5)} for f, d in drops]


if __name__ == "__main__":
    main()
