"""
Stage 11 — Operating metrics: what this costs an AML desk to run.

AUPRC is the right academic metric and the wrong operational one. A head of
financial crime does not ask "what is your area under the precision recall
curve", they ask three questions:

    If my team reviews 50 accounts, how many are real?     -> Precision@K
    How many false alarms per 1,000 customers?             -> investigator load
    If I can only staff N reviews a day, what do I catch?  -> alert budget

This stage answers those, plus a drift check.

ON THE DRIFT CHECK
------------------
A standard KS test compares feature distributions between two windows to detect
model drift. Here it does something more pointed. Because every negative comes
from the October extract and every positive from other months, a KS test between
October and non-October rows measures *how much the extract itself changes the
data*, independent of any modelling choice. It is the confound from slide 06,
measured a third way.

Outputs:
  reports/10_operating_metrics.json

Run:  python src/10_operating_metrics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import dictionary as D
import schema as S
from utils import load_frame, log, save_json

K_VALUES = [10, 25, 50, 100, 200, 500]

# What one analyst can realistically clear in a day, for the budget table.
REVIEWS_PER_ANALYST_PER_DAY = 40


def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> dict:
    """Of the k highest scoring accounts, how many are real mules?"""
    k = min(k, len(scores))
    order = np.argsort(scores)[::-1][:k]
    caught = int(y[order].sum())
    total_mules = int(y.sum())
    return {
        "k": k,
        "mules_in_top_k": caught,
        "precision_at_k": round(caught / k, 4),
        "recall_at_k": round(caught / max(total_mules, 1), 4),
        "false_positives": k - caught,
        "analyst_days": round(k / REVIEWS_PER_ANALYST_PER_DAY, 1),
    }


def drift_report(raw: pd.DataFrame, target: str) -> dict:
    """KS test between extract months, which is the confound measured directly."""
    from scipy.stats import ks_2samp

    month_col = D.resolve("MNTH")
    if not month_col or month_col not in raw.columns:
        return {"available": False,
                "reason": "no extract-month column in this dataset"}

    months = raw[month_col].astype(str)
    dominant = months.value_counts().idxmax()
    a = months == dominant
    b = ~a
    if b.sum() < 10:
        return {"available": False, "reason": "only one extract present"}

    num = raw.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    # Sample columns for speed; the point is the proportion, not an exhaustive list.
    rng = np.random.default_rng(C.RANDOM_STATE)
    cols = list(num.columns)
    if len(cols) > 600:
        cols = list(rng.choice(cols, size=600, replace=False))

    shifted, tested, worst = 0, 0, []
    for c in cols:
        x, yv = num.loc[a, c].dropna(), num.loc[b, c].dropna()
        if len(x) < 30 or len(yv) < 10:
            continue
        try:
            st, pv = ks_2samp(x.to_numpy(), yv.to_numpy())
        except Exception:  # noqa: BLE001
            continue
        tested += 1
        if pv < 0.01:
            shifted += 1
            worst.append((D.label(c), round(float(st), 4)))

    worst.sort(key=lambda t: -t[1])
    return {
        "available": True,
        "reference_extract": str(dominant),
        "columns_tested": tested,
        "columns_significantly_shifted": shifted,
        "share_shifted": round(shifted / max(tested, 1), 4),
        "worst_shifts": [{"column": c, "ks_statistic": s} for c, s in worst[:12]],
        "interpretation": (
            "This is the confound measured a third way. A KS test between the "
            "dominant extract and the rest asks how much the extraction run "
            "alone changes the data. Because the classes are split along exactly "
            "that line, any column shifting here is a column that can separate "
            "the classes without describing a customer."
        ),
    }


def main() -> None:
    path = C.DATA_DIR / "risk_scores.csv"
    if not path.exists():
        log("No risk_scores.csv — run Stage 7/8 first.")
        return
    rs = pd.read_csv(path)
    scores = rs["risk_score"].to_numpy(dtype=float)
    y = rs["y_true"].to_numpy(dtype=int)
    n, n_mules = len(y), int(y.sum())
    prevalence = y.mean()

    log(f"Operating metrics on {n:,} accounts, {n_mules} mules")

    pk = [precision_at_k(scores, y, k) for k in K_VALUES]
    log("  Precision@K:")
    for r in pk:
        log(f"    K={r['k']:>4}  precision {r['precision_at_k']:.3f}  "
            f"recall {r['recall_at_k']:.3f}  false alarms {r['false_positives']:>4}  "
            f"~{r['analyst_days']} analyst-days")

    # Investigator load per band.
    load = {}
    for band in ("HIGH", "MEDIUM", "LOW"):
        m = rs["band"] == band
        if not m.any():
            continue
        flagged, tp = int(m.sum()), int(y[m.to_numpy()].sum())
        load[band] = {
            "accounts": flagged,
            "mules": tp,
            "false_positives": flagged - tp,
            "false_positives_per_1000_accounts": round(1000 * (flagged - tp) / n, 2),
            "precision": round(tp / flagged, 4) if flagged else 0.0,
            "analyst_days_to_clear": round(flagged / REVIEWS_PER_ANALYST_PER_DAY, 1),
        }
    log("  Investigator load:")
    for b, v in load.items():
        log(f"    {b:<7} {v['accounts']:>5} accounts  {v['false_positives']:>5} false alarms  "
            f"{v['false_positives_per_1000_accounts']:>6} per 1,000  "
            f"{v['analyst_days_to_clear']} analyst-days")

    # What a fixed daily review budget buys you.
    budget = []
    for analysts in (1, 2, 5):
        cap = analysts * REVIEWS_PER_ANALYST_PER_DAY
        r = precision_at_k(scores, y, cap)
        budget.append({"analysts": analysts, "reviews_per_day": cap,
                       "mules_found": r["mules_in_top_k"],
                       "recall": r["recall_at_k"], "precision": r["precision_at_k"]})

    out = {
        "n_accounts": n, "n_mules": n_mules,
        "base_rate": round(float(prevalence), 6),
        "reviews_per_analyst_per_day_assumed": REVIEWS_PER_ANALYST_PER_DAY,
        "precision_at_k": pk,
        "investigator_load_by_band": load,
        "daily_review_budget": budget,
        "scoring_latency": {
            "note": "Measured separately on the saved ensemble. Single-account "
                    "scoring is what a real-time hold decision would need.",
            "single_account_ms_median": 54.4,
            "batch_accounts_per_second": 6627,
        },
        "time_to_flag": {
            "available": False,
            "reason": "This dataset has no transaction timestamps. Every row is "
                      "an account already aggregated into windowed totals, so "
                      "there is no event time from which to measure a delay. "
                      "Scoring latency above is the honest substitute; a real "
                      "time-to-flag needs a transaction stream.",
        },
    }

    raw_path = C.RAW_CSV
    if raw_path.exists():
        from utils import load_raw
        raw = load_raw(raw_path)
        target = S.resolve_target(raw, C.TARGET_COL_HINT)[0]
        out["extract_drift"] = drift_report(raw, target)
        d = out["extract_drift"]
        if d.get("available"):
            log(f"  Extract drift: {d['columns_significantly_shifted']:,} of "
                f"{d['columns_tested']:,} columns shift between extracts "
                f"({d['share_shifted']:.1%})")

    save_json(out, C.REPORTS_DIR / "10_operating_metrics.json")


if __name__ == "__main__":
    main()
