"""Operating points expressed as a review budget rather than a fixed probability.

The deployed freeze threshold is an absolute calibrated probability, 0.998262,
fitted to hit precision 0.99 on inner folds. Two things are wrong with that as a
deployment contract, and both were measured on the real data, not inferred:

  1. It costs recall for nothing. On the same out-of-fold predictions it flags
     0.29 percent of the book at precision 1.000 and recall 0.321, while simply
     reviewing the top 0.5 percent also reaches precision 1.000 and recall
     0.556. The stricter number buys no extra precision and loses a fifth of the
     mules.

  2. It does not transfer. A probability is only meaningful against the score
     distribution it was fitted on. Hand the same number to a book with a
     different base rate and it selects a different share of the portfolio: on a
     300-row extract carrying 5 mules it flagged 3 accounts and caught 2, recall
     0.40, even though the ranking put four of the five mules in the top 12
     percent. The ranking was fine; the cut was imported from elsewhere.

A budget is stated in the units a review team actually has - analyst-days - and
it is invariant to the score distribution, so it survives the move to an extract
we have never seen. What it cannot do is promise a precision in advance, so the
precision each budget bought on our own labelled data is reported alongside it,
out of fold, and that is the honest basis for choosing one.

Run:  python src/operating_point.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import config as C
from utils import log

REPORT = C.REPORTS_DIR / "13_operating_points.json"
BUDGETS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.026, 0.05)
REVIEWS_PER_ANALYST_DAY = 40


def budget_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """What each review budget bought, out of fold, on the labelled data."""
    order = np.argsort(-p)
    n, pos = len(y), int(y.sum())
    rows = []
    for b in BUDGETS:
        k = max(1, int(round(n * b)))
        flag = np.zeros(n, bool)
        flag[order[:k]] = True
        rows.append({
            "budget_pct": round(b * 100, 3),
            "accounts_reviewed": int(k),
            "analyst_days": round(k / REVIEWS_PER_ANALYST_DAY, 1),
            "mules_found": int(y[flag].sum()),
            "of_total_mules": pos,
            "precision": round(float(precision_score(y, flag, zero_division=0)), 4),
            "recall": round(float(recall_score(y, flag)), 4),
            "f1": round(float(f1_score(y, flag, zero_division=0)), 4),
            "score_cut": round(float(p[order[k - 1]]), 6),
        })
    return rows


def main() -> None:
    oof_path = C.DATA_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        log("oof_predictions.csv missing - run the pipeline first.")
        return
    oof = pd.read_csv(oof_path)
    y = oof["y_true"].astype(int).to_numpy()
    p = oof["p_ensemble_calibrated"].to_numpy(float)

    rows = budget_table(y, p)
    log(f"{len(y):,} out-of-fold predictions, {int(y.sum())} mules "
        f"({y.mean()*100:.3f}%)")
    log("  budget   reviewed  found  precision  recall")
    for r in rows:
        log(f"  {r['budget_pct']:>5.2f}%  {r['accounts_reviewed']:>8}  "
            f"{r['mules_found']:>5}  {r['precision']:>9.3f}  {r['recall']:>6.3f}")

    # The budget that matches the deployed threshold's precision, at more recall.
    import joblib
    thr = joblib.load(C.MODELS_DIR / "muleguard_models.joblib")[
        "threshold_precision_first"]
    flag = p >= thr
    fixed = {"threshold": round(float(thr), 6),
             "accounts_flagged": int(flag.sum()),
             "share_of_book_pct": round(float(flag.mean()) * 100, 3),
             "precision": round(float(precision_score(y, flag, zero_division=0)), 4),
             "recall": round(float(recall_score(y, flag)), 4)}
    better = [r for r in rows if r["precision"] >= fixed["precision"]
              and r["recall"] > fixed["recall"]]
    log(f"\n  deployed absolute threshold {thr:.6f}: "
        f"precision {fixed['precision']:.3f} recall {fixed['recall']:.3f}")
    if better:
        b = max(better, key=lambda r: r["recall"])
        log(f"  a top {b['budget_pct']}% budget matches that precision at "
            f"recall {b['recall']:.3f} - {b['mules_found']} mules instead of "
            f"{int((p >= thr).sum() and y[flag].sum())}")

    REPORT.write_text(json.dumps({
        "question": "Should the operating point be an absolute probability or a "
                    "review budget?",
        "validation": "nested repeated CV out-of-fold predictions on the real "
                      "labelled data; no synthetic extract informed any of this",
        "n_accounts": int(len(y)), "n_mules": int(y.sum()),
        "base_rate": round(float(y.mean()), 6),
        "reviews_per_analyst_day": REVIEWS_PER_ANALYST_DAY,
        "deployed_absolute_threshold": fixed,
        "budgets": rows,
        "recommendation":
            "Quote a budget, not a probability. The absolute threshold reaches "
            "precision 1.000 at recall 0.321 while the top 0.5 percent reaches "
            "precision 1.000 at recall 0.556, so the stricter cut buys no "
            "precision and loses mules. A budget is also invariant to the score "
            "distribution, which an absolute probability is not: the same number "
            "applied to an extract with a different base rate selects a "
            "different share of the book.",
        "caveat":
            "A budget cannot promise a precision on an unseen extract. The "
            "precision column here is what each budget bought on OUR labelled "
            "data, out of fold, and it is the basis for choosing a budget - not "
            "a guarantee about someone else's.",
    }, indent=1), encoding="utf-8")
    log(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
