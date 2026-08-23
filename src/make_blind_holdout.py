"""Carve a blind 213-row hold-out, then refit on what is left.

A hold-out is only blind if the model never saw it. The deployed ensemble is
refit on all 9,082 rows, so scoring any of those rows with it measures memory,
not generalisation. This script removes the hold-out FIRST and fits a fresh
model on the remainder, which is the only order that produces an honest number.

Composition is 13 positives and 200 negatives, matching the shape of the blind
files supplied for testing: a 6.1 percent base rate, seven times the 0.89
percent of the source data. That makes the set small enough to review by hand
and is deliberately NOT the rate a real book carries.

WHAT THIS SET CAN AND CANNOT TELL YOU
-------------------------------------
It cannot escape the confound of Section III. Every negative in the source data
comes from the October extract and every positive from September, November or
December, so a hold-out drawn from it inherits that structure: any model can
still separate the classes by recognising which extraction run a row came from.
The number this produces is therefore an UPPER BOUND on performance against a
hidden validation set, not an estimate of it. If the hidden set draws both
classes from one extract, expect the behavioural figure instead.

The positives are spread across all three of their months so the set at least
does not test a single extract in isolation.

Run:  python src/make_blind_holdout.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import config as C
from utils import log

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

OUT_DIR = C.REPORTS_DIR / "blind_holdout"
N_POS, N_NEG, SEED = 13, 200, 20260823
MONTH_COL = "F2230"


def _model(y: np.ndarray):
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        n_jobs=4, verbose=-1, random_state=42)


def main() -> None:
    if lgb is None:
        log("lightgbm is not installed."); return
    feats = C.DATA_DIR / "features.parquet"
    if not feats.exists():
        log("features.parquet missing - run the pipeline first."); return

    X = pd.read_parquet(feats)
    raw = pd.read_csv(C.RAW_CSV, low_memory=False)
    y = raw[C.TARGET_COL_HINT].astype(int).to_numpy()
    cols = [c for c in X.columns if c != C.TARGET_COL_HINT]
    log(f"source {X.shape[0]:,} rows x {len(cols)} features, {int(y.sum())} positives")

    rng = np.random.default_rng(SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Spread the positives across their extract months rather than taking 13
    # from whichever month happens to sort first.
    if MONTH_COL in raw.columns:
        months = raw[MONTH_COL].astype(str).to_numpy()
        chosen: list[int] = []
        for m in sorted(set(months[pos_idx])):
            pool = pos_idx[months[pos_idx] == m]
            take = max(1, round(N_POS * len(pool) / len(pos_idx)))
            chosen += list(rng.choice(pool, size=min(take, len(pool)), replace=False))
        chosen = chosen[:N_POS]
        while len(chosen) < N_POS:
            spare = [i for i in pos_idx if i not in chosen]
            chosen.append(int(rng.choice(spare)))
        hold_pos = np.array(sorted(chosen))
        log("  positives drawn per month: " +
            ", ".join(f"{m} {int((months[hold_pos] == m).sum())}"
                      for m in sorted(set(months[hold_pos]))))
    else:
        hold_pos = rng.choice(pos_idx, size=N_POS, replace=False)

    hold_neg = rng.choice(neg_idx, size=N_NEG, replace=False)
    hold = np.concatenate([hold_pos, hold_neg])
    rng.shuffle(hold)
    train = np.setdiff1d(np.arange(len(y)), hold)
    log(f"  hold-out {len(hold)} rows ({N_POS} positive, {N_NEG} negative); "
        f"training on the remaining {len(train):,}")

    model = _model(y[train]).fit(X.iloc[train][cols].to_numpy(np.float32), y[train])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blind = X.iloc[hold][cols].reset_index(drop=True)
    blind.insert(0, "row_number", np.arange(1, len(hold) + 1))
    blind.to_csv(OUT_DIR / "blind_213_features.csv", index=False)
    pd.DataFrame({"row_number": np.arange(1, len(hold) + 1),
                  "is_mule": y[hold]}).to_csv(
        OUT_DIR / "blind_213_labels.csv", index=False)
    joblib.dump({"model": model, "columns": cols},
                OUT_DIR / "model_fitted_without_holdout.joblib")

    (OUT_DIR / "README.md").write_text(
        "# Blind hold-out, 213 rows\n\n"
        f"`blind_213_features.csv` carries {len(cols)} features and no label.\n"
        "`blind_213_labels.csv` carries the answers, keyed by `row_number`.\n"
        "`model_fitted_without_holdout.joblib` never saw these rows.\n\n"
        "Score the features, then join on `row_number` to grade.\n\n"
        "## Read the result as an upper bound\n\n"
        "Every negative in the source data comes from the October extract and "
        "every positive from September, November or December. A hold-out drawn "
        "from it inherits that, so a model can still separate the classes by "
        "recognising the extraction run. Treat the score as a ceiling on what a "
        "hidden validation set would give, not an estimate of it.\n",
        encoding="utf-8")

    log(f"wrote {OUT_DIR}/blind_213_features.csv  (labels held separately)")
    json.dump({
        "n_rows": len(hold), "n_positive": int(N_POS), "n_negative": int(N_NEG),
        "base_rate": round(float(N_POS / len(hold)), 4),
        "n_features": len(cols), "seed": SEED,
        "trained_on": int(len(train)),
        "construction": "hold-out removed before fitting, so the model has "
                        "never seen these rows",
        "caveat": "inherits the extract confound of Section III; an upper bound, "
                  "not an estimate",
    }, open(OUT_DIR / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
