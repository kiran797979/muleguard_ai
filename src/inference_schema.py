"""The exact input contract of the deployed ensemble, and a transform that meets it.

Why this exists
---------------
A test extract kept arriving with 18 columns and the evaluator kept declining to
use the deployed model on it. That was read as an export bug. It is not: the
model is fitted on 1,506 features and an 18-column file does not contain the
information to compute them. The features are not being dropped somewhere, they
were never present in the source. 1,506 cannot be recovered from 18 by any
amount of preprocessing, and a pipeline that appeared to do so would be
fabricating values and calling them measurements.

What a conforming file actually needs
-------------------------------------
The 1,506 columns are 1,439 raw fields carried through from the extract plus 67
engineered from them. So a file conforms if it carries the raw fields, under
either their F-code or their data-dictionary name. `describe()` writes that list
out; `transform()` maps a raw extract onto the model matrix and, crucially,
REPORTS what it could not build rather than quietly imputing it.

Coverage is the number to watch. Masked to 750 of 1,506 the ensemble scores
AUPRC 0.937; masked to 300 it scores 0.009, which is the random baseline. Below
half the schema its output is not a weak signal, it is noise wearing a
probability, which is why the evaluator refuses rather than obliges.

Run:  python src/inference_schema.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
import dictionary as D
from utils import log

OUT = C.REPORTS_DIR / "inference_schema.json"
CSV = C.REPORTS_DIR / "inference_schema.csv"


def _bundle():
    import joblib
    return joblib.load(C.MODEL_PATH if hasattr(C, "MODEL_PATH")
                       else C.MODELS_DIR / "muleguard_models.joblib")


def describe() -> pd.DataFrame:
    """One row per model input: position, F-code, banking name, source, median.

    The median is the value the ensemble falls back on when a column is absent,
    so it belongs in the contract: it is what a missing column silently becomes.
    """
    cols = _bundle()["feat_names"]
    train = pd.read_parquet(C.DATA_DIR / "features.parquet")
    rows = []
    for i, c in enumerate(cols):
        engineered = str(c).startswith("mg_")
        med = np.nan
        if c in train.columns:
            med = pd.to_numeric(train[c], errors="coerce").median()
        rows.append({
            "position": i,
            "column": c,
            "banking_name": c if engineered else D.real_name(c),
            "source": "engineered from raw fields" if engineered
                      else "carried through from the extract",
            "training_median": None if pd.isna(med) else round(float(med), 6),
        })
    return pd.DataFrame(rows)


def transform(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Map a raw extract onto the model matrix, in the fitted column order.

    Accepts F-codes or data-dictionary names. Returns the matrix and a report
    naming every column that could not be filled, because a caller deciding
    whether to trust a score needs to know how much of it is imputation.
    """
    cols = _bundle()["feat_names"]
    index = {c: i for i, c in enumerate(cols)}
    by_real = {}
    for c in cols:
        if not str(c).startswith("mg_"):
            by_real.setdefault(str(D.real_name(c)).strip().upper(), c)

    X = np.full((len(df), len(cols)), np.nan, dtype=np.float32)
    filled, unknown = {}, []
    for src in df.columns:
        tgt = src if src in index else by_real.get(str(src).strip().upper())
        if tgt is None:
            unknown.append(str(src)[:64])
            continue
        if tgt in filled:
            continue
        X[:, index[tgt]] = pd.to_numeric(df[src], errors="coerce").to_numpy(np.float32)
        filled[tgt] = src

    missing = [c for c in cols if c not in filled]
    coverage = len(filled) / len(cols)
    return X, {
        "rows": int(len(df)),
        "model_expects": len(cols),
        "columns_filled": len(filled),
        "coverage_pct": round(coverage * 100, 2),
        "usable_with_deployed_ensemble": bool(coverage >= 0.50),
        "columns_absent": len(missing),
        "columns_absent_sample": [
            f"{c} ({D.real_name(c)})" if not str(c).startswith("mg_") else c
            for c in missing[:15]],
        "columns_not_recognised": unknown[:15],
        "note": "Absent columns are left NaN and the ensemble substitutes the "
                "training median. Below 50 percent coverage that substitution "
                "dominates the input and the output stops being a measurement: "
                "masked to 300 of 1,506 the ensemble scores at the random "
                "baseline. Coverage is the number to check before trusting a "
                "score, not the score itself.",
    }


def main() -> None:
    if not (C.DATA_DIR / "features.parquet").exists():
        log("features.parquet missing - run the pipeline first.")
        return
    d = describe()
    d.to_csv(CSV, index=False)
    eng = int(d["source"].str.startswith("engineered").sum())
    OUT.write_text(json.dumps({
        "n_features": int(len(d)),
        "carried_from_extract": int(len(d) - eng),
        "engineered": eng,
        "column_order": d["column"].tolist(),
        "contract": "A conforming extract carries the raw fields under either "
                    "their F-code or their data-dictionary name. The engineered "
                    "columns are derived from those, not supplied.",
        "coverage_floor": 0.50,
        "why_the_floor": "Masked to 750 of 1,506 the ensemble scores AUPRC "
                         "0.937; masked to 300 it scores 0.009 against a 0.0089 "
                         "base rate.",
    }, indent=1), encoding="utf-8")
    log(f"{len(d)} inference columns: {len(d)-eng} carried, {eng} engineered")
    log(f"wrote {CSV}")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
