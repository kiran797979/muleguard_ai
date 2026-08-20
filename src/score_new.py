"""
MuleGuard — score genuinely NEW accounts from a raw CSV (real inference path).

Before this, the saved model could only be re-applied to the same training rows
(05_score_explain.py re-ran SHAP on data/features.parquet). There was no way to
take an unseen account and produce a risk score, because the cleaning transform
state was never persisted.

This module closes that gap end-to-end:

    raw CSV  ->  Preprocessor.transform (fitted state)  ->  add mg_* row features
             ->  align to the model's trained feature order  ->  calibrated prob
             ->  0-1000 risk score + LOW/MEDIUM/HIGH band + action

It reuses the SAME row-profile feature definitions as Stage 2 (02_features.py) and
the SAME banding/score mapping as Stage 7/8 (05_score_explain.py), so a new account
is treated identically to a training account — no shortcut, no leakage.

Usage:
    from score_new import score_new
    df = score_new("path/to/new_accounts.csv")   # -> DataFrame(risk_score, band, action, prob)

    # or CLI:
    .venv/bin/python src/score_new.py path/to/new_accounts.csv [out.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from preprocess import Preprocessor, PREPROCESSOR_PATH
from utils import die, log

# Row-profile feature names + order must match 02_features.py exactly.
MG_FEATURES = [
    "mg_row_mean", "mg_row_std", "mg_row_max", "mg_row_min",
    "mg_row_nonzero_frac", "mg_row_skew", "mg_row_range",
]


def add_row_profile_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Replicate Stage-2 mg_* features on the cleaned feature columns (leakage-safe)."""
    X = df[feat_cols]
    df["mg_row_mean"] = X.mean(axis=1)
    df["mg_row_std"] = X.std(axis=1)
    df["mg_row_max"] = X.max(axis=1)
    df["mg_row_min"] = X.min(axis=1)
    df["mg_row_nonzero_frac"] = (X != 0).mean(axis=1)
    df["mg_row_skew"] = X.skew(axis=1)
    df["mg_row_range"] = df["mg_row_max"] - df["mg_row_min"]
    df[MG_FEATURES] = df[MG_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    return df


def _band(score: float) -> str:
    if score <= C.BAND_LOW_MAX:
        return "LOW"
    if score <= C.BAND_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def score_new(csv_path: str | Path, threshold: float | None = None) -> pd.DataFrame:
    """Score new raw accounts. Returns a DataFrame with prob, risk_score, band, action.

    The preprocessor and model bundle must already exist (produced by a pipeline run).
    """
    import joblib

    csv_path = Path(csv_path)
    if not csv_path.exists():
        die(f"Input CSV not found: {csv_path}")
    if not PREPROCESSOR_PATH.exists():
        die(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run the pipeline (Stage 1) first.")
    model_path = C.MODELS_DIR / "muleguard_models.joblib"
    if not model_path.exists():
        die(f"Model bundle not found at {model_path}. Run Stage 4/5 first.")

    pre = Preprocessor.load(PREPROCESSOR_PATH)
    bundle = joblib.load(model_path)
    feat_names = bundle["feat_names"]
    xgb = bundle.get("xgb") or bundle.get("lgbm")
    if xgb is None:
        die("No tree model in bundle to score with.")
    iso_cal = bundle.get("iso_calibrator")

    raw = pd.read_csv(csv_path, low_memory=False)
    log(f"Scoring {len(raw):,} new account(s) from {csv_path.name}")

    # 1. Fitted cleaning transform (no target, nothing re-derived).
    clean = pre.transform(raw)
    # 2. Stage-2 row-profile features on the cleaned columns.
    clean = add_row_profile_features(clean, list(pre.feature_columns))
    # 3. Align to the model's trained feature order (missing -> 0, extra -> dropped).
    X = clean.reindex(columns=feat_names).fillna(0.0).values.astype(np.float32)

    # 4. Raw model probability -> isotonic calibration (same calibrator as training).
    raw_prob = xgb.predict_proba(X)[:, 1]
    prob = iso_cal.transform(raw_prob) if iso_cal is not None else raw_prob

    score = np.clip(prob, 0, 1) * C.SCORE_MAX
    bands = np.array([_band(s) for s in score])
    out = pd.DataFrame({
        "prob": np.round(prob, 6),
        "risk_score": np.round(score).astype(int),
        "band": bands,
        "action": [C.BAND_ACTIONS.get(b, "") for b in bands],
    })
    if threshold is not None:
        out["flag"] = (prob >= threshold).astype(int)
    return out


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        die("Usage: python src/score_new.py <new_accounts.csv> [out.csv]")
    result = score_new(argv[0])
    if len(argv) > 1:
        result.to_csv(argv[1], index=False)
        log(f"Wrote scores -> {argv[1]}")
    else:
        # Print a compact preview.
        log(f"Scored {len(result)} accounts. Band counts: "
            f"{result['band'].value_counts().to_dict()}")
        print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
