"""
Stage 1 — Data cleaning & target-leak removal.

Steps (in order):
  1. Load raw DataSet.csv.
  2. Coerce all feature columns to numeric where possible.
  3. Drop columns missing more than MISSING_DROP_FRAC of values.
  4. Correlation scan: flag/remove any column whose |Pearson corr| with the
     target exceeds LEAK_CORR_THRESHOLD (this is how F3912 is caught).
  5. De-duplicate near-identical (collinear) columns — the rolling-window copies.
  6. Median-impute remaining missing values.
  7. Save the clean feature matrix + a JSON cleaning report.

Run:  python src/01_clean.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from utils import load_raw, log, save_frame, save_json


def coerce_numeric(df: pd.DataFrame, protect: list[str]) -> pd.DataFrame:
    """Convert object columns to numeric where sensible; leave protected cols."""
    for col in df.columns:
        if col in protect:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_sparse(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    miss_frac = df.drop(columns=[target]).isna().mean()
    sparse = miss_frac[miss_frac > C.MISSING_DROP_FRAC].index.tolist()
    return df.drop(columns=sparse), sparse


def detect_leaks(df: pd.DataFrame, target: str) -> tuple[list[str], dict]:
    """Return columns whose |corr| with target exceeds the leak threshold."""
    feats = df.drop(columns=[target])
    y = df[target]
    # Pearson corr of each numeric feature with the target.
    corrs = feats.corrwith(y, numeric_only=True).abs().dropna()
    leaks = corrs[corrs > C.LEAK_CORR_THRESHOLD].sort_values(ascending=False)
    # Always report the top correlations for transparency.
    top = corrs.sort_values(ascending=False).head(15).round(4).to_dict()
    return leaks.index.tolist(), top


def drop_collinear(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Remove one of each pair of near-duplicate columns (rolling-window copies)."""
    feats = df.drop(columns=[target])
    # Work on a variance-filtered set to keep the corr matrix tractable.
    nunique = feats.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    feats = feats.drop(columns=constant)

    corr = feats.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dup = [c for c in upper.columns if (upper[c] > C.COLLINEAR_CORR_THRESHOLD).any()]

    to_drop = sorted(set(constant + dup))
    return df.drop(columns=to_drop), to_drop


def main() -> None:
    df = load_raw(C.RAW_CSV)

    if C.TARGET_COL not in df.columns:
        from utils import die
        die(f"Target column {C.TARGET_COL} not found. Columns look like: "
            f"{list(df.columns[:5])} ... total {df.shape[1]}")

    report: dict = {"input_shape": list(df.shape)}

    # Class balance snapshot
    y = df[C.TARGET_COL]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    report["class_balance"] = {
        "positives(mules)": n_pos,
        "negatives(normal)": n_neg,
        "prevalence_pct": round(100 * n_pos / len(y), 4),
    }
    log(f"Class balance: {n_pos} mules / {n_neg} normal "
        f"({report['class_balance']['prevalence_pct']}% prevalence)")

    # 2. Coerce numeric
    df = coerce_numeric(df, protect=[C.TARGET_COL])

    # 3. Drop sparse columns
    df, sparse = drop_sparse(df, C.TARGET_COL)
    report["dropped_sparse_count"] = len(sparse)
    log(f"Dropped {len(sparse)} columns missing >{int(C.MISSING_DROP_FRAC*100)}%")

    # 4. Leak detection (F3912 should surface here)
    leaks, top_corr = detect_leaks(df, C.TARGET_COL)
    report["top_target_correlations"] = top_corr
    # Ensure the known leak column is removed even if just under threshold.
    if C.LEAK_COL in df.columns and C.LEAK_COL not in leaks:
        leaks.append(C.LEAK_COL)
    leaks = [c for c in leaks if c in df.columns]
    df = df.drop(columns=leaks)
    report["removed_leak_columns"] = leaks
    log(f"Removed {len(leaks)} suspected leak column(s): {leaks}")

    # 5. De-duplicate collinear rolling-window copies
    df, collinear = drop_collinear(df, C.TARGET_COL)
    report["dropped_collinear_count"] = len(collinear)
    log(f"Dropped {len(collinear)} constant/near-duplicate columns")

    # 6. Median impute remaining NaNs
    feat_cols = [c for c in df.columns if c != C.TARGET_COL]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    # Any column still all-NaN (no median) -> fill 0 and note it.
    still_na = [c for c in feat_cols if df[c].isna().any()]
    if still_na:
        df[still_na] = df[still_na].fillna(0)
    report["output_shape"] = list(df.shape)
    report["clean_feature_count"] = len(feat_cols)

    log(f"Clean matrix: {df.shape[0]:,} rows x {len(feat_cols):,} features + target")

    save_frame(df, C.CLEAN_PARQUET)
    save_json(report, C.REPORTS_DIR / "01_clean_report.json")


if __name__ == "__main__":
    main()
