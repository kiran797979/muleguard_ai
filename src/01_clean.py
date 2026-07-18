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


def _is_categorical_leak(s: pd.Series, y: pd.Series) -> bool:
    """True if category membership almost perfectly predicts the label.

    Measured as the fractional reduction in majority-class error from
    conditioning on the category. A value near 1.0 means the column essentially
    reproduces the target (e.g. a dataset-assembly artifact) rather than
    carrying a generalizable signal, so it should be dropped as a leak.
    """
    base_err = min(y.mean(), 1 - y.mean())
    if base_err <= 0:
        return False
    g = pd.DataFrame({"cat": s.values, "y": y.values})
    cond_err = (g.groupby("cat")["y"]
                .apply(lambda v: min(v.mean(), 1 - v.mean()) * len(v)).sum()) / len(g)
    err_reduction = (base_err - cond_err) / base_err
    return bool(err_reduction >= C.CATEGORICAL_LEAK_ERROR_REDUCTION)


def encode_categoricals(df: pd.DataFrame, protect: list[str]) -> tuple[pd.DataFrame, dict]:
    """Encode genuine string columns instead of coercing them to NaN.

    The real dataset has string columns (account type, occupation, gender,
    open-date, ...) that plain numeric coercion would silently wipe out — the
    richest semantic signal in the data. We encode them with generic, honest,
    leakage-safe rules that make no guess about which column means what:

      * A string column that is really numeric-with-noise stays numeric.
      * Low-cardinality (<= CATEGORICAL_MAX_CARDINALITY distinct) -> one-hot.
      * High-cardinality that parses as dates -> numeric 'vintage' (days since
        the most recent date in the column; older account => larger value).
      * Anything else high-cardinality (IDs / free text) -> dropped, logged.
    """
    info: dict = {"one_hot": {}, "date_vintage": [], "dropped_highcard": [],
                  "kept_numeric": [], "dropped_leak": []}
    obj_cols = [c for c in df.columns if c not in protect and df[c].dtype == object]
    y = df[C.TARGET_COL] if C.TARGET_COL in df.columns else None

    for col in obj_cols:
        s = df[col]
        # 1. Truly numeric strings -> keep as numeric.
        as_num = pd.to_numeric(s, errors="coerce")
        if as_num.notna().mean() >= C.DATE_PARSE_MIN_FRAC:
            df[col] = as_num
            info["kept_numeric"].append(col)
            continue

        nun = s.nunique(dropna=True)
        # 2. Low-cardinality -> one-hot dummies (drop original).
        if nun <= C.CATEGORICAL_MAX_CARDINALITY:
            # Leak guard: drop the WHOLE column before encoding if the category
            # almost perfectly determines the label (else a below-threshold
            # subset of dummies could still smuggle the leak past Stage 4).
            if y is not None and _is_categorical_leak(s, y):
                df = df.drop(columns=[col])
                info["dropped_leak"].append({"col": col, "cardinality": int(nun)})
                continue
            dummies = pd.get_dummies(s, prefix=col, dummy_na=False, dtype=float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            info["one_hot"][col] = list(dummies.columns)
            continue

        # 3. High-cardinality: try dates -> numeric vintage.
        #    Try both day/month orderings; keep whichever parses more rows.
        dt_mdy = pd.to_datetime(s, errors="coerce", dayfirst=False)
        dt_dmy = pd.to_datetime(s, errors="coerce", dayfirst=True)
        dt = dt_mdy if dt_mdy.notna().mean() >= dt_dmy.notna().mean() else dt_dmy
        if dt.notna().mean() >= C.DATE_PARSE_MIN_FRAC:
            ref = dt.max()
            df[col] = (ref - dt).dt.days.astype(float)  # days-before-latest
            info["date_vintage"].append(col)
            continue

        # 4. Otherwise: ID-like / free text -> drop.
        df = df.drop(columns=[col])
        info["dropped_highcard"].append({"col": col, "cardinality": int(nun)})

    return df, info


def drop_sparse(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    miss_frac = df.drop(columns=[target]).isna().mean()
    sparse = miss_frac[miss_frac > C.MISSING_DROP_FRAC].index.tolist()
    return df.drop(columns=sparse), sparse


def detect_leaks(df: pd.DataFrame, target: str) -> tuple[list[str], dict]:
    """Return columns whose |corr| with target exceeds the leak threshold."""
    feats = df.drop(columns=[target])
    y = df[target]
    # Pearson corr of each numeric feature with the target. Zero-variance columns
    # yield a 0/0 = NaN correlation (dropped below); silence that expected warning.
    with np.errstate(divide="ignore", invalid="ignore"):
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

    # The cleaning logic now lives in a fittable, persistable Preprocessor so the
    # exact transform (column schema, medians, one-hot vocab, date reference, drop
    # lists) can be replayed on genuinely new accounts at inference time — closing
    # the previous no-inference-path gap. fit_transform reproduces the identical
    # cleaned matrix (locked bit-exact by tests/test_preprocess.py), so no measured
    # number changes.
    from preprocess import Preprocessor, PREPROCESSOR_PATH

    pre = Preprocessor()
    df, report = pre.fit_transform(df)

    cat_info = report["categorical_encoding"]
    n_oh = sum(len(v) for v in cat_info["one_hot"].values())
    log(f"Dropped {len(report['dropped_id_columns'])} identifier column(s): "
        f"{report['dropped_id_columns']}")
    log(f"Class balance: {report['class_balance']['positives(mules)']} mules / "
        f"{report['class_balance']['negatives(normal)']} normal "
        f"({report['class_balance']['prevalence_pct']}% prevalence)")
    log(f"Categoricals: {len(cat_info['one_hot'])} one-hot cols -> {n_oh} dummies; "
        f"{len(cat_info['date_vintage'])} date->vintage; "
        f"{len(cat_info['dropped_highcard'])} high-card dropped; "
        f"{len(cat_info['kept_numeric'])} kept numeric")
    if cat_info["dropped_leak"]:
        leak_cols = [d["col"] for d in cat_info["dropped_leak"]]
        log(f"Categorical LEAK guard dropped {len(leak_cols)} column(s): {leak_cols} "
            f"(category almost perfectly determines the label)")
    log(f"Dropped {report['dropped_sparse_count']} columns missing "
        f">{int(C.MISSING_DROP_FRAC*100)}%")
    log(f"Removed {len(report['removed_leak_columns'])} suspected leak column(s): "
        f"{report['removed_leak_columns']}")
    log(f"Dropped {report['dropped_collinear_count']} constant/near-duplicate columns")
    log(f"Clean matrix: {df.shape[0]:,} rows x {report['clean_feature_count']:,} "
        f"features + target")

    save_frame(df, C.CLEAN_PARQUET)
    save_json(report, C.REPORTS_DIR / "01_clean_report.json")

    # Persist the fitted transform for the inference path (src/score_new.py).
    pre.save(PREPROCESSOR_PATH)
    log(f"Saved fitted preprocessor -> {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
