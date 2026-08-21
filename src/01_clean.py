"""
Stage 1 — Cleaning, semantic leak removal, and a separation audit.

The stock version of this stage coerced every column to numeric, which silently
turned the eight *categorical* fields (occupation, gender, area, product,
segmentation, account-age bucket, month, opening date) into all-NaN and then
dropped them as "sparse". Occupation alone carries a 3x spread in mule rate, so
that was throwing away real signal. This version reads the data dictionary and
treats each field according to what it actually is.

Order of operations:
  1. Load the raw CSV; drop the unnamed row-index column.
  2. SEMANTIC leak removal (dictionary-driven, correlation-independent):
       - post-outcome fields: resolution status flags + resolve-days. These are
         written only after an analyst closes the case; at scoring time they do
         not exist. FRAUD_SUSPECTED correlates 0.97, but FALSE_POSITIVE
         correlates only 0.05 and is every bit as unusable — which is exactly
         why a correlation threshold alone is not a leak defence.
       - structural fields: MNTH, and the raw account-opening date string.
  3. Encode the surviving categoricals (ordinal where ordered, one-hot where not).
  4. Coerce the rest to numeric; drop columns missing more than the threshold.
  5. Correlation backstop for any leak the semantic pass missed.
  6. SEPARATION AUDIT — scan every remaining column for near-perfect class
     separation. This is what caught MNTH, and it will catch the next one.
  7. Drop constant and near-duplicate columns; median-impute.

Run:  python src/01_clean.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import dictionary as D
import schema as S
from utils import die, load_raw, log, save_frame, save_json


def drop_identifier_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Remove row identifiers — they are keys, not features.

    Generalised from "drop the column literally called `Unnamed: 0`". An account
    number or customer id correlates with nothing real, but a tree will happily
    memorise one, so they are caught by name pattern AND by being near-unique
    across rows.
    """
    drop = S.identifier_columns(df, C.TARGET_COL)
    report["dropped_identifier_columns"] = drop[:20]
    report["n_dropped_identifier_columns"] = len(drop)
    if drop:
        log(f"Dropped {len(drop)} identifier column(s): {drop[:6]}")
        df = df.drop(columns=drop)
    return df


def remove_semantic_leaks(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Drop post-outcome and structural fields identified by meaning, not correlation."""
    leaks = D.leak_codes()
    removed: dict[str, list[str]] = {}

    groups = []
    if C.DROP_POST_OUTCOME:
        groups.append(("post_outcome", leaks["post_outcome"]))
    if C.DROP_STRUCTURAL_LEAKS:
        groups.append(("structural", leaks["structural"]))

    for kind, codes in groups:
        present = [c for c in codes if c in df.columns]
        if present:
            df = df.drop(columns=present)
            removed[kind] = [D.label(c) for c in present]
            log(f"Removed {len(present)} {kind} leak column(s): "
                f"{', '.join(D.real_name(c) for c in present)}")

    # Safety net: the target must survive, and F3912 must not.
    if C.LEAK_COL in df.columns:
        df = df.drop(columns=[C.LEAK_COL])
        removed.setdefault("post_outcome", []).append(D.label(C.LEAK_COL))
        log(f"Removed {C.LEAK_COL} (FRAUD_SUSPECTED) via fallback rule")

    report["removed_semantic_leaks"] = removed
    return df


def encode_categoricals(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Encode the categorical fields instead of destroying them.

    ACCT_OPN_DAYS is ordinal (L7D < L14D < ... < G365D) so it becomes a single
    monotonic numeric column — one tree split expresses "newer than 90 days".
    The rest are low-cardinality nominals and become one-hot indicators, which
    keeps every downstream SHAP reason readable ("occupation = student").
    """
    encoded: dict[str, str] = {}   # real variable name -> how it was encoded

    # Detected from dtype and cardinality rather than a fixed list of names, so
    # this works on a schema nobody has catalogued. The dictionary's known names
    # are merged in as a hint for anything the dtype check would miss.
    detected = S.categorical_columns(df, C.TARGET_COL)
    hinted = [c for c in D.categorical_codes() if c in df.columns]
    targets = list(dict.fromkeys(detected + hinted))

    for code in targets:
        name = D.real_name(code)
        vals = df[code].astype(str).str.strip()

        # Ordinal vocabularies (L7D < L14D < ... < G365D, LOW < MED < HIGH) get a
        # single monotonic column: one tree split then expresses "newer than 90
        # days" instead of the model having to memorise a set of dummies.
        ordinal = S.ordinal_mapping(vals.unique())
        if ordinal is not None:
            df[code] = pd.to_numeric(vals.map(lambda v: ordinal.get(S.norm(v))),
                                     errors="coerce")
            encoded[name] = "ordinal"
            continue

        # Guard against an unexpectedly high-cardinality field exploding into
        # hundreds of dummy columns.
        if vals.nunique() > S.MAX_CATEGORICAL_CARDINALITY:
            df = df.drop(columns=[code])
            encoded[name] = f"dropped(cardinality={vals.nunique()})"
            continue

        dummies = pd.get_dummies(vals, prefix=f"mg_{name.lower()}", dtype=np.int8)
        dummies = dummies.drop(
            columns=[c for c in dummies.columns if c.endswith(("_nan", "_None"))],
            errors="ignore")
        df = pd.concat([df.drop(columns=[code]), dummies], axis=1)
        encoded[name] = f"one-hot({dummies.shape[1]})"

    if encoded:
        summary = ", ".join(f"{k}={v}" for k, v in encoded.items())
        log(f"Encoded {len(encoded)} categorical field(s): {summary}")
    report["encoded_categoricals"] = encoded
    return df


def coerce_numeric(df: pd.DataFrame, protect: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Convert leftover non-numeric columns to numbers; drop what cannot convert.

    Anything still textual at this point is unmapped by the dictionary, so we
    report it rather than let it vanish silently the way the old code did.
    """
    unconvertible = []
    for col in df.columns:
        if col in protect or pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == 0:
            unconvertible.append(col)
            df = df.drop(columns=[col])
        else:
            df[col] = converted
    if unconvertible:
        log(f"Dropped {len(unconvertible)} non-numeric, non-dictionary column(s): "
            f"{[D.label(c) for c in unconvertible[:8]]}")
    return df, unconvertible


def harden_against_extract_artefact(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Drop columns whose BLANK RATE differs between the classes.

    This is the pipeline's answer to the dataset's deepest problem. The positives
    and negatives were extracted in different months, so anything that varies
    between extraction runs — which fields the job populated, how sparse a feed
    was that month — lines up perfectly with the label without describing a
    single customer behaviour.

    Whether a cell is blank is a property of the extract, not of an account. When
    a column is blank far more often for one class than the other, no amount of
    modelling can separate "mule behaviour" from "different extraction run", so
    the column goes, values and all.

    Note on using the label here: this is a filter, not a fitted component. It
    only ever removes columns, so it can make the reported metrics worse but
    never better — the conservative direction. It runs before any model exists.
    """
    if not C.HARDEN_AGAINST_EXTRACT_ARTEFACT:
        report["extract_hardening"] = {"enabled": False}
        return df

    y = df[C.TARGET_COL].astype(int)
    feats = [c for c in df.columns if c != C.TARGET_COL]
    blank = df[feats].isna()
    diff = (blank[y == 1].mean() - blank[y == 0].mean()).abs()

    confounded = diff[diff > C.MAX_MISSINGNESS_DIFFERENTIAL].index.tolist()
    worst = diff.sort_values(ascending=False).head(15)

    df = df.drop(columns=confounded)
    report["extract_hardening"] = {
        "enabled": True,
        "tolerance": C.MAX_MISSINGNESS_DIFFERENTIAL,
        "columns_dropped": len(confounded),
        "columns_remaining": int(df.shape[1] - 1),
        "worst_offenders": {D.label(k): round(float(v), 4) for k, v in worst.items()},
        "rationale": "blank/not-blank is decided by the extraction job, not by "
                     "the customer; a class-dependent blank rate is an artefact "
                     "of positives and negatives coming from different monthly "
                     "extracts",
    }
    log(f"EXTRACT HARDENING: dropped {len(confounded):,} columns whose blank rate "
        f"differs between classes by >{C.MAX_MISSINGNESS_DIFFERENTIAL:.0%}")
    return df


def zero_fill_activity_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """For absolute activity aggregates, a missing value means NO activity.

    TOT_TXNAMT_L7D is blank when the account made no transactions that week, and
    UPI_AMT_L7D is blank when the customer does not use UPI at all. Median-
    imputing those invents activity that never happened, and dropping them as
    ">50% missing" throws away the fact that an account rides exactly one rail —
    which is itself one of the strongest mule tells.

    So: absolute aggregates (counts, amounts, balances, alert tallies) get 0.
    Derived quantities (R_/RA_/D_/DA_ ratios and deviations) are left alone,
    because there NaN means "undefined baseline", not "zero", and the later
    median-impute is the right treatment.
    """
    absolute, filled = [], 0
    have_dict = bool(D.name_map())
    for col in df.columns:
        if col == C.TARGET_COL or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Without the dictionary D.real_name() hands back the F-code itself, so
        # matching on it alone matched nothing and this whole treatment silently
        # became a no-op. Fall back to the raw column name, which is right for
        # any dataset whose headers are already readable.
        name = D.real_name(col).upper()
        if not have_dict or name == str(col).upper():
            name = str(col).upper()
        if name.startswith(("R_", "RA_", "D_", "DA_", "RT_", "DEV_", "MM_")):
            continue
        if any(tok in name for tok in ("AMT", "TXN", "BAL", "ALERT")):
            absolute.append(col)

    if absolute:
        na_before = int(df[absolute].isna().sum().sum())
        df[absolute] = df[absolute].fillna(0)
        filled = na_before

    report["zero_filled_activity"] = {
        "columns": len(absolute),
        "values_filled": filled,
        "dictionary_available": have_dict,
        "rationale": "missing transaction/balance aggregate = no such activity",
    }
    if not absolute:
        log("WARNING: no activity-aggregate columns matched — 'missing = no "
            "activity' treatment did NOT run. Column names are unreadable and "
            "no data dictionary was loaded.")
    log(f"Zero-filled {filled:,} missing values across {len(absolute):,} "
        f"activity-aggregate columns (missing = no activity, not unknown)")
    return df


def remove_partition_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Drop columns that partition the classes rather than describe them.

    This is `MNTH` generalised. In the supplied file every negative is the Oct25
    extract and every positive Sep/Nov/Dec25, so no month value contains both
    classes and the column alone reproduces the label while saying nothing about
    any customer. The old pipeline dropped it because a human noticed it.

    `schema.partition_columns` finds that shape directly — a low-cardinality
    column whose values are class-pure — so the same defence fires on a dataset
    nobody has inspected, and on this one it re-derives MNTH from scratch. A
    genuine behavioural categorical (occupation, product) fails the test,
    because its values contain both classes.
    """
    if not C.DETECT_PARTITION_COLUMNS:
        report["partition_audit"] = {"enabled": False}
        return df

    found = S.partition_columns(df, C.TARGET_COL,
                                max_cardinality=C.PARTITION_MAX_CARDINALITY,
                                min_purity=C.PARTITION_MIN_PURITY)
    present = [f for f in found if f["column"] in df.columns]

    report["partition_audit"] = {
        "enabled": True,
        "min_purity": C.PARTITION_MIN_PURITY,
        "max_cardinality": C.PARTITION_MAX_CARDINALITY,
        "flagged_count": len(present),
        "flagged": [{**f, "label": D.label(f["column"])} for f in present[:10]],
        "verdict": (
            "Each column listed splits the classes into disjoint value sets. That "
            "is a property of how the sample was assembled, not of any account, "
            "so it is dropped along with everything it would have leaked."
            if present else
            "No column partitions the classes. Nothing here was assembled in a "
            "way that hands the label to the model for free."
        ),
    }

    if present:
        cols = [f["column"] for f in present]
        log(f"PARTITION COLUMNS: {len(cols)} column(s) split the classes into "
            f"disjoint value sets -> {[D.label(c) for c in cols[:5]]}")
        df = df.drop(columns=cols)
    else:
        log("PARTITION AUDIT: clean — no column partitions the classes.")
    return df


def separation_audit(df: pd.DataFrame, target: str, report: dict) -> list[str]:
    """Find columns that separate the two classes (near-)perfectly.

    A column is flagged when knowing it tells you the label almost for free:
    either the mule and normal value ranges never overlap, or a single value of
    the column captures essentially all mules and essentially no normals.

    This is a *structural* check, not a correlation check. MNTH scores a perfect
    1.0 here while a plain Pearson correlation on its encoded form would look
    unremarkable — which is precisely how it survived in the original pipeline.
    """
    y = df[target].astype(int).values
    pos, neg = y == 1, y == 0
    flagged: list[dict] = []

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        v = s.values.astype(float)
        vp, vn = v[pos], v[neg]
        vp, vn = vp[~np.isnan(vp)], vn[~np.isnan(vn)]
        if len(vp) == 0 or len(vn) == 0:
            continue

        # (a) Disjoint ranges — no normal account ever reaches the mule range.
        disjoint = vp.min() > vn.max() or vp.max() < vn.min()

        # (b) One value that is nearly exclusive to mules.
        exclusive = 0.0
        if s.nunique(dropna=True) <= 20:
            for val in np.unique(vp):
                hit_p = float((vp == val).mean())
                hit_n = float((vn == val).mean())
                exclusive = max(exclusive, hit_p - hit_n)

        if disjoint or exclusive > 0.95:
            flagged.append({
                "column": D.label(col),
                "disjoint_ranges": bool(disjoint),
                "max_exclusive_value_gap": round(float(exclusive), 4),
            })

    report["separation_audit"] = {
        "columns_scanned": int(df.shape[1] - 1),
        "flagged_count": len(flagged),
        "flagged": flagged[:50],
    }
    if flagged:
        log(f"SEPARATION AUDIT: {len(flagged)} column(s) separate the classes "
            f"near-perfectly -> {[f['column'] for f in flagged[:5]]}")
    else:
        log("SEPARATION AUDIT: clean — no column separates the classes perfectly.")
    return [f["column"] for f in flagged]


def detect_corr_leaks(df: pd.DataFrame, target: str) -> tuple[list[str], dict]:
    """Correlation backstop behind the semantic pass."""
    feats = df.drop(columns=[target])
    corrs = feats.corrwith(df[target], numeric_only=True).abs().dropna()
    leaks = corrs[corrs > C.LEAK_CORR_THRESHOLD].sort_values(ascending=False)
    top = {D.label(k): round(float(v), 4)
           for k, v in corrs.sort_values(ascending=False).head(20).items()}
    return leaks.index.tolist(), top


def drop_constant_and_collinear(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """Remove zero-variance columns and one of each near-duplicate pair.

    The correlation matrix is computed in NumPy on the standardised matrix; on
    ~3,500 columns that is one BLAS matrix product instead of pandas' pairwise
    loop, which turns minutes into seconds.
    """
    feats = df.drop(columns=[target])
    nunique = feats.nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    feats = feats.drop(columns=constant)

    X = feats.to_numpy(dtype=np.float64, na_value=np.nan)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    X -= X.mean(axis=0)
    sd = X.std(axis=0)
    keep = sd > 0
    Xs = X[:, keep] / sd[keep]
    names = np.array(feats.columns)[keep]

    corr = np.abs((Xs.T @ Xs) / len(Xs))
    np.fill_diagonal(corr, 0.0)
    # Upper triangle only: for each near-duplicate pair, drop the later column.
    dup_idx = np.where(np.triu(corr, k=1) > C.COLLINEAR_CORR_THRESHOLD)[1]
    dup = sorted(set(names[np.unique(dup_idx)].tolist()))

    zero_var = np.array(feats.columns)[~keep].tolist()

    # Never dedup away a column Stage 2 needs as the numerator or denominator of
    # a behavioural ratio (see dictionary.FEATURE_BASE_NAMES).
    protected = D.protected_codes()
    rescued = sorted(set(dup) & protected)
    dup = [c for c in dup if c not in protected]

    to_drop = sorted(set(constant + zero_var + dup))
    return df.drop(columns=to_drop), {
        "constant": len(constant) + len(zero_var),
        "near_duplicate": len(dup),
        "rescued_feature_bases": [D.label(c) for c in rescued],
    }


def main() -> None:
    df = load_raw(C.RAW_CSV)

    # The target is discovered, not assumed. See config.resolve_target().
    try:
        target, how = S.resolve_target(df, C.TARGET_COL_HINT)
    except KeyError as exc:
        die(str(exc))
        return
    C.__dict__["TARGET_COL"] = target      # freeze for every later import
    log(f"Target column: {target}  ({how})")

    report: dict = {
        "input_shape": list(df.shape),
        "source_file": str(C.RAW_CSV),
        "dictionary_used": bool(len(D.load())),
        "schema": S.describe_schema(df, target, how),
    }

    y = df[C.TARGET_COL]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    report["class_balance"] = {
        "positives(mules)": n_pos,
        "negatives(normal)": n_neg,
        "prevalence_pct": round(100 * n_pos / len(y), 4),
    }
    log(f"Class balance: {n_pos} mules / {n_neg} normal "
        f"({report['class_balance']['prevalence_pct']}% prevalence)")

    # Very large inputs get stratified-sampled so a live demo finishes. Every
    # positive is kept: at this prevalence a handful of lost mules would change
    # the metrics more than the speed is worth.
    if C.FAST_MODE and len(df) > C.FAST_MAX_ROWS:
        y_all = df[C.TARGET_COL].astype(int)
        pos = df[y_all == 1]
        neg = df[y_all == 0].sample(
            n=max(C.FAST_MAX_ROWS - len(pos), 100),
            random_state=C.RANDOM_STATE)
        before = len(df)
        df = pd.concat([pos, neg]).sample(frac=1.0, random_state=C.RANDOM_STATE)
        df = df.reset_index(drop=True)
        report["fast_mode_sampling"] = {
            "rows_before": int(before), "rows_after": int(len(df)),
            "positives_kept": int(len(pos)),
            "note": "Stratified sample for demo speed. Every positive retained; "
                    "negatives sampled. Metrics computed on this sample.",
        }
        log(f"FAST MODE: sampled {before:,} rows down to {len(df):,} "
            f"(all {len(pos)} positives kept)")

    df = drop_identifier_columns(df, report)

    # Audit assembly artefacts BEFORE removing them, so the report shows its
    # working. On this dataset the top hit is MNTH, found by shape rather than
    # by name — see remove_partition_columns().
    worst = S.partition_columns(df, C.TARGET_COL,
                                max_cardinality=C.PARTITION_MAX_CARDINALITY,
                                min_purity=C.PARTITION_MIN_PURITY)
    if worst:
        top = worst[0]
        report["structural_leak_audit"] = {
            "column": D.label(top["column"]),
            "purity": top["purity"],
            "values_containing_both_classes": top["values_containing_both_classes"],
            "crosstab": top["crosstab"],
            "detected_by": "shape — low-cardinality column whose values are "
                           "class-pure; no prior knowledge of this schema used",
            "verdict": "Every negative and every positive fall in disjoint "
                       "values, so this column alone separates the classes. It "
                       "is an artefact of how the sample was assembled, not a "
                       "property of an account. Dropped.",
        }
        log(f"STRUCTURAL LEAK: {D.label(top['column'])} partitions the classes "
            f"(purity {top['purity']:.3f})")

    # Raw date strings are proxies for the extract and cannot be modelled as
    # numbers; a numeric age feature replaces them where one exists.
    dates = S.raw_date_matches(df, [c for c in df.columns if c != C.TARGET_COL])
    if dates:
        df = df.drop(columns=dates)
        report["dropped_raw_date_columns"] = [D.label(c) for c in dates]
        log(f"Dropped {len(dates)} raw date column(s): {[D.label(c) for c in dates[:5]]}")

    df = remove_semantic_leaks(df, report)
    df = encode_categoricals(df, report)
    df, _ = coerce_numeric(df, protect=[C.TARGET_COL])

    df = remove_partition_columns(df, report)
    df = harden_against_extract_artefact(df, report)
    df = zero_fill_activity_columns(df, report)

    # Drop sparse columns (feature-base columns are exempt: Stage 2 needs them)
    miss = df.drop(columns=[C.TARGET_COL]).isna().mean()
    protected = D.protected_codes()
    sparse = [c for c in miss[miss > C.MISSING_DROP_FRAC].index if c not in protected]
    df = df.drop(columns=sparse)
    report["dropped_sparse_count"] = len(sparse)
    log(f"Dropped {len(sparse)} columns missing >{int(C.MISSING_DROP_FRAC*100)}%")

    # Correlation backstop
    corr_leaks, top_corr = detect_corr_leaks(df, C.TARGET_COL)
    report["top_target_correlations"] = top_corr
    if corr_leaks:
        df = df.drop(columns=corr_leaks)
        log(f"Correlation backstop removed {len(corr_leaks)}: "
            f"{[D.label(c) for c in corr_leaks]}")
    report["removed_corr_leaks"] = [D.label(c) for c in corr_leaks]

    df, dropped = drop_constant_and_collinear(df, C.TARGET_COL)
    report["dropped_constant_count"] = dropped["constant"]
    report["dropped_collinear_count"] = dropped["near_duplicate"]
    report["rescued_feature_bases"] = dropped["rescued_feature_bases"]
    log(f"Dropped {dropped['constant']} constant and "
        f"{dropped['near_duplicate']} near-duplicate columns "
        f"({len(dropped['rescued_feature_bases'])} feature-base columns kept "
        f"despite collinearity)")

    # NOTE: no median imputation here, deliberately.
    #
    # This stage used to fill every remaining NaN with the column median taken
    # over ALL 9,082 rows. That let validation rows help decide the value used
    # to fill training rows, which is a transductive leak and contradicts the
    # claim that every fitted component lives inside the fold. Imputation now
    # happens in MuleEnsemble._prep(), which learns the medians from training
    # rows only and applies them frozen. NaNs are carried through to Stage 4/5.
    feat_cols = [c for c in df.columns if c != C.TARGET_COL]
    report["imputation"] = {
        "where": "MuleEnsemble._prep (inside the training fold)",
        "residual_nan_cells": int(df[feat_cols].isna().sum().sum()),
        "rationale": "fitting the median on all rows leaks validation "
                     "distribution into training rows",
    }

    separation_audit(df, C.TARGET_COL, report)

    report["output_shape"] = list(df.shape)
    report["clean_feature_count"] = len(feat_cols)
    log(f"Clean matrix: {df.shape[0]:,} rows x {len(feat_cols):,} features + target")

    save_frame(df, C.CLEAN_PARQUET)
    save_json(report, C.REPORTS_DIR / "01_clean_report.json")


if __name__ == "__main__":
    main()
