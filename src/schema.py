"""
MuleGuard AI — dataset adaptation layer.

Everything in this project that used to assume "the PSB hackathon file" lives
here instead. Give the pipeline a different CSV — different target name,
different column naming, no data dictionary at all — and this module works out
what it is looking at.

Four questions, answered without hardcoding:

  1. WHICH COLUMN IS THE TARGET?      `resolve_target`
  2. WHICH COLUMNS ARE ROW IDs?       `identifier_columns`
  3. WHICH COLUMNS ARE CATEGORICAL?   `categorical_columns`
  4. WHICH COLUMNS LEAK?              `post_outcome_matches`, `partition_columns`

The important one is (4). The original pipeline removed `MNTH` because someone
looked at the file and noticed it. `partition_columns()` finds that class of
column by its SHAPE — a low-cardinality column whose values are split between
the classes rather than shared by them — so the same defence works on a dataset
nobody has inspected by hand. On the supplied file it re-derives MNTH from first
principles.

Name matching is normalisation-based, not exact: `UPI_AMT_L7D`, `upi amount
l7d` and `Upi.Amt.L7d` all reduce to the same key, so a dataset that uses the
same vocabulary with different punctuation still resolves.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Name normalisation
# --------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def norm(name: object) -> str:
    """Reduce a column name to a comparable key.

    'RA_CI_Non-Cash Chq.Txn' -> 'RACINONCASHCHQTXN'. Punctuation, case and
    separator style stop mattering, so two datasets using the same vocabulary
    with different house style still line up.
    """
    return _NON_ALNUM.sub("", str(name).upper())


def norm_index(columns) -> dict[str, str]:
    """{normalised name -> original column}. First occurrence wins."""
    out: dict[str, str] = {}
    for c in columns:
        out.setdefault(norm(c), c)
    return out


# --------------------------------------------------------------------------
# 1. The target column
# --------------------------------------------------------------------------
# Ordered strongest-first. A name matching an earlier pattern beats a later one,
# so FRAUD_TGT is preferred over a column merely called "class".
TARGET_PATTERNS = [
    r"^FRAUDTGT$", r"^ISMULE$", r"^MULE$", r"^ISFRAUD$",
    r"FRAUD.*(TGT|TARGET|LABEL|FLAG)", r"MULE.*(TGT|TARGET|LABEL|FLAG)",
    r"^(TARGET|TGT|LABEL|CLASS|Y)$", r"(TARGET|LABEL)$",
]


def _binary_columns(df: pd.DataFrame) -> list[str]:
    """Columns holding exactly two distinct non-null values, one of them 0/false."""
    out = []
    for c in df.columns:
        vals = pd.unique(df[c].dropna())
        if len(vals) != 2:
            continue
        try:
            nums = sorted(float(v) for v in vals)
        except (TypeError, ValueError):
            continue
        if nums == [0.0, 1.0]:
            out.append(c)
    return out


def resolve_target(df: pd.DataFrame, configured: str | None = None) -> tuple[str, str]:
    """Return (column, how_it_was_decided).

    Priority: MULEGUARD_TARGET env var > a configured name that actually exists >
    a binary column whose name matches a target pattern > the only binary column
    > the last column if it is binary. Raises when none of those hold, because
    guessing a target is worse than stopping.
    """
    env = os.environ.get("MULEGUARD_TARGET")
    if env:
        if env in df.columns:
            return env, "MULEGUARD_TARGET environment variable"
        hit = norm_index(df.columns).get(norm(env))
        if hit:
            return hit, f"MULEGUARD_TARGET environment variable (matched {hit!r})"
        raise KeyError(f"MULEGUARD_TARGET={env!r} is not a column in this dataset.")

    if configured and configured in df.columns:
        return configured, f"config.TARGET_COL_HINT ({configured}) exists in this dataset"

    binaries = _binary_columns(df)
    if not binaries:
        raise KeyError(
            "No binary 0/1 column found, so the target cannot be identified. "
            "Set MULEGUARD_TARGET=<column> to name it explicitly."
        )

    bin_norm = {norm(c): c for c in binaries}
    for pat in TARGET_PATTERNS:
        rx = re.compile(pat)
        for n, original in bin_norm.items():
            if rx.search(n):
                return original, f"name matches /{pat}/ and the column is binary"

    if len(binaries) == 1:
        return binaries[0], "the only binary column in the dataset"

    if df.columns[-1] in binaries:
        # "The target is the last column" is a real convention and worth using,
        # but with several binary columns in play it is a guess, and a silently
        # wrong target is the worst failure this pipeline has: everything
        # downstream still runs and still looks correct. So the guess is taken
        # and then declared, naming the columns it passed over, rather than
        # being buried in a one-line provenance string nobody reads.
        rivals = [c for c in binaries if c != df.columns[-1]]
        if rivals:
            shown = ", ".join(map(str, rivals[:6]))
            more = f" and {len(rivals) - 6} more" if len(rivals) > 6 else ""
            return df.columns[-1], (
                f"LOW CONFIDENCE: taken as the last column and binary, but "
                f"{len(rivals)} other binary column(s) could equally be the "
                f"target ({shown}{more}). Set MULEGUARD_TARGET to remove the "
                f"ambiguity.")
        return df.columns[-1], "last column, and it is binary"

    raise KeyError(
        f"{len(binaries)} binary columns and none is recognisably a target "
        f"({binaries[:8]}...). Set MULEGUARD_TARGET=<column>."
    )


# --------------------------------------------------------------------------
# 2. Identifier columns
# --------------------------------------------------------------------------
ID_NAME_PATTERNS = [r"^UNNAMED", r"^INDEX$", r"^ROWID$", r"^ROWNUM(BER)?$",
                    r"^ID$", r"ACC(OUN)?T(NO|NUM|ID)", r"CUST(OMER)?ID",
                    r"^SRNO$", r"^SLNO$"]


def identifier_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Row identifiers — never features, however predictive they look.

    Caught two ways: by name, and by being near-unique per row. An account
    number correlates with nothing real, but a tree will happily memorise it.

    The uniqueness test deliberately EXCLUDES floating-point columns. Any
    continuous measurement is ~100% unique across 3,000 rows — a transaction
    amount, a balance, a ratio — so applying the rule to floats classifies the
    entire feature matrix as identifiers and deletes the dataset. Only strings
    and integers can be ids in the sense that matters here.
    """
    out, by_uniqueness = [], []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        if any(re.search(p, norm(c)) for p in ID_NAME_PATTERNS):
            out.append(c)
            continue
        if n <= 50 or pd.api.types.is_float_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) >= 0.99 * n:
            by_uniqueness.append(c)

    # Safety valve: if "near-unique" would take out a large share of the
    # columns, the heuristic is wrong about this dataset rather than the dataset
    # being all identifiers. Trust only the name matches in that case.
    if len(by_uniqueness) <= max(5, 0.10 * df.shape[1]):
        out.extend(by_uniqueness)
    return out


# --------------------------------------------------------------------------
# 3. Categorical columns
# --------------------------------------------------------------------------
MAX_CATEGORICAL_CARDINALITY = 30


def categorical_columns(df: pd.DataFrame, target: str,
                        exclude: set[str] | None = None) -> list[str]:
    """Non-numeric or low-cardinality-integer columns worth encoding.

    The stock pipeline coerced everything to numeric, which turned every
    categorical field into all-NaN and then dropped it as "sparse". Occupation
    alone carried a 3x spread in mule rate, so that was throwing away real
    signal. Detection is by dtype and cardinality, so it works on any schema.
    """
    exclude = exclude or set()
    out = []
    for c in df.columns:
        if c == target or c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        nun = s.nunique(dropna=True)
        if 1 < nun <= MAX_CATEGORICAL_CARDINALITY:
            out.append(c)
    return out


# Ordinal vocabularies: when a column's values are drawn from one of these, an
# ordinal encoding beats one-hot because a single split expresses "newer than X".
ORDINAL_VOCABULARIES: list[dict[str, float]] = [
    {"L7D": 7, "L14D": 14, "L31D": 31, "L90D": 90,
     "L180D": 180, "L365D": 365, "G365D": 730},
    {"NONE": 0, "LOW": 1, "MEDIUM": 2, "MED": 2, "HIGH": 3, "VERYHIGH": 4},
    {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4},
]


def ordinal_mapping(values) -> dict[str, float] | None:
    """Return a value->rank map if the column's vocabulary is a known ordinal."""
    seen = {norm(v) for v in values if pd.notna(v)}
    if not seen:
        return None
    for vocab in ORDINAL_VOCABULARIES:
        keys = {norm(k) for k in vocab}
        if seen <= keys and len(seen) > 2:
            return {norm(k): v for k, v in vocab.items()}
    return None


# --------------------------------------------------------------------------
# 4a. Post-outcome leakage, by meaning
# --------------------------------------------------------------------------
# Fields written only AFTER a human closes an investigation. At the moment a live
# account must be scored they do not exist, so training on them is leakage no
# matter how weak the correlation looks — FALSE_POSITIVE correlates 0.05 on the
# supplied file and no threshold would ever catch it.
POST_OUTCOME_PATTERNS = [
    r"FRAUDSUSPECT", r"SUSPECTEDFRAUD", r"FALSEPOSITIVE", r"TRUEPOSITIVE",
    r"RESOLUTION", r"RESOLVE(D)?DAYS", r"RESOLVEDBY", r"UNATTENDED",
    r"DISPOSITION", r"INVESTIGAT", r"CASESTATUS", r"CASEOUTCOME",
    r"ALERTOUTCOME", r"CONFIRMEDFRAUD", r"^SAR", r"^STR(FILED|RAISED)",
    r"CHARGEBACK", r"WRITEOFF", r"CLOSUREREASON", r"ANALYSTVERDICT",
]

# Artefacts of how the dataset was ASSEMBLED rather than properties of a
# customer: which extract a row came from, when the file was cut.
STRUCTURAL_PATTERNS = [
    r"^MNTH$", r"^MONTH$", r"SNAPSHOT", r"EXTRACT(DATE|MONTH|ID)?$",
    r"^BATCH", r"LOADDATE", r"ASOFDATE", r"^PARTITION", r"^DATAMONTH",
    r"^RUNDATE", r"^FILEDATE", r"^VINTAGE$",
]

# Raw date strings: unusable as numbers, and usually a proxy for the extract.
RAW_DATE_PATTERNS = [r"DATE$", r"^DT", r"DTTM", r"TIMESTAMP$"]


def _match(names, patterns) -> list[str]:
    rx = [re.compile(p) for p in patterns]
    return [c for c in names if any(r.search(norm(c)) for r in rx)]


def post_outcome_matches(columns) -> list[str]:
    return _match(columns, POST_OUTCOME_PATTERNS)


def structural_matches(columns) -> list[str]:
    return _match(columns, STRUCTURAL_PATTERNS)


def raw_date_matches(df: pd.DataFrame, columns) -> list[str]:
    """Date-looking columns that are not already numeric."""
    return [c for c in _match(columns, RAW_DATE_PATTERNS)
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------
# 4b. Partition columns — the generalisation of MNTH
# --------------------------------------------------------------------------
def partition_columns(df: pd.DataFrame, target: str,
                      max_cardinality: int = 60,
                      min_purity: float = 0.98) -> list[dict]:
    """Find low-cardinality columns whose VALUES are split between the classes.

    This is the shape of an assembly artefact. If every positive came from one
    monthly extract and every negative from another, then no value of `MNTH` is
    shared by both classes: each value is "pure". A genuine behavioural
    categorical does not behave that way — occupation codes contain both mules
    and normal customers.

    `purity` is the share of rows sitting in class-pure values. At 1.0 the
    column reproduces the label exactly while describing nothing about a
    customer. The original pipeline removed MNTH because a human spotted it;
    this finds the same column, and its equivalent in a file nobody has read,
    from the structure alone.

    Returns one record per flagged column, worst first, with the crosstab as
    evidence so a report can show its working rather than assert a verdict.
    """
    y = df[target].astype(int)
    n = len(df)
    flagged: list[dict] = []

    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        try:
            nun = s.nunique(dropna=True)
        except TypeError:          # unhashable contents
            continue
        if not 1 < nun <= max_cardinality:
            continue

        tab = pd.crosstab(s, y)
        if tab.shape[1] < 2:       # only one class present at all
            continue
        neg = tab.get(0, pd.Series(0, index=tab.index))
        pos = tab.get(1, pd.Series(0, index=tab.index))

        pure = ((neg == 0) | (pos == 0))
        rows_in_pure = int((neg + pos)[pure].sum())
        purity = rows_in_pure / max(n, 1)
        if purity < min_purity:
            continue

        # A column is only interesting if BOTH classes actually appear in it;
        # otherwise it is merely sparse, not a partition.
        if int(pos.sum()) == 0 or int(neg.sum()) == 0:
            continue

        flagged.append({
            "column": str(c),
            "n_values": int(nun),
            "purity": round(float(purity), 4),
            "values_containing_both_classes": int((~pure).sum()),
            "crosstab": {str(k): {"0": int(neg.get(k, 0)), "1": int(pos.get(k, 0))}
                         for k in tab.index},
        })

    flagged.sort(key=lambda r: (-r["purity"], r["n_values"]))
    return flagged


# --------------------------------------------------------------------------
# 4c. Row order — the artefact that leaves no column behind
# --------------------------------------------------------------------------
def row_order_leak(y) -> dict:
    """Is the label predictable from a row's position in the file?

    A dataset assembled by stacking a block of negatives on a block of positives
    carries a perfect predictor that appears in no column at all. It cannot leak
    into the model directly, because position is not a feature. It does
    something worse: it silently breaks any split that is not shuffled and
    stratified, so a naive holdout hands one class entirely to train and the
    other entirely to test.

    Measured as the AUROC of row index as a classifier. 0.5 is a shuffled file;
    1.0 means every positive sits after every negative; 0.0 means the reverse.
    Both extremes are the same finding.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n:
        return {"applicable": False, "reason": "only one class present"}

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    # AUROC of position, via the rank-sum identity: no sorting of scores needed
    # because the "score" IS the row index and is already in order.
    rank_sum = float(pos_idx.sum()) + n_pos          # ranks are 1-based
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))

    separation = abs(auroc - 0.5) * 2                # 0 shuffled, 1 fully sorted
    contiguous = bool(pos_idx.max() - pos_idx.min() + 1 == n_pos)
    return {
        "applicable": True,
        "position_auroc": round(float(auroc), 6),
        "separation": round(float(separation), 6),
        "positives_are_contiguous": contiguous,
        "first_positive_row": int(pos_idx.min()),
        "last_positive_row": int(pos_idx.max()),
        "n_rows": n,
        "sorted_by_label": bool(separation > 0.98),
        "verdict": ("The file is ordered by label: every positive sits in one "
                    "contiguous block. Row position is a perfect predictor that "
                    "belongs to no column, so it cannot be dropped — it has to be "
                    "shuffled away. Any split that is not shuffled and stratified "
                    "will be catastrophically wrong on this file."
                    if separation > 0.98 else
                    "Row position carries little information about the label."),
    }


# --------------------------------------------------------------------------
# Summary, for the reports
# --------------------------------------------------------------------------
def describe_schema(df: pd.DataFrame, target: str, how: str) -> dict:
    ids = identifier_columns(df, target)
    y = df[target].astype(int)
    coded = sum(1 for c in df.columns if re.fullmatch(r"[A-Z]\d+", str(c).upper()))
    return {
        "target_low_confidence": how.startswith("LOW CONFIDENCE"),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_column": target,
        "target_resolved_by": how,
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "prevalence_pct": round(100 * float(y.mean()), 4),
        "identifier_columns": ids[:20],
        "n_identifier_columns": len(ids),
        "column_naming": ("coded (e.g. F123) — a data dictionary adds a great deal"
                          if coded > 0.5 * df.shape[1] else
                          "readable — names are used directly"),
        "n_numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
        "n_non_numeric": int(df.shape[1] - df.select_dtypes(include=[np.number]).shape[1]),
    }


def bind_target(df: pd.DataFrame, config_module, hint: str | None = None) -> str:
    """Pin the target column for a stage that starts from an intermediate file.

    Stages after cleaning read `data/*.parquet`, not the raw CSV, so they cannot
    re-sniff the original. This trusts an already-resolved `TARGET_COL` when the
    frame actually contains it, and re-resolves otherwise — which is what makes
    a stage runnable on its own against a dataset it did not clean.
    """
    current = config_module.__dict__.get("TARGET_COL")
    if current and current in df.columns:
        return current
    col, _how = resolve_target(df, hint or getattr(config_module, "TARGET_COL_HINT", None))
    config_module.__dict__["TARGET_COL"] = col
    return col


@lru_cache(maxsize=8)
def _cached_header(path: str) -> tuple[str, ...]:
    return tuple(pd.read_csv(path, nrows=0).columns)
