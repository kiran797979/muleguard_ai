"""
MuleGuard AI — the data dictionary as executable knowledge.

The hackathon ships `Description.xlsx` (sheet `Data_Dicitionary`), which maps every
anonymised column `F1..F3924` to a real banking variable name and description.
This module turns that spreadsheet into the semantic backbone of the pipeline:

  * `F` code  <->  real name (e.g. F3891 -> CUST_OCCP)  <->  human description
  * the *family* decomposition every name encodes:
        STAT (R / RA / D / DA / AVG / MAX / MIN / MM / TOT)
      x CHANNEL (CASH / CHQ / UPI / ATM / ELEC / POS / NET / BBPS / APB / GST ...)
      x DIRECTION (CR / DB / total)
      x WINDOW (7D / 14D / 31D / 7-14 / 7-31 / 14-31)
  * the bank's own 18 hand-picked variables (`Bank_Finalized_Variables` column)
  * a LEAKAGE CLASSIFICATION grounded in what each field *means*, not in how
    strongly it happens to correlate with the target.

Why this matters: without the dictionary, a model on this data is 3,924 opaque
numbers. With it, every feature, every SHAP reason, and every leak decision is
explainable to a regulator in plain English.

The dictionary is optional at runtime — if the workbook is absent the pipeline
degrades to code-only names and logs that reason lists will be less readable.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

import config as C
import roles as R
import schema as S
from utils import log

# The PSB workbook's own (misspelled) sheet name. Tried first, but the loader
# falls back to scanning every sheet, so a differently-named workbook works too.
PREFERRED_SHEET = "Data_Dicitionary"

# --------------------------------------------------------------------------
# Leakage classification, by MEANING
# --------------------------------------------------------------------------
# These are recorded only AFTER a human analyst finishes investigating an alert.
# At the moment we must score a live account they do not exist. Training on them
# is target leakage no matter how weak the correlation looks.
# These names are the PSB file's instances. The GENERAL rule lives in
# `schema.POST_OUTCOME_PATTERNS`, which matches by meaning on any schema; the
# explicit set below is kept so this file still documents what was found here.
POST_OUTCOME_NAMES = {
    "FRAUD_SUSPECTED",    # F3912 — the investigation's verdict (corr ~0.97)
    "OTHER_RESOLUTION",   # F3913 — resolution bucket
    "FALSE_POSITIVE",     # F3914 — resolution bucket
    "UNATTENDED",         # F3915 — resolution bucket
    "MIN_RESOLVE_DAYS",   # F3898 — how long the investigation took
    "MAX_RESOLVE_DAYS",   # F3899 — how long the investigation took
}

# Structural artefacts of how the dataset was ASSEMBLED, not properties of an
# account. MNTH is the severe one: in the supplied file every negative is Oct25
# and every positive is Sep/Nov/Dec25, so the month alone separates the classes
# perfectly. Keeping it would produce a meaningless 100% score.
# Again: the names found here. `schema.STRUCTURAL_PATTERNS` generalises by name
# and `schema.partition_columns()` generalises by SHAPE — the latter re-derives
# MNTH from the data alone, without anyone having to know it exists.
STRUCTURAL_LEAK_NAMES = {
    "MNTH",           # F2230 — month of the data snapshot
    "ACCT_OPN_DATE",  # F3888 — raw date string; replaced by a numeric age feature
}

# Categorical fields that must be ENCODED rather than coerced-to-NaN-and-dropped.
# The stock pipeline silently destroyed all of these.
CATEGORICAL_NAMES = {
    "PRODUCT_NAME",         # F3886
    "ACCT_OPN_DAYS",        # F3889 — ordinal buckets (L7D < L14D < ... < G365D)
    "AREA_CATEGORY",        # F3890 — R / SU / M / U
    "CUST_OCCP",            # F3891 — occupation code
    "GENDER",               # F3892
    "SEGMENTATION_CLASS",   # F3893 — RETAIL / CORPORATE
}

# ACCT_OPN_DAYS is ordinal, not nominal — encode it monotonically so trees can
# split on "newer than X" in one cut instead of memorising dummies.
ACCT_OPN_DAYS_ORDER = {
    "L7D": 7, "L14D": 14, "L31D": 31, "L90D": 90,
    "L180D": 180, "L365D": 365, "G365D": 730,
}

# Base columns that Stage 2 needs to build its behavioural ratios. They must
# survive the near-duplicate filter in Stage 1: TOT_TXNAMT_CR_L7D is ~collinear
# with TOT_TXNAMT_L7D whenever flow is mostly one-directional, so the dedup pass
# happily discards it — and with it the pass-through ratio, which is the single
# most diagnostic mule feature there is. Redundancy costs a tree model nothing;
# losing the numerator of a ratio costs it the signal.
FEATURE_BASE_NAMES = {
    # flow, by direction and window
    "TOT_TXNAMT_L7D", "TOT_TXNAMT_CR_L7D", "TOT_TXNAMT_DB_L7D",
    "TOT_TXNAMT_L14D", "TOT_TXNAMT_CR_L14D", "TOT_TXNAMT_DB_L14D",
    "TOT_TXNAMT_L31D", "TOT_TXNAMT_CR_L31D", "TOT_TXNAMT_DB_L31D",
    "TOT_TXNS_L7D", "TOT_TXNS_L31D",
    # balances
    "AVG_BAL_7DAYS", "AVG_BAL_14DAYS", "AVG_BAL_31DAYS",
    "MAX_BAL_7DAYS", "MIN_BAL_7DAYS", "MAX_BAL_31DAYS", "MIN_BAL_31DAYS",
    # per-rail volume for the channel-mix features
    "CASH_AMT_L7D", "CHQ_AMT_L7D", "UPI_AMT_L7D", "ATM_AMT_L7D",
    "ELEC_XFER_AMT_L7D", "POS_PYMT_AMT_L7D", "NET_BNKING_AMT_L7D",
    "APB_AMT_L7D", "BBPS_AMT_L7D", "GST_AMT_L7D",
    # cash-out / digital-in legs
    "CASH_AMT_DB_L7D", "ATM_AMT_DB_L7D",
    "UPI_AMT_CR_L7D", "ELEC_XFER_AMT_CR_L7D",
    "NET_BNKING_AMT_CR_L7D", "APB_AMT_CR_L7D",
    # alert timing
    "COUNT_ALERTS", "MORNING_ALERTS", "AFTERNOON_ALERTS",
    "EVENING_ALERTS", "NIGHT_ALERTS",
}


@lru_cache(maxsize=1)
def protected_codes() -> set[str]:
    """Columns Stage 1 must not discard as near-duplicates.

    Resolved fuzzily, so a dataset spelling these differently still protects the
    right columns.
    """
    found = {resolve(n) for n in FEATURE_BASE_NAMES}
    return {c for c in found if c}


STAT_PREFIXES = {
    "R": "ratio", "RA": "ratio_of_averages", "D": "deviation",
    "DA": "deviation_of_averages", "AVG": "average", "MAX": "maximum",
    "MIN": "minimum", "MM": "max_minus_min", "TOT": "total",
}

CHANNEL_TOKENS = {
    "CASH": "cash", "CHQ": "cheque", "UPI": "UPI", "ATM": "ATM",
    "ELEC_XFER": "online transfer (IMPS/NEFT/RTGS)", "POS_PYMT": "merchant payment",
    "NET_BNKING": "net banking", "MBNKING": "mobile banking", "BBPS": "bill payment",
    "APB": "Aadhaar Payment Bridge", "GST": "GST", "LOAN": "loan",
    "STDNG": "standing instruction", "NON_CASH_CHQ": "non-cash non-cheque",
    "CI": "customer-induced", "BI_FEES_CHRGS": "bank fees/charges",
}


EMPTY_DICT = ["feature", "name", "desc", "bank_final"]


@lru_cache(maxsize=1)
def dataset_columns() -> tuple[str, ...]:
    """The raw dataset's own column names, read from the header alone.

    Needed for the identity fallback below: when a dataset already has readable
    headers, the "dictionary" is the header row.
    """
    try:
        return tuple(pd.read_csv(C.RAW_CSV, nrows=0).columns)
    except Exception:  # noqa: BLE001 — missing file, unreadable, wrong format
        return ()


def _pick_dictionary_frame(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Find the (code, name, description) columns in an arbitrary sheet.

    The PSB workbook happens to put them in the first four columns, but nothing
    guarantees that for another file. We score each column by how many of its
    values are actual columns of the dataset, and take the best match as the
    code column; the next two textual columns become name and description.
    """
    cols = set(dataset_columns())
    if raw.empty:
        return None

    best, best_hits = None, 0
    for c in raw.columns:
        vals = raw[c].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        hits = int(vals.isin(cols).sum()) if cols else 0
        if hits > best_hits:
            best, best_hits = c, hits

    if best is None or best_hits < max(5, 0.2 * len(raw)):
        # No column looks like a key into the dataset. Fall back to positional
        # order, which is what the supplied workbook uses.
        if raw.shape[1] < 2:
            return None
        best = raw.columns[0]

    others = [c for c in raw.columns if c != best]
    name_col = others[0] if others else best
    desc_col = others[1] if len(others) > 1 else name_col
    flag_col = others[2] if len(others) > 2 else None

    out = pd.DataFrame({
        "feature": raw[best].astype(str).str.strip(),
        "name": raw[name_col].astype(str).str.strip(),
        "desc": raw[desc_col].astype(str).str.strip(),
        "bank_final": raw[flag_col] if flag_col is not None else pd.NA,
    })
    return out.dropna(subset=["feature"])


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    r"""Load a data dictionary from .xlsx or .csv; empty frame if there is none.

    Generalised from the original, which required an .xlsx with one specific
    sheet name and discarded any row whose code did not match an F-number.
    made the dictionary useless for any other dataset. Now: either file format,
    any sheet, and codes are kept when they name a real column of the dataset.
    """
    path = C.DICTIONARY_XLSX
    if not path.exists():
        log(f"No data dictionary at {path} — using the dataset's own column names.")
        return pd.DataFrame(columns=EMPTY_DICT)

    frames: list[pd.DataFrame] = []
    try:
        if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else None
            frames.append(pd.read_csv(path, sep=sep, engine="python"))
        else:
            book = pd.read_excel(path, sheet_name=None, header=0)
            ordered = ([book[PREFERRED_SHEET]] if PREFERRED_SHEET in book else [])
            ordered += [v for k, v in book.items() if k != PREFERRED_SHEET]
            frames.extend(ordered)
    except Exception as exc:  # noqa: BLE001 — openpyxl missing, corrupt file
        log(f"Could not read data dictionary ({exc}) — using column names instead.")
        return pd.DataFrame(columns=EMPTY_DICT)

    cols = set(dataset_columns())
    best, best_hits = None, -1
    for raw in frames:
        cand = _pick_dictionary_frame(raw)
        if cand is None or cand.empty:
            continue
        hits = int(cand["feature"].isin(cols).sum()) if cols else len(cand)
        if hits > best_hits:
            best, best_hits = cand, hits

    if best is None:
        log("Data dictionary contained no usable mapping — using column names.")
        return pd.DataFrame(columns=EMPTY_DICT)

    if cols:
        best = best[best["feature"].isin(cols)]
    best = best[best["name"].notna() & (best["name"].astype(str).str.len() > 0)]
    best = best[best["name"].astype(str).str.lower() != "nan"]

    if best.empty:
        log("Data dictionary rows did not match any dataset column — using names.")
        return pd.DataFrame(columns=EMPTY_DICT)

    log(f"Data dictionary loaded: {len(best):,} variable definitions "
        f"({100*len(best)/max(len(cols), 1):.0f}% of columns covered)")
    return best.reset_index(drop=True)


@lru_cache(maxsize=1)
def name_map() -> dict[str, str]:
    """column -> human-readable variable name.

    Falls back to the IDENTITY map over the dataset's own columns. That single
    change is what lets every downstream lookup — leak classification, feature
    resolution, SHAP labelling — work unchanged on a dataset whose headers are
    already readable and which therefore ships no dictionary at all.
    """
    d = load()
    if not d.empty:
        mapped = dict(zip(d["feature"], d["name"]))
        # Columns the dictionary does not cover still need an entry.
        for c in dataset_columns():
            mapped.setdefault(c, c)
        return mapped
    return {c: c for c in dataset_columns()}


@lru_cache(maxsize=1)
def desc_map() -> dict[str, str]:
    d = load()
    return dict(zip(d["feature"], d["desc"])) if not d.empty else {}


@lru_cache(maxsize=1)
def code_map() -> dict[str, str]:
    """Real name -> column (inverse of name_map). Exact keys only."""
    return {v: k for k, v in name_map().items()}


@lru_cache(maxsize=1)
def _norm_map() -> dict[str, str]:
    """Normalised real name -> column, plus normalised column -> column.

    Punctuation and case stop mattering, so `UPI_AMT_L7D` in this code resolves
    against `upi.amt.l7d` or `Upi Amt L7D` in somebody else's file.
    """
    out: dict[str, str] = {}
    for code, nm in name_map().items():
        out.setdefault(S.norm(nm), code)
        out.setdefault(S.norm(code), code)
    return out


@lru_cache(maxsize=1)
def _role_index() -> R.RoleIndex:
    """Every dataset column indexed by what it MEANS rather than what it is called."""
    return R.RoleIndex(dataset_columns(), label_of=real_name)


def resolve(name: str, by_role: bool = True) -> str | None:
    """Real variable name -> the dataset column holding it, or None.

    Three passes, cheapest and most certain first:

      1. exact name
      2. normalised name, so punctuation and case stop mattering
      3. ROLE, so `TOT_TXNAMT_CR_L7D` also finds `credit_value_week` or
         `InwardAmt7Day` on a schema that has never been seen before

    Pass 3 is what lets the 29 behavioural features build on an unfamiliar
    extract. It is last on purpose: role matching infers, and an inference should
    never override a name that actually matched.

    Pass `by_role=False` where a wrong-but-plausible match would be worse than no
    match at all.
    """
    cm = code_map()
    if name in cm:
        return cm[name]
    hit = _norm_map().get(S.norm(name))
    if hit is not None:
        return hit
    if not by_role:
        return None

    want = R.parse_role(name)
    # A role with almost nothing in it would match far too much.
    if want.specificity() < 2 or want.measure is None:
        return None
    return _role_index().find(want)


def real_name(code: str) -> str:
    """F3891 -> 'CUST_OCCP'. Returns the code itself when unknown."""
    return name_map().get(code, code)


def describe(code: str) -> str:
    """F3891 -> 'Occupation code of customer'. Empty string when unknown."""
    return desc_map().get(code, "")


def label(code: str) -> str:
    """Display label for reports: 'F3891 (CUST_OCCP)'."""
    nm = name_map().get(code)
    return f"{code} ({nm})" if nm and nm != code else code


def codes_for(*names: str) -> list[str]:
    """Map real variable names to the dataset columns holding them."""
    return [c for c in (resolve(n) for n in names) if c]


def _codes_matching(patterns: list[str]) -> list[str]:
    """Every dataset column whose real name matches one of these patterns.

    This is the generalisation of the hand-written name sets above: the sets say
    what was found in the PSB file, the patterns say what the same thing looks
    like anywhere.
    """
    rx = [re.compile(p) for p in patterns]
    return sorted(
        code for code, nm in name_map().items()
        if any(r.search(S.norm(nm)) or r.search(S.norm(code)) for r in rx)
    )


@lru_cache(maxsize=1)
def bank_finalized_codes() -> list[str]:
    """The 18 variables the bank's own analysts shortlisted (excluding the target).

    A domain-expert feature selection handed to us for free — we use it as a
    prior, not as a hard filter.
    """
    d = load()
    if d.empty:
        return []
    sel = d[d["bank_final"].notna() & (d["feature"] != C.TARGET_COL)]
    return sel["feature"].tolist()


@lru_cache(maxsize=1)
def leak_codes() -> dict[str, list[str]]:
    """Columns to remove before modelling, split by reason.

    Union of the explicit names known from this dataset and everything matching
    the general patterns in `schema`. The explicit list alone would find nothing
    in a file that spells its fields differently; the patterns alone would miss
    a locally-known oddity. Both, so neither gap matters.

    Returns {'post_outcome': [...], 'structural': [...]}.
    """
    post = set(codes_for(*POST_OUTCOME_NAMES)) | set(_codes_matching(S.POST_OUTCOME_PATTERNS))
    struct = set(codes_for(*STRUCTURAL_LEAK_NAMES)) | set(_codes_matching(S.STRUCTURAL_PATTERNS))
    struct -= post
    return {"post_outcome": sorted(post), "structural": sorted(struct)}


@lru_cache(maxsize=1)
def categorical_codes() -> list[str]:
    """Named categoricals known from this dataset, in a STABLE order.

    Only a hint now: Stage 1 detects categoricals from dtype and cardinality via
    `schema.categorical_columns`, which needs no prior knowledge of the schema.

    The `sorted()` is not cosmetic. `CATEGORICAL_NAMES` is a set, and iterating a
    set of strings follows the per-process hash seed, so this returned a
    different order on every run. Stage 1 encodes in that order, which fixes the
    one-hot column order, and XGBoost's `colsample_bytree` samples columns BY
    INDEX — so the same data with the same seed produced different metrics from
    one process to the next. Sorting removes the last source of run-to-run drift.
    """
    return sorted(codes_for(*sorted(CATEGORICAL_NAMES)))


def parse_family(name: str) -> dict[str, str | None]:
    """Decompose a variable name into (stat, channel, direction, window).

    'RA_CI_NON_CASH_CHQ_TXN_CR_L7_31D' ->
        stat='RA', channel='CI/NON_CASH_CHQ', direction='CR', window='7_31D'
    Used to group thousands of columns into a handful of interpretable families.
    """
    n = str(name).upper()

    win = None
    m = re.search(r"_L(\d+)_(\d+)D$", n)
    if m:
        win = f"{m.group(1)}_{m.group(2)}D"
    else:
        m = re.search(r"_L(\d+)D$", n)
        if m:
            win = f"{m.group(1)}D"

    stat = None
    for p in ("RA", "DA", "D_TA", "AVG", "MAX", "MIN", "MM", "TOT", "R", "D"):
        if n.startswith(p + "_"):
            stat = p
            break

    direction = "CR" if "_CR_" in n or n.endswith("_CR") else (
        "DB" if "_DB_" in n or n.endswith("_DB") else None
    )

    channel = next((v for k, v in CHANNEL_TOKENS.items() if k in n), None)

    return {"stat": stat, "channel": channel, "direction": direction, "window": win}


# Plain-English meanings for the features Stage 2 engineers. The dictionary only
# covers the bank's own F-codes, so without this an engineered feature would
# explain itself with its own variable name in SHAP reason lists and reports.
MG_DESCRIPTIONS = {
    "mg_passthrough_7d": "Credit/debit symmetry over last 7 days (1.0 = every rupee received left again)",
    "mg_passthrough_14d": "Credit/debit symmetry over last 14 days",
    "mg_passthrough_31d": "Credit/debit symmetry over last 31 days",
    "mg_net_flow_7d": "Normalised net flow over last 7 days (near 0 = pure pass-through)",
    "mg_net_flow_14d": "Normalised net flow over last 14 days",
    "mg_net_flow_31d": "Normalised net flow over last 31 days",
    "mg_turnover_over_balance_7d": "Weekly throughput as a multiple of average balance held",
    "mg_turnover_over_balance_14d": "Fortnightly throughput as a multiple of average balance held",
    "mg_turnover_over_balance_31d": "Monthly throughput as a multiple of average balance held",
    "mg_amount_burst_7v31": "Daily rupee velocity last week vs last month (>1 = sudden activation)",
    "mg_count_burst_7v31": "Daily transaction-count velocity last week vs last month",
    "mg_cash_out_share_7d": "Share of last week's debits taken out as cash",
    "mg_digital_in_cash_out_7d": "Digital money in against cash money out (layering handoff)",
    "mg_atm_out_share_7d": "Share of last week's debits withdrawn at ATMs",
    "mg_channel_hhi_7d": "Concentration of value across payment rails (single-purpose account)",
    "mg_channel_active_7d": "Number of distinct payment rails used in the last week",
    "mg_channel_top_share_7d": "Share of last week's value on the single busiest rail",
    "mg_avg_ticket_7d": "Average rupees per transaction over last 7 days (structuring signal)",
    "mg_avg_ticket_31d": "Average rupees per transaction over last 31 days",
    "mg_alert_share_morning": "Fraction of this account's alerts raised in the morning",
    "mg_alert_share_afternoon": "Fraction of this account's alerts raised in the afternoon",
    "mg_alert_share_evening": "Fraction of this account's alerts raised in the evening",
    "mg_alert_share_night": "Fraction of this account's alerts raised at night",
    "mg_alert_time_entropy": "Spread of alerts across the day (low = fixed operating window)",
    "mg_balance_volatility_7d": "Peak-to-trough balance swing over 7 days, relative to average",
    "mg_balance_volatility_31d": "Peak-to-trough balance swing over 31 days, relative to average",
    "mg_occ_deviation_mean": "Average divergence from the customer's occupation cohort",
    "mg_occ_deviation_max": "Largest single divergence from the occupation cohort norm",
    "mg_occ_deviation_extreme_count": "Count of behaviours more than 3 units from the occupation norm",
    "mg_row_mean": "Mean of all behavioural features (overall activity level)",
    "mg_row_std": "Spread across all behavioural features (profile irregularity)",
    "mg_row_max": "Largest value across all behavioural features",
    "mg_row_min": "Smallest value across all behavioural features",
    "mg_row_range": "Range across all behavioural features",
    "mg_row_nonzero_frac": "Fraction of behavioural features that are non-zero",
}


def explain(code: str) -> str:
    """One plain-English line for a feature — used in SHAP reason lists.

    Falls back gracefully: description -> name -> code.
    """
    if code in MG_DESCRIPTIONS:
        return MG_DESCRIPTIONS[code]
    desc = describe(code)
    if desc:
        return desc
    nm = name_map().get(code)
    if not nm:
        return code
    fam = parse_family(nm)
    bits = [STAT_PREFIXES.get(fam["stat"] or "", "")]
    if fam["channel"]:
        bits.append(f"of {fam['channel']}")
    if fam["direction"]:
        bits.append("credits" if fam["direction"] == "CR" else "debits")
    if fam["window"]:
        bits.append(f"over last {fam['window'].replace('_', ' to ').replace('D', ' days')}")
    return " ".join(b for b in bits if b) or nm
