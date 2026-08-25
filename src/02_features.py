"""
Stage 2/3 — Mule-typology feature engineering.

With the data dictionary in hand we no longer have to guess at column meanings.
Every feature below encodes a documented money-mule behaviour and is computed
from named columns, so each one can be defended in plain English to an auditor.

The mule signature, in one sentence: a mule account *receives* money and pushes
it straight back out, holding almost nothing, in bursts, through digital rails,
often at odd hours, on an account whose owner profile does not match the volume.
Each feature family below measures one clause of that sentence:

  1. PASS-THROUGH      credits ~= debits          -> the account is a conduit
  2. TURNOVER/BALANCE  volume >> balance held     -> money never rests
  3. BURST             7-day rate >> 31-day rate  -> sudden activation
  4. CASH-OUT          digital in, cash out       -> layering / placement
  5. CHANNEL MIX       concentration across rails -> single-purpose account
  6. TICKET SIZE       amount per transaction     -> structuring below limits
  7. ALERT TIMING      night-hour alert share     -> mules skew nocturnal (3x)
  8. BALANCE SHAPE     volatility vs average      -> spike-and-drain pattern
  9. PROFILE MISMATCH  volume vs occupation norm  -> the dataset ships 444 of
                       these already (D_*_OCC); we surface them as a family
 10. ROW PROFILE       semantics-free aggregates  -> catches what 1-9 miss

Graph feasibility is still checked honestly: the data is an account-level
feature matrix with no counterparty identifiers, so Stage 6 self-skips.

Run:  python src/02_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import dictionary as D
import roles as R
import schema as S
from utils import load_frame, log, save_frame, save_json

EPS = 1.0  # rupee-scale floor: keeps ratios finite without distorting real values

# Channels whose 7-day amount columns define the "rail mix" of an account.
CHANNELS = ["CASH", "CHQ", "UPI", "ATM", "ELEC_XFER", "POS_PYMT",
            "NET_BNKING", "APB", "BBPS", "GST"]


class Resolver:
    """Look up a documented variable by NAME and return the column if it survived.

    Cleaning drops sparse, constant, and near-duplicate columns, so a name in the
    dictionary is not guaranteed to be in the matrix. Every miss is recorded so
    the report can state exactly which features could not be built and why,
    rather than silently producing a column of zeros.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.missing: list[str] = []
        self.by_role: dict[str, str] = {}
        self.resolved_exactly: list[str] = []

    def get(self, name: str) -> pd.Series | None:
        """Find the column holding this quantity, however the dataset spells it.

        D.resolve() tries the exact name, then a normalised form, then the
        column's ROLE. The last one is what makes these features survive a
        schema nobody has seen: `TOT_TXNAMT_CR_L7D` will find a column that
        means "total credit amount over 7 days" whatever its name is.

        Every resolution is recorded, and anything found by role is recorded
        separately, so the report can show exactly which features were built on
        an inference rather than on a name that genuinely matched.
        """
        code = D.resolve(name)
        if code is None or code not in self.df.columns:
            self.missing.append(name)
            return None
        exact = D.resolve(name, by_role=False)
        if exact is None:
            self.by_role[name] = f"{code} ({R.describe(D.real_name(code))})"
        else:
            self.resolved_exactly.append(name)
        return pd.to_numeric(self.df[code], errors="coerce")

    def sum_of(self, names: list[str]) -> pd.Series | None:
        """Sum the available members of a group; None if none survived."""
        parts = [s for s in (self.get(n) for n in names) if s is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).sum(axis=1)


def _safe_ratio(num: pd.Series, den: pd.Series, eps: float = EPS) -> pd.Series:
    return (num / (den.abs() + eps)).replace([np.inf, -np.inf], 0).fillna(0)


def add_mule_typology_features(df: pd.DataFrame, report: dict) -> list[str]:
    """Build the nine behavioural families. Returns the list of columns added."""
    r = Resolver(df)
    added: dict[str, str] = {}  # column -> plain-English justification

    def put(col: str, series: pd.Series | None, why: str) -> None:
        if series is None:
            return
        df[col] = series.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
        added[col] = why

    # --- 1. Pass-through: does everything that comes in go straight out? -----
    for win in ("7D", "14D", "31D"):
        cr = r.get(f"TOT_TXNAMT_CR_L{win}")
        db = r.get(f"TOT_TXNAMT_DB_L{win}")
        if cr is None or db is None:
            continue
        lo, hi = np.minimum(cr, db), np.maximum(cr, db)
        put(f"mg_passthrough_{win.lower()}", pd.Series(lo / (hi + EPS), index=df.index),
            f"credit/debit symmetry over last {win} — 1.0 means every rupee in "
            f"left again — the money is passing straight through")
        put(f"mg_net_flow_{win.lower()}", _safe_ratio(cr - db, cr + db),
            f"normalised net flow over last {win}; near 0 = pure pass-through")

    # --- 2. Turnover vs balance held ---------------------------------------
    for win, bal in (("7D", "AVG_BAL_7DAYS"), ("14D", "AVG_BAL_14DAYS"),
                     ("31D", "AVG_BAL_31DAYS")):
        amt, b = r.get(f"TOT_TXNAMT_L{win}"), r.get(bal)
        if amt is None or b is None:
            continue
        put(f"mg_turnover_over_balance_{win.lower()}", _safe_ratio(amt, b),
            f"total {win} throughput divided by average balance — a mule moves "
            f"many multiples of what it ever holds")

    # --- 3. Burst: recent rate against the monthly baseline ------------------
    a7, a31 = r.get("TOT_TXNAMT_L7D"), r.get("TOT_TXNAMT_L31D")
    if a7 is not None and a31 is not None:
        put("mg_amount_burst_7v31", _safe_ratio(a7 / 7.0, a31 / 31.0),
            "daily rupee velocity in the last week against the last month — "
            ">1 means the account just woke up")
    n7, n31 = r.get("TOT_TXNS_L7D"), r.get("TOT_TXNS_L31D")
    if n7 is not None and n31 is not None:
        put("mg_count_burst_7v31", _safe_ratio(n7 / 7.0, n31 / 31.0),
            "daily transaction-count velocity, last week vs last month")

    # --- 4. Cash-out and digital-in: the layering pattern --------------------
    digital_in = r.sum_of(["UPI_AMT_CR_L7D", "ELEC_XFER_AMT_CR_L7D",
                           "NET_BNKING_AMT_CR_L7D", "APB_AMT_CR_L7D"])
    cash_out = r.get("CASH_AMT_DB_L7D")
    tot_db = r.get("TOT_TXNAMT_DB_L7D")
    if cash_out is not None and tot_db is not None:
        put("mg_cash_out_share_7d", _safe_ratio(cash_out, tot_db),
            "share of last-week debits taken out as cash — the classic exit leg")
    if digital_in is not None and cash_out is not None:
        put("mg_digital_in_cash_out_7d", _safe_ratio(digital_in, cash_out),
            "digital money in against cash money out — high values are the "
            "placement-to-layering handoff")
    atm_db = r.get("ATM_AMT_DB_L7D")
    if atm_db is not None and tot_db is not None:
        put("mg_atm_out_share_7d", _safe_ratio(atm_db, tot_db),
            "share of last-week debits withdrawn at ATMs")

    # --- 5. Channel mix: concentration and breadth --------------------------
    chan_cols, chan_names = [], []
    for ch in CHANNELS:
        s = r.get(f"{ch}_AMT_L7D")
        if s is not None:
            chan_cols.append(s.abs())
            chan_names.append(ch)
    if len(chan_cols) >= 3:
        M = pd.concat(chan_cols, axis=1).fillna(0)
        total = M.sum(axis=1)
        share = M.div(total + EPS, axis=0)
        put("mg_channel_hhi_7d", (share ** 2).sum(axis=1),
            "Herfindahl concentration across payment rails — a mule is usually "
            "a single-purpose account riding one rail")
        put("mg_channel_active_7d", (M > 0).sum(axis=1).astype(float),
            "how many distinct payment rails were used in the last week")
        put("mg_channel_top_share_7d", share.max(axis=1),
            "share of last-week value on the single busiest rail")
        report["channels_used_for_mix"] = chan_names

    # --- 6. Ticket size: structuring below reporting limits -----------------
    for win in ("7D", "31D"):
        amt, cnt = r.get(f"TOT_TXNAMT_L{win}"), r.get(f"TOT_TXNS_L{win}")
        if amt is None or cnt is None:
            continue
        put(f"mg_avg_ticket_{win.lower()}", _safe_ratio(amt, cnt),
            f"average rupees per transaction over {win} — many small tickets "
            f"suggests structuring")

    # --- 7. Alert timing: mules skew nocturnal ------------------------------
    total_alerts = r.get("COUNT_ALERTS")
    buckets = {b: r.get(f"{b}_ALERTS")
               for b in ("MORNING", "AFTERNOON", "EVENING", "NIGHT")}
    if total_alerts is not None:
        for b, s in buckets.items():
            if s is None:
                continue
            put(f"mg_alert_share_{b.lower()}", _safe_ratio(s, total_alerts),
                f"fraction of this account's alerts raised in the {b.lower()} — "
                f"night-hour share runs ~3x higher for mules")
        present = [s for s in buckets.values() if s is not None]
        if len(present) >= 3:
            B = pd.concat(present, axis=1).fillna(0).clip(lower=0)
            p = B.div(B.sum(axis=1) + EPS, axis=0)
            ent = -(p * np.log(p + 1e-9)).sum(axis=1)
            put("mg_alert_time_entropy", ent,
                "spread of alerts across the day; low entropy means a fixed "
                "operating window, typical of a handled account")

    # --- 8. Balance shape: spike and drain ----------------------------------
    for win, (mx, mn, av) in {
        "7d": ("MAX_BAL_7DAYS", "MIN_BAL_7DAYS", "AVG_BAL_7DAYS"),
        "31d": ("MAX_BAL_31DAYS", "MIN_BAL_31DAYS", "AVG_BAL_31DAYS"),
    }.items():
        hi, lo, avg = r.get(mx), r.get(mn), r.get(av)
        if hi is None or lo is None or avg is None:
            continue
        put(f"mg_balance_volatility_{win}", _safe_ratio(hi - lo, avg),
            f"peak-to-trough balance swing over {win} relative to the average — "
            f"money arrives and is drained rather than held")

    # --- 9. Profile mismatch: the occupation-deviation family ---------------
    occ_codes = [c for c in df.columns
                 if D.real_name(c).endswith("_OCC") and c.startswith("F")]
    if occ_codes:
        block = df[occ_codes].apply(pd.to_numeric, errors="coerce")
        put("mg_occ_deviation_mean", block.mean(axis=1),
            "average deviation of this customer's behaviour from their "
            "occupation cohort — the 'occupation-income divergence' signal")
        put("mg_occ_deviation_max", block.abs().max(axis=1),
            "largest single deviation from the occupation cohort norm")
        put("mg_occ_deviation_extreme_count",
            (block.abs() > 3).sum(axis=1).astype(float),
            "how many behaviours sit more than 3 units from the occupation norm")
        report["occupation_deviation_columns_used"] = len(occ_codes)

    report["missing_base_columns"] = sorted(set(r.missing))
    report["base_columns_resolved_by_role"] = r.by_role
    report["base_column_resolution"] = {
        "exact_or_normalised": len(set(r.resolved_exactly)),
        "by_role_inference": len(r.by_role),
        "unresolved": len(set(r.missing)),
        "note": "Columns counted under 'by_role_inference' were matched on what "
                "they measure rather than on their name. That is what lets these "
                "features build on a schema this pipeline has never seen before.",
    }
    report["mule_typology_features"] = added
    if r.by_role:
        log(f"Resolved {len(r.by_role)} base column(s) by ROLE rather than name: "
            f"{list(r.by_role)[:4]}")
    log(f"Added {len(added)} mule-typology features "
        f"({len(set(r.missing))} base columns unavailable after cleaning)")
    return list(added)


def add_row_profile_features(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """Semantics-free aggregates across the whole row.

    These assume nothing about column meaning, so they cannot leak or
    misattribute, and they catch structure the nine named families miss.
    """
    X = df[feat_cols].to_numpy(dtype=np.float32, na_value=np.nan)
    with np.errstate(all="ignore"):
        stats = {
            "mg_row_mean": np.nanmean(X, axis=1),
            "mg_row_std": np.nanstd(X, axis=1),
            "mg_row_max": np.nanmax(X, axis=1),
            "mg_row_min": np.nanmin(X, axis=1),
            "mg_row_nonzero_frac": np.nanmean(X != 0, axis=1),
        }
    stats["mg_row_range"] = stats["mg_row_max"] - stats["mg_row_min"]
    for k, v in stats.items():
        df[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    log(f"Added {len(stats)} generic row-profile features")
    return list(stats)


def detect_edge_columns(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Decide honestly whether a who-paid-whom edge list exists.

    The dictionary settles this: every variable is an aggregate over an account's
    own activity (counts, amounts, ratios, deviations). None names a counterparty.
    We still run the numeric probe so the conclusion rests on the data too.
    """
    n = len(df)
    candidates = []
    for col in feat_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty or s.nunique() <= 0.3 * n:
            continue
        if not np.allclose(s, s.round()):
            continue
        if 0 <= s.min() and s.max() <= n - 1:
            candidates.append(col)

    named_counterparty = [c for c in feat_cols
                          if any(tok in D.real_name(c).upper()
                                 for tok in ("CPARTY", "COUNTERPARTY", "BENEF",
                                             "REMIT", "PAYEE", "PAYER", "VPA"))]
    return {
        "n_accounts": n,
        "edge_columns": candidates,
        "counterparty_named_columns": [D.label(c) for c in named_counterparty],
        "graph_possible": bool(candidates) or bool(named_counterparty),
    }


def main() -> None:
    df = load_frame(C.CLEAN_PARQUET)
    S.bind_target(df, C)
    feat_cols = [c for c in df.columns if c != C.TARGET_COL]
    log(f"Loaded clean matrix: {df.shape[0]:,} x {len(feat_cols):,} features")

    report: dict = {"input_feature_count": len(feat_cols)}

    typology = add_mule_typology_features(df, report)
    profile = add_row_profile_features(df, feat_cols)
    report["added_row_profile_features"] = profile

    # The bank's own 18 shortlisted variables, recorded as a domain prior.
    shortlist = [c for c in D.bank_finalized_codes() if c in df.columns]
    report["bank_shortlisted_present"] = [D.label(c) for c in shortlist]
    log(f"Bank-shortlisted variables still present after cleaning: "
        f"{len(shortlist)}/{len(D.bank_finalized_codes())}")

    edge_info = detect_edge_columns(df, feat_cols)
    report["graph_detection"] = edge_info
    if edge_info["graph_possible"]:
        log(f"Edge-like columns detected: {edge_info['edge_columns'][:5]} "
            f"-> graph stage can run.")
    else:
        log("No counterparty identifiers in the dictionary or the data: every "
            "variable aggregates an account's own activity.")
        log("=> Graph/PageRank/label-propagation SKIPPED (not fabricated). "
            "The behavioural model carries the result.")

    report["output_feature_count"] = len([c for c in df.columns if c != C.TARGET_COL])
    log(f"Feature matrix: {df.shape[0]:,} x {report['output_feature_count']:,} "
        f"(+{len(typology) + len(profile)} engineered)")

    save_frame(df, C.FEATURES_PARQUET)
    save_json(report, C.REPORTS_DIR / "02_features_report.json")


if __name__ == "__main__":
    main()
