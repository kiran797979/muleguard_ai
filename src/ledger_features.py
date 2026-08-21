"""
Behavioural features built directly from a transaction ledger.

Why this exists
---------------
`02_features.py` builds the 29 mule-typology features from an account-level
feature matrix — pre-aggregated columns like TOT_TXNAMT_CR_L7D. That is the
right thing when a bank hands you its aggregates, and useless when it hands you
a transaction table instead.

This computes the same ideas from raw transactions: pass-through, burst,
velocity, ticket size, counterparty concentration, round-amount and
threshold-hugging behaviour. It is what lets the whole system run on a ledger,
which is the form the network and temporal components need anyway.

Everything here is a groupby. No loops over accounts: on a 9.5M-row ledger with
855,000 accounts, per-account iteration does not finish, and that lesson was
learned the hard way twice already in this project.

Input schema
------------
    src, dst, amount, timestamp      one row per transfer

An account is whichever side of a transfer it sits on, so every account is
aggregated twice — once as sender, once as receiver — and the two halves joined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("src", "dst", "amount", "timestamp")

ROUND_AMOUNTS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000)
THRESHOLDS = (10_000, 50_000, 100_000, 1_000_000)
THRESHOLD_BAND = 0.10
NIGHT_HOURS = set(range(0, 6))


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"ledger is missing {missing}; expected {REQUIRED}")
    t = df[list(REQUIRED)].copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
    t = t.dropna(subset=["timestamp"])
    t["amount"] = t["amount"].abs()
    t["day"] = t["timestamp"].dt.normalize()
    t["hour"] = t["timestamp"].dt.hour

    amt = t["amount"].to_numpy(dtype=float)
    t["is_round"] = np.isin(amt, ROUND_AMOUNTS)
    near = np.zeros(len(t), dtype=bool)
    for thr in THRESHOLDS:
        near |= (amt >= thr * (1 - THRESHOLD_BAND)) & (amt < thr)
    t["near_threshold"] = near
    t["is_night"] = t["hour"].isin(NIGHT_HOURS)
    return t


def _side(t: pd.DataFrame, key: str, other: str, tag: str) -> pd.DataFrame:
    """Aggregate one side of the ledger: the account as sender, or as receiver."""
    g = t.groupby(key, sort=False)
    out = g.agg(**{
        f"n_{tag}": ("amount", "size"),
        f"sum_{tag}": ("amount", "sum"),
        f"mean_{tag}": ("amount", "mean"),
        f"max_{tag}": ("amount", "max"),
        f"std_{tag}": ("amount", "std"),
        f"cp_{tag}": (other, "nunique"),
        f"days_{tag}": ("day", "nunique"),
        f"round_{tag}": ("is_round", "mean"),
        f"near_{tag}": ("near_threshold", "mean"),
        f"night_{tag}": ("is_night", "mean"),
        f"first_{tag}": ("timestamp", "min"),
        f"last_{tag}": ("timestamp", "max"),
    })
    out.index.name = "account_id"
    return out


def build(txns: pd.DataFrame) -> pd.DataFrame:
    """One row per account, with the behavioural signature a mule leaves.

    Ratios are preferred over absolute magnitudes throughout, for the same
    reason the account-level pipeline prefers them: a ratio survives a change of
    currency, of bank, and of which fields an extract happened to populate.
    """
    t = _prep(txns)
    debit = _side(t, "src", "dst", "out")     # money leaving the account
    credit = _side(t, "dst", "src", "in")     # money arriving

    f = credit.join(debit, how="outer")
    num = f.select_dtypes(include=[np.number]).columns
    f[num] = f[num].fillna(0.0)

    sin, sout = f["sum_in"], f["sum_out"]
    nin, nout = f["n_in"], f["n_out"]
    total = sin + sout
    denom = np.maximum(sin, sout).replace(0, np.nan)

    # --- the conduit signature -------------------------------------------
    # 1.0 means every rupee that arrived left again.
    f["passthrough"] = (np.minimum(sin, sout) / denom).fillna(0.0)
    f["net_flow"] = ((sin - sout) / total.replace(0, np.nan)).fillna(0.0)
    f["turnover"] = total
    # Retention: what share of arriving value stayed. A conduit keeps ~nothing.
    f["retention"] = ((sin - sout).abs() / sin.replace(0, np.nan)).fillna(1.0).clip(0, 1)

    # --- shape of the traffic --------------------------------------------
    f["txn_count"] = nin + nout
    f["in_out_ratio"] = (nin / nout.replace(0, np.nan)).fillna(0.0).clip(0, 50)
    f["avg_ticket_in"] = f["mean_in"]
    f["avg_ticket_out"] = f["mean_out"]
    f["ticket_cv_in"] = (f["std_in"] / f["mean_in"].replace(0, np.nan)).fillna(0.0)
    f["ticket_cv_out"] = (f["std_out"] / f["mean_out"].replace(0, np.nan)).fillna(0.0)

    # --- counterparties ---------------------------------------------------
    f["counterparties"] = f["cp_in"] + f["cp_out"]
    # Transfers per distinct counterparty. A mule fans wide and shallow; a
    # household pays the same few counterparties repeatedly.
    f["txn_per_counterparty"] = (f["txn_count"] /
                                 f["counterparties"].replace(0, np.nan)).fillna(0.0)
    f["fan_in_ratio"] = (f["cp_in"] / f["counterparties"].replace(0, np.nan)).fillna(0.0)

    # --- burst and velocity ------------------------------------------------
    first = f[["first_in", "first_out"]].min(axis=1)
    last = f[["last_in", "last_out"]].max(axis=1)
    span = (last - first).dt.total_seconds() / 86400.0
    f["active_days"] = np.maximum(f["days_in"], f["days_out"])
    f["span_days"] = span.fillna(0.0)
    # Concentration in time: 1.0 means the whole history happened on one day.
    f["burstiness"] = (f["active_days"] /
                       f["span_days"].replace(0, np.nan)).fillna(1.0).clip(0, 1)
    f["txn_per_active_day"] = (f["txn_count"] /
                               f["active_days"].replace(0, np.nan)).fillna(0.0)

    # --- tradecraft tells --------------------------------------------------
    w_in = (nin / (nin + nout).replace(0, np.nan)).fillna(0.5)
    f["round_share"] = f["round_in"] * w_in + f["round_out"] * (1 - w_in)
    f["threshold_share"] = f["near_in"] * w_in + f["near_out"] * (1 - w_in)
    f["night_share"] = f["night_in"] * w_in + f["night_out"] * (1 - w_in)

    keep = ["passthrough", "net_flow", "turnover", "retention", "txn_count",
            "in_out_ratio", "avg_ticket_in", "avg_ticket_out", "ticket_cv_in",
            "ticket_cv_out", "counterparties", "txn_per_counterparty",
            "fan_in_ratio", "active_days", "span_days", "burstiness",
            "txn_per_active_day", "round_share", "threshold_share", "night_share",
            "n_in", "n_out", "sum_in", "sum_out", "max_in", "max_out",
            "cp_in", "cp_out"]
    out = f[keep].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out.index.name = "account_id"
    return out


FEATURE_MEANINGS = {
    "passthrough": "credit/debit symmetry - 1.0 means every unit in left again",
    "net_flow": "normalised net flow; near 0 is a pure conduit",
    "retention": "share of arriving value that stayed; a conduit keeps ~nothing",
    "burstiness": "activity concentrated into few days of a long span",
    "txn_per_counterparty": "fan-wide-and-shallow versus repeat relationships",
    "threshold_share": "transfers sitting just under a reporting line",
    "round_share": "disproportionate use of exact round amounts",
    "night_share": "share of activity in the small hours",
    "ticket_cv_in": "uniformity of incoming amounts; a split sum is uniform",
}
