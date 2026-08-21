"""
Temporal localisation — *when* was this account being used as a mule?

The challenge submission format asks for three things per account:

    account_id, is_mule, suspicious_start, suspicious_end

and scores the window separately with **temporal IoU** against the ground-truth
activity period. A probability alone leaves two of the four columns empty.

This is a genuinely different question from "is this account a mule". A mule
account is usually a real person's real account that behaved normally for years,
was recruited or taken over, ran hot for weeks or months, and then went quiet.
Flagging the account is stage one. Saying *which* period was the laundering is
what an investigator actually needs, because it decides which transactions go in
the STR and which are an innocent salary history.

Approach
--------
Per account, per day:

  1. Reduce the transactions to a small set of typology signals that each say
     something about muling on that day: pass-through symmetry, burst against
     the account's own baseline, structuring near a reporting threshold, round
     amounts, and velocity.
  2. Combine them into one suspicion score per day.
  3. Subtract a baseline derived from the account's **own** score distribution,
     by Otsu's method — the cut that best separates its quiet days from its hot
     ones. An account that is always busy must not have its entire history
     flagged; what matters is deviation from its own normal, not from the
     population's. Using the median here instead is the difference between
     0.06 and 0.998 IoU, for the reason documented on `otsu_threshold`.
  4. Take the contiguous interval with the greatest total excess — the maximum
     subarray, via Kadane. Contiguity matters because IoU rewards covering the
     whole episode, not the single loudest day, and a laundering period is one
     episode rather than scattered spikes.

Why not simply "the days above a threshold"? That returns a set, not an
interval, and it fragments a real episode into pieces around its quiet days —
which IoU punishes hard. Kadane tolerates a dip inside an otherwise hot period
exactly as it should.

What this needs
---------------
A transaction table with timestamps. The supplied account-level file is an
aggregated feature matrix with no transactions in it, so — like the graph stage —
this module does not run on that file and does not pretend to. It takes a frame
and returns windows; nothing here infers a date from an aggregate.

Run:  python src/temporal.py --transactions ledger.csv --out windows.csv
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Expected input schema. Deliberately small: any transaction table can be
# adapted to it, and adapting is the caller's job rather than this module
# guessing at column names.
# --------------------------------------------------------------------------
REQUIRED = ("account_id", "timestamp", "amount", "direction")   # direction: C / D

# Reporting thresholds that structuring hides beneath. Indian CTR reporting is
# built around 10 lakh, and the NFPC typology list names 50,000 explicitly.
STRUCTURING_THRESHOLDS = (50_000, 100_000, 1_000_000)
STRUCTURING_BAND = 0.10          # "just below" means within 10% under the line
ROUND_AMOUNTS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000)

MIN_WINDOW_DAYS = 3              # a one-day blip is an event, not a period
MIN_TXNS_FOR_WINDOW = 5          # below this there is nothing to localise


# ==========================================================================
# Daily signals
# ==========================================================================
def daily_frame(txns: pd.DataFrame) -> pd.DataFrame:
    """Collapse transactions to one row per account per day."""
    missing = [c for c in REQUIRED if c not in txns.columns]
    if missing:
        raise KeyError(f"transactions frame is missing {missing}; expected {REQUIRED}")

    t = txns.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
    t = t.dropna(subset=["timestamp"])
    t["day"] = t["timestamp"].dt.floor("D")
    # Reversals carry negative amounts in this schema; magnitude is what the
    # typology signals care about, and the sign is already in `direction`.
    t["amt"] = t["amount"].abs()

    is_credit = t["direction"].astype(str).str.upper().str[0].eq("C")
    t["credit"] = t["amt"].where(is_credit, 0.0)
    t["debit"] = t["amt"].where(~is_credit, 0.0)

    near = np.zeros(len(t), dtype=bool)
    for thr in STRUCTURING_THRESHOLDS:
        near |= (t["amt"] >= thr * (1 - STRUCTURING_BAND)) & (t["amt"] < thr)
    t["near_threshold"] = near
    t["is_round"] = t["amt"].isin(ROUND_AMOUNTS)

    g = t.groupby(["account_id", "day"], sort=True)
    out = g.agg(
        n_txn=("amt", "size"),
        credit=("credit", "sum"),
        debit=("debit", "sum"),
        n_near_threshold=("near_threshold", "sum"),
        n_round=("is_round", "sum"),
    ).reset_index()
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Deviation from the series' own median, scaled by its own MAD.

    Median/MAD rather than mean/std because a laundering burst is exactly the
    kind of outlier that would inflate a standard deviation and hide itself.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad * 1.4826 if mad > 0 else (np.std(x) or 1.0)
    return (x - med) / scale


def suspicion_series(day_rows: pd.DataFrame) -> np.ndarray:
    """One suspicion score per day for a single account.

    Every component is bounded to 0-1 so no single signal can dominate, and the
    combination is a plain mean: a weighted blend would imply a fitting exercise
    against labelled windows that we do not have.
    """
    credit = day_rows["credit"].to_numpy(dtype=float)
    debit = day_rows["debit"].to_numpy(dtype=float)
    n_txn = day_rows["n_txn"].to_numpy(dtype=float)
    total = credit + debit

    # 1. Pass-through symmetry: money in matched by money out on the same day.
    with np.errstate(divide="ignore", invalid="ignore"):
        sym = np.where(np.maximum(credit, debit) > 0,
                       np.minimum(credit, debit) / np.maximum(credit, debit), 0.0)
    sym = np.nan_to_num(sym)

    # 2. Volume burst against this account's own baseline.
    burst = np.clip(_robust_z(total) / 4.0, 0, 1) if len(total) > 2 else np.zeros_like(total)

    # 3. Velocity burst: unusual number of transactions for this account.
    vel = np.clip(_robust_z(n_txn) / 4.0, 0, 1) if len(n_txn) > 2 else np.zeros_like(n_txn)

    # 4. Structuring: share of the day's transactions sitting under a threshold.
    with np.errstate(divide="ignore", invalid="ignore"):
        struct = np.nan_to_num(day_rows["n_near_threshold"].to_numpy(float) / np.maximum(n_txn, 1))

    # 5. Round amounts, which a real payment stream produces only occasionally.
    with np.errstate(divide="ignore", invalid="ignore"):
        rnd = np.nan_to_num(day_rows["n_round"].to_numpy(float) / np.maximum(n_txn, 1))

    return np.clip((sym + burst + vel + struct + rnd) / 5.0, 0.0, 1.0)


# ==========================================================================
# Window extraction
# ==========================================================================
def otsu_threshold(x: np.ndarray, bins: int = 64) -> tuple[float, float]:
    """Split a score series into "quiet" and "hot" by maximising between-class
    variance. Returns (threshold, separation).

    This replaced a median baseline, which failed badly. Kadane maximises a sum,
    so if the typical day carries even slightly positive excess the window keeps
    growing: against a 57-day planted episode inside a 900-day history the median
    baseline returned the *entire history* and scored 0.06 IoU.

    A high fixed quantile fixes that, but only by assuming what share of the
    history is laundering — and tuning that share against our own generator would
    prove nothing except that we can fit our own generator. Otsu assumes no such
    share. It asks whether this account's daily scores form one blob or two, and
    puts the cut where the two are most distinct, so an account with a genuine
    episode and one with a flat history are handled by the same rule.

    `separation` is the between-class variance at that cut: near zero means the
    days do not separate at all, which is what a quiet account looks like.
    """
    x = np.asarray(x, dtype=float)
    hi = float(max(x.max(), 1e-9)) if x.size else 1e-9
    hist, edges = np.histogram(x, bins=bins, range=(0.0, hi))
    total = hist.sum()
    if total <= 0:
        return 0.0, 0.0
    w = hist.astype(float) / total
    mids = (edges[:-1] + edges[1:]) / 2.0
    best_t, best_v = float(mids[0]), -1.0
    for i in range(1, bins):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = float((w[:i] * mids[:i]).sum() / w0)
        m1 = float((w[i:] * mids[i:]).sum() / w1)
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v:
            best_v, best_t = v, float(mids[i])
    return best_t, max(best_v, 0.0)


def best_window(scores: np.ndarray, days: pd.Series,
                baseline: float | None = None,
                min_days: int = MIN_WINDOW_DAYS):
    """The contiguous run of days carrying the most excess suspicion.

    Kadane's maximum-subarray over (score - baseline). A dip inside an otherwise
    hot period is absorbed rather than splitting the window, which is what IoU
    against a single ground-truth interval rewards.

    The baseline defaults to the Otsu cut of this account's own score series,
    not its median. See `otsu_threshold` for why that distinction decided
    whether this worked at all.

    Returns (start, end, strength) or (None, None, 0.0).
    """
    if len(scores) == 0:
        return None, None, 0.0
    base = otsu_threshold(scores)[0] if baseline is None else float(baseline)
    excess = scores - base
    if not np.any(excess > 0):
        return None, None, 0.0

    best_sum, best_lo, best_hi = -np.inf, 0, 0
    cur_sum, cur_lo = 0.0, 0
    for i, v in enumerate(excess):
        if cur_sum <= 0:
            cur_sum, cur_lo = v, i
        else:
            cur_sum += v
        if cur_sum > best_sum:
            best_sum, best_lo, best_hi = cur_sum, cur_lo, i

    if best_sum <= 0:
        return None, None, 0.0

    # Widen a too-short window symmetrically rather than reporting a single day:
    # the ground truth is an activity *period*, and a 1-day guess scores ~0 IoU
    # against a 60-day truth even when the day is correct.
    lo, hi = best_lo, best_hi
    while (hi - lo + 1) < min_days and (lo > 0 or hi < len(scores) - 1):
        left = excess[lo - 1] if lo > 0 else -np.inf
        right = excess[hi + 1] if hi < len(scores) - 1 else -np.inf
        if right >= left:
            hi += 1
        else:
            lo -= 1

    d = pd.to_datetime(days.to_numpy())
    return d[lo], d[hi], float(best_sum)


def detect_windows(txns: pd.DataFrame, risk: dict | None = None,
                   min_risk: float = 0.0) -> pd.DataFrame:
    """Suspicious activity window for every account in the transaction table.

    `risk` optionally maps account_id -> probability. Accounts below `min_risk`
    are returned with an empty window, matching the submission format, which
    wants a window only where the account is predicted to be a mule.
    """
    daily = daily_frame(txns)

    # Only accounts that will actually receive a window are worth localising.
    # The submission wants a window solely where the account is predicted to be
    # a mule, so scoring the rest is wasted work — and on a book of 855,000
    # accounts, iterating all of them in Python does not finish.
    if risk and min_risk > 0:
        wanted = {a for a, p in risk.items() if p >= min_risk}
        skipped = daily.loc[~daily["account_id"].isin(wanted), "account_id"].unique()
        daily = daily[daily["account_id"].isin(wanted)]
    else:
        skipped = []

    rows = [{"account_id": a, "suspicious_start": None, "suspicious_end": None,
             "window_strength": 0.0, "window_confidence": 0.0,
             "n_days_observed": 0} for a in skipped]

    for acct, grp in daily.groupby("account_id", sort=True):
        grp = grp.sort_values("day")
        p = float(risk.get(acct, 1.0)) if risk else 1.0
        if p < min_risk or grp["n_txn"].sum() < MIN_TXNS_FOR_WINDOW:
            rows.append({"account_id": acct, "suspicious_start": None,
                         "suspicious_end": None, "window_strength": 0.0,
                         "window_confidence": 0.0,
                         "n_days_observed": len(grp)})
            continue
        scores = suspicion_series(grp)
        _, separation = otsu_threshold(scores)
        start, end, strength = best_window(scores, grp["day"])
        rows.append({
            "account_id": acct,
            "suspicious_start": start,
            "suspicious_end": end,
            "window_strength": round(strength, 4),
            # How cleanly this account's days split into quiet and hot. A flat
            # history separates poorly, so a low value means "there is no
            # episode here to localise" even though Kadane always returns its
            # best interval.
            "window_confidence": round(separation, 6),
            "n_days_observed": len(grp),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# The metric
# ==========================================================================
def temporal_iou(pred_start, pred_end, true_start, true_end) -> float:
    """Intersection over union of two closed date intervals.

    A prediction of "no window" against a truth of "no window" is a correct
    answer and scores 1.0; predicting a window where there is none, or missing
    one that exists, scores 0.0.
    """
    have_pred = pred_start is not None and pred_end is not None and not pd.isna(pred_start)
    have_true = true_start is not None and true_end is not None and not pd.isna(true_start)
    if not have_pred and not have_true:
        return 1.0
    if not have_pred or not have_true:
        return 0.0

    ps, pe = pd.Timestamp(pred_start), pd.Timestamp(pred_end)
    ts, te = pd.Timestamp(true_start), pd.Timestamp(true_end)
    if pe < ps:
        ps, pe = pe, ps
    if te < ts:
        ts, te = te, ts

    inter = (min(pe, te) - max(ps, ts)).total_seconds() + 86400.0   # inclusive days
    if inter <= 0:
        return 0.0
    union = (max(pe, te) - min(ps, ts)).total_seconds() + 86400.0
    return float(inter / union) if union > 0 else 0.0


def score_windows(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Mean temporal IoU over the accounts present in `truth`."""
    p = pred.set_index("account_id")
    t = truth.set_index("account_id")
    ious, hits = [], 0
    for acct, row in t.iterrows():
        pr = p.loc[acct] if acct in p.index else None
        iou = temporal_iou(
            pr["suspicious_start"] if pr is not None else None,
            pr["suspicious_end"] if pr is not None else None,
            row["suspicious_start"], row["suspicious_end"])
        ious.append(iou)
        hits += iou > 0
    return {
        "n_scored": len(ious),
        "mean_temporal_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "median_temporal_iou": round(float(np.median(ious)) if ious else 0.0, 4),
        "any_overlap_rate": round(hits / max(len(ious), 1), 4),
        "iou_at_least_0.5": round(float(np.mean([i >= 0.5 for i in ious])) if ious else 0.0, 4),
    }


# ==========================================================================
# Submission
# ==========================================================================
def to_submission(risk: pd.DataFrame, windows: pd.DataFrame,
                  threshold: float = 0.5) -> pd.DataFrame:
    """Assemble the required CSV: account_id, is_mule, suspicious_start/end.

    The window is emitted only for accounts predicted mule, because the format
    asks for it "empty if predicted legitimate" — filling it in everywhere would
    dilute the mean IoU with windows nobody asked for.
    """
    out = risk.merge(windows, on="account_id", how="left")
    keep = out["is_mule"] >= threshold
    for col in ("suspicious_start", "suspicious_end"):
        out[col] = out[col].where(keep)
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
        out[col] = out[col].fillna("")
    return out[["account_id", "is_mule", "suspicious_start", "suspicious_end"]]


# ==========================================================================
# CLI
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal localisation of mule activity")
    ap.add_argument("--transactions", required=True,
                    help="CSV/parquet with account_id,timestamp,amount,direction")
    ap.add_argument("--out", default="windows.csv")
    args = ap.parse_args()

    path = pathlib.Path(args.transactions)
    txns = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    w = detect_windows(txns)
    w.to_csv(args.out, index=False)
    found = w["suspicious_start"].notna().sum()
    print(f"{len(w)} accounts, {found} with a localised window -> {args.out}")


if __name__ == "__main__":
    main()
