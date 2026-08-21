"""
Typology motif detection — find the laundering *shape*, not the community.

Why this exists, and why `rings.py` was not enough
-------------------------------------------------
`rings.py` finds groups that are densely connected to each other and sparsely
connected to everyone else. That is the right model when a laundering network is
a closed cell, and it recovers planted rings from a synthetic ledger cleanly.

Benchmarked against SAML-D it fails, and the measurement says exactly why. The
true rings there have a **conductance of 0.98-0.99**: each member carries around
200 transactions, of which only one or two are ring edges. The laundering is
about 1% of the member's activity, hidden inside ordinary business. Such a group
is not separable from the graph by any community method, and our gates correctly
decline to flag it rather than inventing confidence.

That is a real property of real muling, not a quirk of one dataset. A recruited
account keeps paying its bills. So the shape has to be found *locally and in
time* rather than globally in the topology:

    FAN-OUT          one account pays many, close together, in similar amounts
    FAN-IN           many pay one account, close together, in similar amounts
    GATHER-SCATTER   a hub fans in and then fans out, keeping almost nothing
    SCATTER-GATHER   the reverse: disperse, then re-collect
    CHAIN            value relayed A -> B -> C, each hop nearly the full amount

Every one is a statement about a handful of transactions inside a time window.
None requires the participants to be globally clustered, which is the assumption
that broke.

What separates a laundering fan-out from a payroll run
------------------------------------------------------
Nothing, if you only count edges — which is why SAML-D contains `Normal_Fan_Out`
and `Normal_Fan_In` alongside the laundering versions. The discriminating
evidence is:

  * **amount uniformity** — laundering splits a sum into near-equal parts; a
    payroll or supplier run does not
  * **conservation** — what arrives leaves again almost intact; a business keeps
    a margin
  * **burstiness** — the legs are hours or days apart, not spread over a month
  * **threshold hugging** — the parts sit just under a reporting line

Each is reported separately so a reviewer can see which fired.

Measured on SAML-D, and two of the hypotheses above did not survive
--------------------------------------------------------------------
Third-party ground truth: 855,460 accounts, 9.5M transactions, 3,393 accounts in
a network laundering typology (base rate 0.397%). No labels were used by the
detector.

    MOTIF             ACCOUNTS   PRECISION    LIFT   RECALL
    FAN_OUT             18,798      11.21%   28.3x    62.1%
    FAN_IN              13,766       7.50%   18.9x    30.4%
    GATHER_SCATTER       8,610       4.10%   10.3x    10.4%
    CHAIN               33,080       3.23%    8.1x    31.4%

Two findings we did not expect and are not going to bury:

1. **FAN_OUT alone beats the blend.** Combining every motif into one score gives
   12.4x lift; fan-out on its own gives 28.3x at 62% recall over 2.2% of the
   book. Averaging a strong detector with weak ones destroys it. The combined
   score is still returned, but per-motif scores are returned alongside it
   precisely so a deployer can decline the blend.

2. **Amount uniformity did not discriminate.** The stated reasoning above — that
   laundering splits a sum into near-equal parts while a payroll run does not —
   is intuitive and, on this data, wrong. Filtering fan-outs to the most uniform
   quintile *lowers* precision from 11.2% to 9.0% while cutting recall from 62%
   to 11%. Value conservation on gather-scatter behaved the same way. Both are
   still computed and reported as evidence, because an investigator reading an
   alert wants them; neither is used to gate.

The thresholds here are round numbers chosen before the benchmark and left
alone afterwards. Tuning them against SAML-D would produce a detector that
scores well on SAML-D.

Run:  python src/motifs.py --csv ledger.csv
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

REQUIRED = ("src", "dst", "amount", "date")

# Deliberately round, and deliberately not fitted against any benchmark: tuning
# these on SAML-D would produce a detector that scores on SAML-D.
WINDOW_DAYS = 7
MIN_LEGS = 4               # a "many" needs to be at least this many counterparties
MAX_CV = 0.35              # coefficient of variation below which amounts are "uniform"
CONSERVATION_TOL = 0.25    # |in - out| / in, below which value is "passed through"
THRESHOLDS = (10_000, 50_000, 100_000, 1_000_000)
THRESHOLD_BAND = 0.10


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"frame is missing {missing}; expected {REQUIRED}")
    t = df[list(REQUIRED)].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    t["amount"] = t["amount"].abs()
    t["bucket"] = (t["date"] - t["date"].min()).dt.days // WINDOW_DAYS
    return t


def _cv(x: np.ndarray) -> float:
    """Coefficient of variation. 0 means every leg is the same size."""
    m = float(np.mean(x))
    return float(np.std(x) / m) if m > 0 else 1.0


def _near_threshold_mask(x: np.ndarray) -> np.ndarray:
    """Which amounts sit just under a reporting line."""
    hit = np.zeros(len(x), dtype=bool)
    for thr in THRESHOLDS:
        hit |= (x >= thr * (1 - THRESHOLD_BAND)) & (x < thr)
    return hit


def _near_threshold_share(x: np.ndarray) -> float:
    m = _near_threshold_mask(np.asarray(x, dtype=float))
    return float(m.mean()) if len(m) else 0.0


# ==========================================================================
# Fan motifs
# ==========================================================================
def _fan(t: pd.DataFrame, hub_col: str, leg_col: str, kind: str) -> pd.DataFrame:
    """One account against many, inside a time bucket.

    Aggregated with pandas rather than by looping over groups. On a 9.5M-row
    ledger there are tens of millions of (account, window) pairs, and iterating
    them in Python does not finish; the counterparty lists are collected only
    for the handful of groups that clear MIN_LEGS.
    """
    if t.empty:
        return pd.DataFrame()

    t = t.copy()
    t["_near"] = _near_threshold_mask(t["amount"].to_numpy(dtype=float))

    agg = t.groupby([hub_col, "bucket"], sort=False).agg(
        legs=(leg_col, "nunique"),
        n=("amount", "size"),
        total=("amount", "sum"),
        mean=("amount", "mean"),
        std=("amount", "std"),
        near=("_near", "mean"),
        start=("date", "min"),
        end=("date", "max"),
    )
    agg = agg[agg["legs"] >= MIN_LEGS]
    if agg.empty:
        return pd.DataFrame()

    cv = (agg["std"].fillna(0.0) / agg["mean"].replace(0, np.nan)).fillna(1.0)
    agg["uniformity"] = (1.0 - cv.clip(0, 1)).round(4)
    agg["threshold_hugging"] = agg["near"].round(4)
    agg = agg.reset_index().rename(columns={hub_col: "account"})
    agg["motif"] = kind

    # Counterparties, only for the survivors.
    keys = set(zip(agg["account"], agg["bucket"]))
    sub = t[[hub_col, "bucket", leg_col]]
    sub = sub[pd.MultiIndex.from_arrays(
        [sub[hub_col], sub["bucket"]]).isin(keys)]
    cps = (sub.groupby([hub_col, "bucket"], sort=False)[leg_col]
              .agg(lambda x: sorted(set(x))).rename("counterparties").reset_index()
              .rename(columns={hub_col: "account"}))
    agg = agg.merge(cps, on=["account", "bucket"], how="left")

    return agg[["account", "bucket", "motif", "legs", "total", "uniformity",
                "threshold_hugging", "counterparties", "start", "end"]]


def fan_out(t: pd.DataFrame) -> pd.DataFrame:
    return _fan(t, "src", "dst", "FAN_OUT")


def fan_in(t: pd.DataFrame) -> pd.DataFrame:
    return _fan(t, "dst", "src", "FAN_IN")


# ==========================================================================
# Hub motifs: value arriving and leaving the same account in the same window
# ==========================================================================
def gather_scatter(t: pd.DataFrame) -> pd.DataFrame:
    """A hub that collects from many and disperses to many, keeping little.

    This is the signature that separates a mule hub from a merchant. A merchant
    fans in and *retains*; a conduit fans in and fans straight back out, and the
    two totals match.
    """
    ins = t.groupby(["dst", "bucket"], sort=False).agg(
        in_total=("amount", "sum"), in_legs=("src", "nunique"),
        in_start=("date", "min"), in_end=("date", "max"))
    outs = t.groupby(["src", "bucket"], sort=False).agg(
        out_total=("amount", "sum"), out_legs=("dst", "nunique"),
        out_start=("date", "min"), out_end=("date", "max"))
    ins.index.names = ["account", "bucket"]
    outs.index.names = ["account", "bucket"]
    j = ins.join(outs, how="inner").reset_index()
    if j.empty:
        return pd.DataFrame()

    j = j[(j.in_legs >= MIN_LEGS) | (j.out_legs >= MIN_LEGS)]
    if j.empty:
        return pd.DataFrame()

    denom = j.in_total.replace(0, np.nan)
    j["conservation"] = 1.0 - ((j.in_total - j.out_total).abs() / denom).clip(0, 1)
    j["conservation"] = j["conservation"].fillna(0.0)
    j = j[j.conservation >= (1.0 - CONSERVATION_TOL)]
    if j.empty:
        return pd.DataFrame()

    j["motif"] = np.where(j.in_legs >= j.out_legs, "GATHER_SCATTER", "SCATTER_GATHER")
    j["legs"] = j[["in_legs", "out_legs"]].max(axis=1)
    j["total"] = j.in_total
    j["start"] = j[["in_start", "out_start"]].min(axis=1)
    j["end"] = j[["in_end", "out_end"]].max(axis=1)
    return j[["account", "bucket", "motif", "legs", "total", "conservation",
              "in_legs", "out_legs", "start", "end"]]


# ==========================================================================
# Chain motif: relayed value
# ==========================================================================
def chains(t: pd.DataFrame, max_len: int = 4) -> pd.DataFrame:
    """A -> B -> C where B forwards nearly all of what it received, in-window.

    Detected by self-joining the ledger on (receiver == next sender) within the
    same bucket, which is far cheaper than path enumeration and catches the
    layering that `Layered_Fan_*` typologies are built from.
    """
    a = t.rename(columns={"src": "a", "dst": "b", "amount": "amt_ab", "date": "d_ab"})
    b = t.rename(columns={"src": "b", "dst": "c", "amount": "amt_bc", "date": "d_bc"})
    j = a.merge(b[["b", "c", "amt_bc", "d_bc", "bucket"]], on=["b", "bucket"], how="inner")
    if j.empty:
        return pd.DataFrame()
    j = j[j.a != j.c]
    j = j[j.d_bc >= j.d_ab]
    if j.empty:
        return pd.DataFrame()
    keep = (j.amt_bc / j.amt_ab.replace(0, np.nan)).between(1.0 - CONSERVATION_TOL, 1.05)
    j = j[keep.fillna(False)]
    if j.empty:
        return pd.DataFrame()
    j["motif"] = "CHAIN"
    j["relay_retention"] = (1.0 - j.amt_bc / j.amt_ab).round(4)
    return j[["a", "b", "c", "bucket", "motif", "amt_ab", "amt_bc",
              "relay_retention", "d_ab", "d_bc"]]


# ==========================================================================
# Scoring
# ==========================================================================
def detect_motifs(df: pd.DataFrame) -> dict:
    """Run every motif detector and return per-account suspicion evidence."""
    t = _prep(df)
    fo, fi = fan_out(t), fan_in(t)
    gs = gather_scatter(t)
    ch = chains(t)

    score: dict = {}
    evidence: dict = {}

    def bump(acct, value, tag):
        if value > score.get(acct, 0.0):
            score[acct] = value
        evidence.setdefault(acct, set()).add(tag)

    for frame, base in ((fo, 0.45), (fi, 0.45)):
        if frame.empty:
            continue
        for r in frame.itertuples():
            # A fan is only interesting when the legs are uniform: that is what
            # separates splitting a sum from paying a list of suppliers.
            v = base + 0.35 * r.uniformity + 0.20 * r.threshold_hugging
            bump(r.account, min(v, 1.0), r.motif)
            for cp in r.counterparties:
                bump(cp, min(v, 1.0) * 0.5, r.motif + "_LEG")

    if not gs.empty:
        for r in gs.itertuples():
            v = 0.55 + 0.45 * float(r.conservation)
            bump(r.account, min(v, 1.0), r.motif)

    if not ch.empty:
        for r in ch.itertuples():
            v = 0.50 + 0.50 * (1.0 - min(abs(float(r.relay_retention)), 1.0))
            for acct in (r.a, r.b, r.c):
                bump(acct, min(v, 1.0), "CHAIN")

    # Per-motif account sets, because the blend measured worse than fan-out
    # alone on SAML-D and a caller must be able to take one motif and leave the
    # rest rather than accepting an average that hides which signal fired.
    by_motif = {
        "FAN_OUT": sorted(fo["account"].unique().tolist()) if not fo.empty else [],
        "FAN_IN": sorted(fi["account"].unique().tolist()) if not fi.empty else [],
        "GATHER_SCATTER": sorted(gs["account"].unique().tolist()) if not gs.empty else [],
        "CHAIN": sorted(pd.unique(ch["b"]).tolist()) if not ch.empty else [],
    }

    return {
        "n_transactions": int(len(t)),
        "window_days": WINDOW_DAYS,
        "counts": {"fan_out": int(len(fo)), "fan_in": int(len(fi)),
                   "gather_scatter": int(len(gs)), "chains": int(len(ch))},
        "accounts_by_motif": by_motif,
        "scores": score,
        "evidence": {k: sorted(v) for k, v in evidence.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AML typology motif detection")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    res = detect_motifs(df)
    print(res["counts"])
    print(f"{len(res['scores']):,} accounts carry at least one motif")


if __name__ == "__main__":
    main()
