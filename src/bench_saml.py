"""
Benchmark structural ring detection against SAML-D — third-party ground truth.

Everything `rings.py` has been measured against so far was planted by our own
generator and then found by our own code. That proves the code path executes and
nothing else, and we said so. This is the correction.

SAML-D (Oztas et al., IEEE ICEBE 2023) is an independent synthetic AML dataset:
9.5M transactions, 855K accounts, and — the part that matters — a
`Laundering_type` column naming 28 typologies. Critically it includes both the
laundering network shapes AND their innocent look-alikes:

    Layered_Fan_Out   vs   Normal_Fan_Out
    Layered_Fan_In    vs   Normal_Fan_In
    Stacked Bipartite vs   Normal_Group

Distinguishing those is the actual problem. A detector that flags every dense
cluster scores well against "is there a cluster here" and is useless to a bank.

What is and is not a fair test
------------------------------
The typologies split into two kinds, and we report them separately because
conflating them would flatter the result:

  NETWORK SHAPES     Layered_Fan_In/Out, Stacked Bipartite, Bipartite, Cycle,
                     Gather-Scatter, Scatter-Gather, Fan_In, Fan_Out.
                     These are multi-account structures. Ring detection should
                     find them.

  SINGLE-ACCOUNT     Structuring, Smurfing, Cash_Withdrawal, Deposit-Send,
                     Behavioural_Change, Single_large, Over-Invoicing.
                     These are behaviours of one account over time. A community
                     detector has no business finding them, and credit for
                     catching them would be accidental.

Run:  python src/bench_saml.py --csv "D:/SAML-D.csv/SAML-D.csv"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

import config as C
import rings as R
from utils import log

OUT_DIR = C.REPORTS_DIR / "bench_saml"

NETWORK_TYPOLOGIES = {
    "Layered_Fan_In", "Layered_Fan_Out", "Stacked Bipartite", "Bipartite",
    "Cycle", "Gather-Scatter", "Scatter-Gather", "Fan_In", "Fan_Out",
}
USECOLS = ["Date", "Sender_account", "Receiver_account", "Amount",
           "Is_laundering", "Laundering_type"]


def load(path: pathlib.Path, months: int | None) -> pd.DataFrame:
    log(f"Loading {path} ...")
    t0 = time.time()
    df = pd.read_csv(path, usecols=USECOLS)
    log(f"  {len(df):,} transactions in {time.time() - t0:.1f}s")
    if months:
        df["Date"] = pd.to_datetime(df["Date"])
        cutoff = df["Date"].min() + pd.DateOffset(months=months)
        df = df[df["Date"] < cutoff]
        log(f"  restricted to the first {months} month(s): {len(df):,} transactions")
    return df


def truth_sets(df: pd.DataFrame) -> tuple[set, set, dict]:
    """Accounts touched by laundering, split by whether it is a network shape."""
    laund = df[df.Is_laundering == 1]
    net = laund[laund.Laundering_type.isin(NETWORK_TYPOLOGIES)]

    def accounts(frame):
        return set(frame.Sender_account.unique()) | set(frame.Receiver_account.unique())

    per_typology = {
        t: accounts(g) for t, g in laund.groupby("Laundering_type")
    }
    return accounts(net), accounts(laund), per_typology


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"D:/SAML-D.csv/SAML-D.csv")
    ap.add_argument("--months", type=int, default=None,
                    help="restrict to the first N months (whole file if omitted)")
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load(pathlib.Path(args.csv), args.months)

    net_truth, all_truth, per_typ = truth_sets(df)
    log(f"Ground truth: {len(all_truth):,} accounts touched by laundering, "
        f"of which {len(net_truth):,} by a NETWORK typology")

    edges = list(zip(df.Sender_account.to_numpy(),
                     df.Receiver_account.to_numpy(),
                     df.Amount.to_numpy(dtype=float)))
    log(f"Building graph from {len(edges):,} transfers ...")

    t0 = time.time()
    res = R.detect_rings(edges)
    elapsed = time.time() - t0
    log(f"  {res['n_nodes']:,} nodes, {res['n_edges']:,} edges, "
        f"{res['n_communities_examined']:,} communities examined, "
        f"{res['n_rings_flagged']:,} rings flagged  [{elapsed:.1f}s]")

    ranked = res["rings"]
    base_rate = len(net_truth) / max(res["n_nodes"], 1)
    log(f"  base rate of network-laundering accounts: {base_rate:.4%}")

    curve = []
    for k in (10, 25, 50, 100, 200, 500, len(ranked)):
        if k > len(ranked) or k == 0:
            continue
        acc = {m for r in ranked[:k] for m in r["members"]}
        tp = len(acc & net_truth)
        prec = tp / max(len(acc), 1)
        curve.append({
            "top_k_rings": k,
            "accounts_surfaced": len(acc),
            "network_laundering_found": tp,
            "precision": round(prec, 4),
            "recall_of_network_accounts": round(tp / max(len(net_truth), 1), 4),
            "lift_over_base_rate": round(prec / base_rate, 1) if base_rate else None,
        })

    log("")
    log("  TOP-K   ACCOUNTS    FOUND   PRECISION    RECALL     LIFT")
    for c in curve:
        log(f"  {c['top_k_rings']:>5}   {c['accounts_surfaced']:>8,}   "
            f"{c['network_laundering_found']:>6,}   {c['precision']:>9.2%}   "
            f"{c['recall_of_network_accounts']:>7.2%}   {c['lift_over_base_rate']:>6}x")

    # Per typology: which shapes does structure actually recover?
    flagged = {m for r in ranked[:args.top_k] for m in r["members"]}
    by_typ = {}
    for t, accts in sorted(per_typ.items()):
        hit = len(accts & flagged)
        by_typ[t] = {
            "is_network_shape": t in NETWORK_TYPOLOGIES,
            "accounts": len(accts),
            "found_in_top_k": hit,
            "recall": round(hit / max(len(accts), 1), 4),
        }
    log("")
    log(f"  Per typology, within the top {args.top_k} candidate rings:")
    log(f"  {'TYPOLOGY':<24} {'KIND':<9} {'ACCTS':>7} {'FOUND':>6} {'RECALL':>8}")
    for t, v in sorted(by_typ.items(), key=lambda kv: (-kv[1]["is_network_shape"],
                                                       -kv[1]["recall"])):
        kind = "network" if v["is_network_shape"] else "single"
        log(f"  {t:<24} {kind:<9} {v['accounts']:>7,} {v['found_in_top_k']:>6,} "
            f"{v['recall']:>7.1%}")

    out = {
        "source": "SAML-D (Oztas et al., IEEE ICEBE 2023) - third-party ground truth",
        "months_used": args.months,
        "transactions": int(len(df)),
        "graph": {"nodes": res["n_nodes"], "edges": res["n_edges"],
                  "communities_examined": res["n_communities_examined"],
                  "rings_flagged": res["n_rings_flagged"],
                  "runtime_seconds": round(elapsed, 1)},
        "ground_truth": {
            "accounts_touched_by_laundering": len(all_truth),
            "accounts_in_network_typologies": len(net_truth),
            "base_rate_network": round(base_rate, 6),
        },
        "uses_labels": False,
        "curve": curve,
        "per_typology_top_k": {"k": args.top_k, "results": by_typ},
    }
    (OUT_DIR / "saml_ring_benchmark.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"\nWrote {OUT_DIR / 'saml_ring_benchmark.json'}")


if __name__ == "__main__":
    main()
