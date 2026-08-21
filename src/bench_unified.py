"""
End-to-end validation of the whole system on SAML-D.

Every component of this project has been measured on its own. The system has
not. This runs the unified scorer — behavioural features from the raw ledger,
motif detection, structural ring detection and temporal localisation, blended by
a fitted model — against third-party ground truth, and reports what the complete
pipeline achieves rather than what its best part does.

SAML-D (Oztas et al., IEEE ICEBE 2023): 9.5M transactions, 855K accounts,
28 named typologies. We derive an account-level label as "this account was a
counterparty to at least one laundering transaction", which is the closest
account-level truth the dataset supports.

Honest framing, stated before the numbers
-----------------------------------------
The label is derived, not given. SAML-D labels transactions; a bank labels
accounts. Deriving one from the other is a modelling choice and a reader should
know it was made. It is also the same choice any account-level method has to
make on this dataset, so the comparison is fair even if the absolute number
would not transfer to a differently-labelled book.

Validation is a stratified split with the model fitted on train only. This is
not the nested CV used for the headline dataset; at 855K accounts a single
held-out split is adequate and honest, and it is labelled as such.

Run:  python src/bench_unified.py --months 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config as C
import score_unified as SU
from utils import log, save_json

OUT_DIR = C.REPORTS_DIR / "bench_unified"
USECOLS = ["Date", "Time", "Sender_account", "Receiver_account", "Amount",
           "Is_laundering", "Laundering_type"]


def load(path: pathlib.Path, months: int | None) -> pd.DataFrame:
    log(f"Loading {path} ...")
    t0 = time.time()
    d = pd.read_csv(path, usecols=USECOLS)
    log(f"  {len(d):,} rows in {time.time() - t0:.1f}s")
    d["timestamp"] = pd.to_datetime(d["Date"] + " " + d["Time"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    if months:
        cut = d["timestamp"].min() + pd.DateOffset(months=months)
        d = d[d["timestamp"] < cut]
        log(f"  first {months} month(s): {len(d):,} rows")
    return d.rename(columns={"Sender_account": "src", "Receiver_account": "dst",
                             "Amount": "amount"})


def account_labels(d: pd.DataFrame) -> pd.Series:
    laund = d[d.Is_laundering == 1]
    bad = set(laund.src) | set(laund.dst)
    accounts = pd.Index(sorted(set(d.src) | set(d.dst)))
    return pd.Series(accounts.isin(bad).astype(int), index=accounts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"D:/SAML-D.csv/SAML-D.csv")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--no-network", action="store_true",
                    help="behavioural features only, to isolate what the network adds")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load(pathlib.Path(args.csv), args.months)
    y_all = account_labels(d)
    log(f"Accounts {len(y_all):,}, laundering-touched {int(y_all.sum()):,} "
        f"({y_all.mean():.4%})")

    ledger = d[["src", "dst", "amount", "timestamp"]]

    log("Building features (behavioural + network) ...")
    t0 = time.time()
    feats, meta = SU.build_features(ledger, with_network=not args.no_network)
    log(f"  {feats.shape[0]:,} accounts x {feats.shape[1]} features "
        f"[{time.time() - t0:.0f}s]")
    log(f"  {meta}")

    y = y_all.reindex(feats.index).fillna(0).astype(int)
    idx_tr, idx_te = train_test_split(
        np.arange(len(feats)), test_size=0.3, random_state=C.RANDOM_STATE,
        stratify=y.to_numpy())

    log("Fitting the unified scorer on the training split only ...")
    t0 = time.time()
    scorer = SU.UnifiedScorer(seed=C.RANDOM_STATE).fit(
        feats.iloc[idx_tr], y.iloc[idx_tr].to_numpy())
    p_te = scorer.predict_proba(feats.iloc[idx_te])
    log(f"  fitted and scored in {time.time() - t0:.0f}s")

    y_te = y.iloc[idx_te].to_numpy()
    base = float(y_te.mean())
    auprc = float(average_precision_score(y_te, p_te))
    auroc = float(roc_auc_score(y_te, p_te))
    log("")
    log(f"  HELD-OUT: {len(y_te):,} accounts, {int(y_te.sum()):,} positive "
        f"(base rate {base:.4%})")
    log(f"  AUPRC {auprc:.4f}   ({auprc / base:.0f}x the base rate)")
    log(f"  AUROC {auroc:.4f}")

    order = np.argsort(-p_te)
    curve = []
    log("")
    log("     TOP-K   FOUND   PRECISION    RECALL      LIFT")
    for k in (50, 100, 250, 500, 1000, 2500, 5000):
        if k > len(order):
            break
        sel = order[:k]
        tp = int(y_te[sel].sum())
        prec = tp / k
        curve.append({"k": k, "found": tp, "precision": round(prec, 4),
                      "recall": round(tp / max(y_te.sum(), 1), 4),
                      "lift": round(prec / base, 1)})
        log(f"  {k:>8,}   {tp:>5,}   {prec:>9.2%}   {tp / y_te.sum():>7.2%}   {prec / base:>7.1f}x")

    # What did the network signals actually contribute?
    contrib = {}
    if not args.no_network:
        net_cols = [c for c in feats.columns
                    if c.startswith(("motif_", "ring_", "role_"))]
        contrib = {"network_columns": len(net_cols)}
        log("")
        log(f"  network signals supplied {len(net_cols)} of {feats.shape[1]} columns")

    out = {
        "source": "SAML-D (Oztas et al., IEEE ICEBE 2023), third-party ground truth",
        "label_definition": ("account was a counterparty to at least one laundering "
                             "transaction; derived from transaction labels, which is a "
                             "modelling choice and is stated as one"),
        "validation": "single stratified 70/30 split, model fitted on train only",
        "months_used": args.months,
        "transactions": int(len(d)),
        "accounts": int(len(feats)),
        "features": int(feats.shape[1]),
        "network_used": not args.no_network,
        "held_out": {"accounts": int(len(y_te)), "positives": int(y_te.sum()),
                     "base_rate": round(base, 6)},
        "auprc": round(auprc, 4), "auprc_lift": round(auprc / base, 1),
        "auroc": round(auroc, 4),
        "precision_at_k": curve,
        "pipeline_meta": meta,
        **contrib,
    }
    name = "unified_saml_behavioural.json" if args.no_network else "unified_saml.json"
    save_json(out, OUT_DIR / name)
    log(f"\nWrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
