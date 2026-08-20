"""
Stage 8 — Feature ablation: how much do the 29 typology features actually carry?

This exists to answer one question honestly, because the answer decides what we
are allowed to claim about a dataset we have never seen:

    If we are handed a bank extract whose columns are named differently enough
    that we CANNOT build the mule-typology features, how much detection power do
    we lose?

Three conditions, identical harness, identical seed, identical folds:

    FULL        everything (what we normally report)
    RAW ONLY    every mg_* feature removed, simulating a dataset where the
                behavioural columns could not be resolved
    TYPOLOGY    only the mg_* features, to see what they carry on their own

The gap between FULL and RAW ONLY is the honest value of the domain work, and
the RAW ONLY score is the floor we can promise on an unfamiliar schema.

Run:  python src/08_feature_ablation.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config as C
import schema as S
from ensemble import MuleEnsemble, metrics_at
from utils import load_frame, log, save_json

warnings.filterwarnings("ignore")

N_FOLDS = 5
SEED = C.RANDOM_STATE


def evaluate(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """One pass of stratified CV with the production ensemble."""
    if X.shape[1] == 0:
        return {"condition": label, "n_features": 0, "note": "no features available"}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aups, aurocs, precs, recs = [], [], [], []
    t0 = time.time()

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        ens = MuleEnsemble(seed=SEED).fit(X[tr], y[tr])
        p = ens.predict_proba(X[va])
        mt = metrics_at(y[va], p, ens.thr_precision)
        aups.append(average_precision_score(y[va], p))
        aurocs.append(roc_auc_score(y[va], p))
        precs.append(mt["precision"])
        recs.append(mt["recall"])
        log(f"  [{label}] fold {fold}/{N_FOLDS} AUPRC={aups[-1]:.3f} "
            f"({time.time()-t0:.0f}s)")

    def ms(v):
        a = np.asarray(v, dtype=float)
        return {"mean": round(float(a.mean()), 4),
                "std": round(float(a.std(ddof=1)), 4)}

    return {
        "condition": label,
        "n_features": int(X.shape[1]),
        "auprc": ms(aups), "auroc": ms(aurocs),
        "precision": ms(precs), "recall": ms(recs),
    }


def main() -> None:
    df = load_frame(C.FEATURES_PARQUET)
    target = S.bind_target(df, C)
    y = df[target].astype(int).values
    Xdf = df.drop(columns=[target]).select_dtypes(include=[np.number])

    cols = list(Xdf.columns)
    typology = [c for c in cols if str(c).startswith("mg_")]
    # The one-hot encodings of categorical fields are also prefixed mg_, but they
    # are not behavioural features; they belong to the raw side of the split.
    behavioural = [c for c in typology
                   if not any(str(c).startswith(p) for p in
                              ("mg_product_name", "mg_area_category", "mg_gender",
                               "mg_segmentation", "mg_cust_occp", "mg_row_"))]
    raw = [c for c in cols if c not in behavioural]

    log(f"Total features {len(cols):,} · behavioural typology {len(behavioural)} · "
        f"everything else {len(raw):,}")
    log(f"Positives {int(y.sum())} of {len(y):,}")

    conditions = [
        ("FULL", cols),
        ("RAW ONLY (typology removed)", raw),
        ("TYPOLOGY ONLY", behavioural),
    ]

    results = []
    for label, use in conditions:
        log(f"--- {label}: {len(use):,} features ---")
        X = Xdf[use].to_numpy(dtype=np.float32)
        results.append(evaluate(X, y, label))

    full = next(r for r in results if r["condition"] == "FULL")
    rawr = next(r for r in results if r["condition"].startswith("RAW"))
    delta = round(full["auprc"]["mean"] - rawr["auprc"]["mean"], 4)

    out = {
        "question": "How much detection power depends on the mule-typology "
                    "features, and therefore how much is lost on a dataset where "
                    "they cannot be built?",
        "scheme": f"{N_FOLDS}-fold stratified CV, seed {SEED}, identical folds "
                  f"across conditions, production ensemble",
        "behavioural_features": behavioural,
        "results": results,
        "auprc_attributable_to_typology": delta,
        "interpretation": (
            f"Removing the {len(behavioural)} behavioural features changes AUPRC by "
            f"{delta:+.4f}. That figure is the honest ceiling on what the domain "
            f"engineering contributes here, and the RAW ONLY row is the floor we "
            f"can promise on a schema where those columns cannot be resolved."
        ),
    }
    save_json(out, C.REPORTS_DIR / "08_feature_ablation.json")

    log("=== FEATURE ABLATION ===")
    for r in results:
        if "auprc" in r:
            log(f"  {r['condition']:<30} n={r['n_features']:>5}  "
                f"AUPRC={r['auprc']['mean']:.4f} +/- {r['auprc']['std']:.4f}  "
                f"P={r['precision']['mean']:.3f} R={r['recall']['mean']:.3f}")
    log(f"  AUPRC attributable to the typology features: {delta:+.4f}")


if __name__ == "__main__":
    main()
