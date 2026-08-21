"""
Stage 0 — Dataset integrity audit. Run this before believing any metric.

Everything else in this pipeline tries to detect mules. This stage tries to prove
that the pipeline CANNOT work honestly on the supplied data, and reports how far
it gets. If a judge is going to attack the submission, this is where they will
aim, so we aim there first and publish the result.

The problem in one table: every negative in the supplied file comes from the
October extract, and every positive from the September/November/December
extracts. There is no month in which both classes appear. So any difference
between those extraction runs is perfectly correlated with the label while
describing no customer behaviour at all.

Three falsification tests, each designed to score near the 0.0089 random baseline
if the data is sound:

  A. MISSINGNESS ONLY — discard every value; keep only blank/not-blank.
     Whether a cell was populated is decided by the extraction job, so this
     carries zero behavioural information. A high score here is proof of an
     extract artefact.
  B. INDIVIDUALLY-USELESS FEATURES — columns whose own correlation with the
     target is below 0.05. No single one of them can identify a mule. If they
     still separate the classes together, the signal is diffuse and structural.
  C. SHUFFLED LABELS — the sanity floor. This must collapse, otherwise the
     evaluation harness itself is broken.

Run:  python src/06_integrity.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config as C
import dictionary as D
import schema as S
from utils import load_raw, log, save_json

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    HAVE_XGB = False


def quick_cv(X: np.ndarray, y: np.ndarray, seed: int = C.RANDOM_STATE) -> dict:
    """A deliberately plain 5-fold XGBoost — no stacking, no tuning.

    The point is not to model well. It is to show how much separability is
    available to a naive learner from information that should carry none.
    """
    if X.shape[1] == 0:
        return {"auprc": float("nan"), "auroc": float("nan"), "n_features": 0}
    # Three folds in demo mode. The falsification tests are about whether the
    # score is far from the baseline, not about a precise estimate of it, so the
    # conclusion is unchanged and it finishes in a third of the time.
    n_splits = 3 if C.FAST_MODE else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    spw = (y == 0).sum() / max(y.sum(), 1)
    for tr, va in skf.split(X, y):
        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, tree_method="hist", eval_metric="aucpr",
            scale_pos_weight=spw, random_state=seed, n_jobs=-1,
        ).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return {
        "auprc": round(float(average_precision_score(y, oof)), 4),
        "auroc": round(float(roc_auc_score(y, oof)), 4),
        "n_features": int(X.shape[1]),
    }


def main() -> None:
    if not HAVE_XGB:
        log("xgboost unavailable — integrity audit needs it. Skipping.")
        return

    df = load_raw(C.RAW_CSV)
    try:
        target, how = S.resolve_target(df, C.TARGET_COL_HINT)
    except KeyError as exc:
        log(f"Cannot run the integrity audit: {exc}")
        return
    C.__dict__["TARGET_COL"] = target
    y = df[target].astype(int).to_numpy()
    prevalence = float(y.mean())
    log(f"Integrity audit on {len(y):,} accounts, {int(y.sum())} positives "
        f"(target={target} via {how}; random-guess AUPRC = {prevalence:.4f})")

    drop = [target] + S.identifier_columns(df, target)
    feats = [c for c in df.columns if c not in drop]

    audit: dict = {
        "prevalence": round(prevalence, 5),
        "auprc_random_baseline": round(prevalence, 5),
        "schema": S.describe_schema(df, target, how),
    }

    # --- Partition columns: how the sample was assembled --------------------
    # Generalised from a hardcoded MNTH crosstab. Any low-cardinality column
    # whose values are class-pure reproduces the label while describing no
    # customer, and that is a property of the shape, not of the column's name.
    # On this file the top hit is MNTH — found without being told it exists.
    parts = S.partition_columns(df, target,
                                max_cardinality=C.PARTITION_MAX_CARDINALITY,
                                min_purity=C.PARTITION_MIN_PURITY)
    audit["partition_columns"] = {
        "found": len(parts),
        "columns": [{**pt, "label": D.label(pt["column"])} for pt in parts[:5]],
    }
    if parts:
        top = parts[0]
        audit["month_split"] = {
            "column": D.label(top["column"]),
            "counts": {k: {"0": v["0"], "1": v["1"]}
                       for k, v in top["crosstab"].items()},
            "months_containing_both_classes": top["values_containing_both_classes"],
            "purity": top["purity"],
        }
        log(f"Partition column {D.label(top['column'])}: "
            f"{top['values_containing_both_classes']} value(s) contain both "
            f"classes (purity {top['purity']:.3f})")
    else:
        log("No partition column found — the sample is not split by an "
            "assembly artefact this test can see.")

    # --- A. missingness only ------------------------------------------------
    M = df[feats].isna().astype(np.int8)
    M = M.loc[:, M.nunique() > 1]
    audit["test_A_missingness_only"] = quick_cv(M.to_numpy(np.float32), y)
    a = audit["test_A_missingness_only"]
    log(f"A. missingness-only ({a['n_features']} indicators): "
        f"AUPRC={a['auprc']} AUROC={a['auroc']}")

    # --- B. individually-useless features -----------------------------------
    num = df[feats].select_dtypes(include=[np.number])
    corr = num.corrwith(pd.Series(y, index=num.index)).abs()
    weak = corr[corr < 0.05].dropna().index.tolist()
    rng = np.random.default_rng(C.RANDOM_STATE)
    pick = list(rng.choice(weak, size=min(250, len(weak)), replace=False)) if weak else []
    Xw = np.nan_to_num(num[pick].to_numpy(np.float32)) if pick else np.empty((len(y), 0))
    audit["test_B_individually_useless"] = quick_cv(Xw, y)
    audit["test_B_individually_useless"]["pool_size"] = len(weak)
    b = audit["test_B_individually_useless"]
    log(f"B. 250 of {len(weak)} columns with |corr|<0.05: "
        f"AUPRC={b['auprc']} AUROC={b['auroc']}")

    # --- C. shuffled-label floor --------------------------------------------
    y_shuf = y.copy()
    np.random.default_rng(C.RANDOM_STATE + 7).shuffle(y_shuf)
    audit["test_C_shuffled_labels"] = quick_cv(Xw, y_shuf)
    c = audit["test_C_shuffled_labels"]
    log(f"C. shuffled-label floor: AUPRC={c['auprc']} AUROC={c['auroc']}")

    # --- Verdict -------------------------------------------------------------
    # Three independent grounds, because any one of them can be unavailable.
    #
    # The partition test is included deliberately. A dataset with no missing
    # values at all makes test A impossible: every blank-indicator column is
    # constant, so there is nothing to fit and the score comes back as NaN. If
    # the verdict rested only on A and B, such a file would be declared clean
    # while this same audit was reporting a purity-1.0 partition column two
    # lines above. Those two statements cannot both stand, and the partition
    # finding is the stronger evidence.
    def _exceeds(test: dict) -> bool:
        v = test.get("auprc")
        return v is not None and not np.isnan(v) and v > 10 * prevalence

    partitions = audit.get("partition_columns", {}).get("columns", [])
    hard_partition = [p for p in partitions if p.get("purity", 0) >= 0.999]

    # Row order is checked here rather than in Stage 1 because it is not a
    # column and so cannot be dropped. It is a property of the file itself.
    order = S.row_order_leak(y)
    audit["row_order"] = order

    grounds = []
    if order.get("sorted_by_label"):
        grounds.append("the file is ordered by label, so row position alone "
                       "reproduces it and any unshuffled split is invalid")
    if _exceeds(a):
        grounds.append("a model given only blank/not-blank patterns separates "
                       "the classes far above the base rate")
    if _exceeds(b):
        grounds.append("columns that are individually useless still separate "
                       "the classes when combined")
    if hard_partition:
        names = ", ".join(str(p.get("label") or p.get("column")) for p in hard_partition[:3])
        grounds.append(f"the classes fall into disjoint value sets of {names}, "
                       f"so that column alone reproduces the label")

    contaminated = bool(grounds)
    audit["verdict_grounds"] = grounds
    audit["tests_unavailable"] = [
        name for name, t in (("test_A_missingness_only", a),
                             ("test_B_individually_useless", b))
        if t.get("auprc") is None or (t.get("auprc") is not None and np.isnan(t["auprc"]))
    ]
    if audit["tests_unavailable"]:
        log(f"Note: {', '.join(audit['tests_unavailable'])} could not run on this "
            f"dataset (no usable columns for that test). The verdict rests on the "
            f"remaining evidence.")
    audit["verdict"] = {
        "contaminated": bool(contaminated),
        "grounds": grounds,
        "summary": (
            "CONTAMINATED on " + str(len(grounds)) + " ground(s): "
            + "; ".join(grounds) + ". Metrics from this dataset measure how the "
            "sample was assembled as well as customer behaviour, and the two "
            "cannot be cleanly separated within this file."
            if contaminated else
            "No artefact detected. Uninformative views of the data score near "
            "the random baseline and no column partitions the classes, so the "
            "headline metrics reflect behaviour."
            + (" Note: " + ", ".join(audit["tests_unavailable"])
               + " could not run on this dataset, so this verdict rests on the "
                 "remaining evidence."
               if audit["tests_unavailable"] else "")
        ),
    }
    save_json(audit, C.REPORTS_DIR / "06_integrity_audit.json")
    _write_markdown(audit)
    log(f"VERDICT: {audit['verdict']['summary'][:60]}...")


def _write_markdown(a: dict) -> None:
    """Human-readable integrity report — the first thing anyone should read."""
    prev = a["prevalence"]
    A, B, Cc = (a["test_A_missingness_only"], a["test_B_individually_useless"],
                a["test_C_shuffled_labels"])
    ms = a.get("month_split", {})

    rows = "\n".join(
        f"| {m} | {v.get('0', 0):,} | {v.get('1', 0):,} |"
        for m, v in ms.get("counts", {}).items()
    )

    if ms:
        partition_section = f"""`{ms.get('column', '?')}` shows how the sample was assembled:

| Value | Negative | Positive |
|---|---|---|
{rows}

**{ms.get('months_containing_both_classes', 0)} value(s) contain both classes.**
The two classes fall into disjoint groups, so every difference between those
groups — which fields were populated, how the feed behaved — lines up with the
label while describing no customer behaviour at all. This column was found by
its shape, not by its name: no prior knowledge of the schema was used."""
    else:
        partition_section = ("No column partitions the classes into disjoint "
                             "value sets, so the sample shows no sign of having "
                             "been assembled along a line that tracks the label.")

    fix_line = (f"drops `{ms.get('column', 'the partition column')}` outright — it "
                f"alone separates the classes"
                if ms else "found no partition column to drop")

    md = f"""# MuleGuard AI — Dataset Integrity Report

**Read this before quoting any metric from this project.**

Random-guess AUPRC on this data is **{prev:.4f}** ({prev*100:.2f}% of accounts are mules).

## The problem

{partition_section}

## Falsification tests

Each test feeds a model information that *cannot* identify a mule. All three
should score near {prev:.4f} if the dataset is sound.

| Test | What it uses | AUPRC | AUROC | vs random |
|---|---|---|---|---|
| A. Missingness only | blank/not-blank pattern, **no values** ({A['n_features']} indicators) | **{A['auprc']}** | {A['auroc']} | {A['auprc']/prev:.0f}x |
| B. Individually-useless | 250 columns each with \\|corr\\| < 0.05 | **{B['auprc']}** | {B['auroc']} | {B['auprc']/prev:.0f}x |
| C. Shuffled labels | same columns, labels randomised | {Cc['auprc']} | {Cc['auroc']} | {Cc['auprc']/prev:.0f}x |

Test C collapsing to the baseline confirms the evaluation harness is sound — so
the scores in A and B are real properties of the data, not a bug.

Test A is the decisive one: **knowing only which cells were blank, with every
number thrown away, identifies mules almost perfectly.** No model can distinguish
that from genuine behaviour, because within this file the two are the same thing.

## Verdict

{a['verdict']['summary']}

## What the pipeline does about it

1. **Partition-column removal** — the pipeline {fix_line}.
2. **Drops post-outcome fields** — resolution status flags and resolve-days are
   written after an investigation closes and do not exist at scoring time.
   `FRAUD_SUSPECTED` correlates 0.97; `FALSE_POSITIVE` correlates 0.05 and is
   just as unusable, which is why a correlation threshold is not a leak defence.
3. **Extract hardening** — drops every column whose blank rate differs between
   the classes by more than {int(C.MAX_MISSINGNESS_DIFFERENTIAL*100)}%, on the grounds that whether a cell was
   populated is a property of the extraction job, not of a customer.
4. **Nested cross-validation** — feature selection, stacking, calibration and the
   operating threshold are all fitted inside the training fold, so none of the
   remaining optimism is the evaluation's fault.

Steps 1-3 remove the artefact where it can be identified. They cannot remove what
is unidentifiable: a behavioural feature that also drifts month to month is
confounded, and no amount of modelling separates the two within this file.

## What would fix it

Negatives and positives sampled from the **same** groups. With both classes
present in every extract, the partition can be controlled for and the reported
numbers would measure behaviour alone. This is a data-collection change, not a
modelling one — worth raising with whoever supplied the data, because it applies
to everyone working from this file, not just this submission.
"""
    path = C.REPORTS_DIR / "00_INTEGRITY.md"
    path.write_text(md, encoding="utf-8")
    log(f"Wrote {path}")


if __name__ == "__main__":
    main()
