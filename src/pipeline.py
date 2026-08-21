"""
MuleGuard AI — end-to-end orchestrator.

Runs every stage in order and prints a final summary table whose row names match
the submission PDF, so the bold placeholder numbers (pages 3 & 13) can be
replaced with these MEASURED 5-fold cross-validation results.

Run:  python src/pipeline.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import config as C
from utils import log

HERE = Path(__file__).resolve().parent

# (module file, human label)
STAGES = [
    ("06_integrity.py", "Stage 0 — Dataset integrity audit (run first, read first)"),
    ("01_clean.py", "Stage 1 — Cleaning & leak removal"),
    ("02_features.py", "Stage 2/3 — Feature engineering"),
    ("03_train.py", "Stage 4/5 — Ensemble + 5-fold CV"),
    ("04_graph.py", "Stage 6 — Graph label propagation"),
    ("05_score_explain.py", "Stage 7/8 — Risk score + SHAP"),
    # Stage 9 is the interpretable first line every bank runs before a model.
    # It is measured, not tuned: see the docstring in 09_rules.py.
    ("09_rules.py", "Stage 9 — AML rule layer, measured against the base rate"),
    # Stage 10 asks how much of our OWN score is the extract artefact. It reads
    # the feature matrix and refits, so it must run after Stage 2/3.
    ("08_feature_ablation.py", "Stage 10 — Feature ablation: which half is real"),
    # Stage 11 answers the questions an AML desk actually asks: Precision@K,
    # false alarms per 1,000 customers, and what a fixed review budget buys.
    ("10_operating_metrics.py", "Stage 11 — Operating metrics + extract drift"),
    # Stage 12 writes the vendor-neutral alert/case-pack bundle an EFRMS or AML
    # case manager would ingest. Documented schema, explicitly not certified.
    ("integration.py", "Stage 12 — EFRMS / AML export bundle"),
    ("plots.py", "Reporting — PR/ROC/calibration plots"),
]


def run_stage(fname: str, label: str) -> None:
    """Import the stage and call its main().

    This used to be runpy.run_path(..., run_name="__main__"), which re-executes
    the file as a throwaway module. That made the stages untestable (nothing
    could import them) and gave every class defined in them the module identity
    "__main__", the exact hazard ensemble.py exists to avoid. Importing them
    properly means a stage can be unit-tested, and the API can call the same
    entry point the CLI does.
    """
    log("=" * 64)
    log(label)
    log("=" * 64)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    mod = importlib.import_module(fname[:-3])
    mod.main()


def _integrity_verdict() -> dict | None:
    """The audit's own conclusion about THIS dataset, or None if it never ran."""
    path = C.REPORTS_DIR / "06_integrity_audit.json"
    if not path.exists():
        return None
    try:
        a = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {
        "contaminated": bool(a.get("verdict", {}).get("contaminated")),
        "partition_columns": [c.get("label") or c.get("column")
                              for c in a.get("partition_columns", {}).get("columns", [])],
    }


def print_summary() -> None:
    mpath = C.REPORTS_DIR / "03_metrics.json"
    if not mpath.exists():
        log("No metrics file found; did Stage 4/5 run?")
        return
    m = json.loads(mpath.read_text(encoding="utf-8"))
    e = m["ensemble_precision_first"]
    hr = m["ensemble_high_recall"]

    def ms(block: dict, key: str) -> str:
        if key not in block:
            return "n/a"
        return f"{block[key]['mean']:.3f} +/- {block[key]['std']:.3f}"

    print("\n" + "=" * 72)
    print(" MULEGUARD AI — MEASURED RESULTS")
    print("=" * 72)
    print(f" Validation : {m['validation']['scheme']}")
    print(f" Accounts   : {m['n_accounts']:,}   Mules: {m['n_mules']}   "
          f"Prevalence: {m['prevalence_pct']}%")
    print(f" Features   : {m['n_features_available']:,} available, "
          f"top {m['top_k_features_per_fold']} selected inside each fold")
    print(f" Engines    : {m['engines']}")
    print("-" * 72)
    print(" PRECISION-FIRST OPERATING POINT   (mean +/- std across folds)")
    for k in ("precision", "recall", "f1", "auprc", "auroc", "fpr"):
        print(f"   {k.upper():<10}: {ms(e, k)}")
    print(f"   {'LIFT':<10}: {ms(e, 'lift_over_prevalence')} x better than random")
    print(f"   Confusion (summed over folds): "
          f"TP={e['tp_total']} FP={e['fp_total']} FN={e['fn_total']}")
    print(f"   Precision target {m['precision_target']} met in "
          f"{m['precision_target_met_in_folds_pct']}% of folds")
    print("-" * 72)
    print(" HIGH-RECALL OPERATING POINT (analyst review queue)")
    print(f"   PRECISION : {ms(hr, 'precision')}")
    print(f"   RECALL    : {ms(hr, 'recall')}")
    print("-" * 72)
    print(" PER-MODEL (AUPRC):")
    for name, r in m["per_model"].items():
        print(f"   {name:<6} AUPRC={r['auprc']['mean']:.3f} +/- {r['auprc']['std']:.3f}"
              f"   AUROC={r['auroc']['mean']:.3f}")
    print("-" * 72)
    print(f" Random-guess AUPRC baseline: {m['auprc_baseline_random']:.4f}")
    print("=" * 72)

    # The warning is only true if THIS dataset is contaminated. Asserting it
    # unconditionally would be exactly the kind of unverified claim the
    # integrity audit exists to catch, so it is read from the audit's verdict.
    verdict = _integrity_verdict()
    if verdict is None:
        print(" No integrity audit found — run 06_integrity.py before quoting these.")
    elif verdict["contaminated"]:
        cols = verdict.get("partition_columns") or []
        where = f" ({', '.join(cols[:2])})" if cols else ""
        print(" READ reports/00_INTEGRITY.md BEFORE QUOTING ANY OF THESE NUMBERS.")
        print(f" This dataset's classes fall into disjoint groups{where}, which")
        print(" inflates every metric above. The integrity report quantifies it.")
    else:
        print(" Integrity audit: no assembly artefact detected. Uninformative views")
        print(" of this data score near the random baseline, so these metrics")
        print(" reflect behaviour. Still read reports/00_INTEGRITY.md.")
    print("=" * 72)
    print(f" Full artefacts in: {C.REPORTS_DIR}")
    print("=" * 72 + "\n")


# Stages that answer a research question about OUR dataset rather than scoring
# somebody else's. Skipped in demo mode, where the job is to produce a result on
# a file that arrived five minutes ago.
RESEARCH_ONLY = {"08_feature_ablation.py"}


def main() -> None:
    import time

    stages = STAGES
    if C.FAST_MODE:
        stages = [(f, l) for f, l in STAGES if f not in RESEARCH_ONLY]
        log("=" * 64)
        log("FAST MODE — built for a dataset handed over live.")
        log(f"  {len(STAGES) - len(stages)} research-only stage(s) skipped; "
            f"every leak defence and validation guarantee is unchanged.")
        log("=" * 64)

    t0 = time.time()
    timings = []
    for fname, label in stages:
        t = time.time()
        run_stage(fname, label)
        timings.append((label.split("—")[0].strip(), time.time() - t))

    log("=" * 64)
    log(f"TIMINGS (total {time.time() - t0:.0f}s)")
    for name, secs in sorted(timings, key=lambda x: -x[1]):
        log(f"  {name:<34} {secs:>6.1f}s")
    print_summary()


if __name__ == "__main__":
    main()
