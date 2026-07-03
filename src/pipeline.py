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
import runpy
from pathlib import Path

import config as C
from utils import log

HERE = Path(__file__).resolve().parent

# (module file, human label)
STAGES = [
    ("01_clean.py", "Stage 1 — Cleaning & leak removal"),
    ("02_features.py", "Stage 2/3 — Feature engineering"),
    ("03_train.py", "Stage 4/5 — Ensemble + 5-fold CV"),
    ("04_graph.py", "Stage 6 — Graph label propagation"),
    ("05_score_explain.py", "Stage 7/8 — Risk score + SHAP"),
    ("plots.py", "Reporting — PR/ROC/calibration plots"),
]


def run_stage(fname: str, label: str) -> None:
    log("=" * 64)
    log(label)
    log("=" * 64)
    runpy.run_path(str(HERE / fname), run_name="__main__")


def print_summary() -> None:
    mpath = C.REPORTS_DIR / "03_metrics.json"
    if not mpath.exists():
        log("No metrics file found; did Stage 4/5 run?")
        return
    m = json.loads(mpath.read_text(encoding="utf-8"))
    e = m["ensemble_precision_first"]
    hr = m["ensemble_high_recall"]

    print("\n" + "=" * 64)
    print(" MULEGUARD AI — MEASURED RESULTS (5-fold stratified CV, out-of-fold)")
    print("=" * 64)
    print(f" Accounts: {m['n_accounts']:,}   Mules: {m['n_mules']}   "
          f"Prevalence: {m['prevalence_pct']}%")
    print(f" Engines: {m['engines']}   Features: {m['n_features']}")
    print("-" * 64)
    print(" PRECISION-FIRST OPERATING POINT (replace PDF bold figures with these)")
    print(f"   Precision : {e['precision']:.3f}")
    print(f"   Recall    : {e['recall']:.3f}")
    print(f"   F1        : {e['f1']:.3f}")
    print(f"   AUPRC     : {e['auprc']:.3f}")
    print(f"   AUROC     : {e['auroc']:.4f}")
    print(f"   FPR       : {e['fpr']:.4f}")
    print(f"   Threshold : {e['threshold']:.4f}")
    print("-" * 64)
    print(" HIGH-RECALL OPERATING POINT (analyst-review queue)")
    print(f"   Precision : {hr['precision']:.3f}   Recall: {hr['recall']:.3f}   "
          f"F1: {hr['f1']:.3f}")
    print("-" * 64)
    print(" PER-MODEL (AUPRC / AUROC):")
    for name, r in m["per_model"].items():
        print(f"   {name:<6} AUPRC={r['auprc']:.3f}  AUROC={r['auroc']:.4f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}")
    print("=" * 64)
    print(f" Full artefacts in: {C.REPORTS_DIR}")
    print("=" * 64 + "\n")


def main() -> None:
    for fname, label in STAGES:
        run_stage(fname, label)
    print_summary()


if __name__ == "__main__":
    main()
