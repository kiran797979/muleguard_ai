"""
Assemble a single measured-results summary from every experiment + metrics JSON.

Reads only files already written by the pipeline/experiments (never recomputes),
so it is safe to run any time and always reflects the latest measured numbers. It
writes reports/HONEST_SUMMARY.json and prints a human-readable digest — the one
artefact that answers "what are the honest numbers?" end to end.

Run:  .venv/bin/python src/experiments/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import config as C  # noqa: E402
from utils import log, save_json  # noqa: E402


def _load(name):
    p = C.REPORTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    metrics = _load("03_metrics.json")
    leak = _load("06_leak_audit.json")
    resamp = _load("07_resampling_ablation.json")
    feats = _load("08_feature_selection.json")
    hp = _load("09_hp_search.json")

    summary = {
        "headline": None,
        "operating_point": None,
        "optimistic_ceiling": None,
        "leak_audit": None,
        "resampling_winner": None,
        "feature_selection": None,
        "hp_tuning": None,
    }

    if metrics:
        summary["headline"] = {
            "auprc": metrics["headline_ranking"]["auprc"],
            "auprc_ci": metrics["headline_ranking_ci"]["auprc"],
            "auroc": metrics["headline_ranking"]["auroc"],
            "auroc_ci": metrics["headline_ranking_ci"]["auroc"],
            "brier_oos": metrics["headline_ranking"].get("brier_oos"),
            "ece_oos": metrics["headline_ranking"].get("ece_oos"),
        }
        summary["operating_point"] = {
            "precision": metrics["honest_operating_point"]["precision"],
            "recall": metrics["honest_operating_point"]["recall"],
            "f1": metrics["honest_operating_point"]["f1"],
            "recall_ci": metrics["honest_operating_point_recall_ci"],
            "threshold_spread": metrics["honest_operating_point"]["threshold_spread"],
            "mean_tp": metrics["honest_operating_point"]["mean_tp"],
            "mean_fp": metrics["honest_operating_point"]["mean_fp"],
        }
        summary["optimistic_ceiling"] = {
            "precision": metrics["optimistic_ceiling"]["precision"],
            "recall": metrics["optimistic_ceiling"]["recall"],
        }
    if leak:
        summary["leak_audit"] = {
            "auprc_with_calendar": leak["with_calendar"]["auprc"],
            "auprc_without_calendar": leak["without_calendar"]["auprc"],
            "auprc_drop_when_removed": leak["auprc_drop_when_removed"],
            "calendar_verdict": leak["calendar_verdict"],
            "top5_only_auroc": leak["top5_only"]["auroc"],
            "concentration_verdict": leak["concentration_verdict"],
            "temporal_split_feasible": leak["temporal_split"]["feasible"],
        }
    if resamp:
        summary["resampling_winner"] = {
            "winner": resamp["winner"],
            "arms": {k: {"auprc": v["auprc"]["mean"], "ece": v["ece"]["mean"]}
                     for k, v in resamp["arms"].items()},
        }
    if feats:
        summary["feature_selection"] = {"recommended_k": feats["recommended_k"],
                                        "sweep": feats["sweep"]}
    if hp:
        summary["hp_tuning"] = {"best": hp["best"], "tuning_verdict": hp["tuning_verdict"],
                                "tuning_gain": hp["tuning_gain"]}

    save_json(summary, C.REPORTS_DIR / "HONEST_SUMMARY.json")

    # Human digest.
    print("\n" + "=" * 70)
    print(" MULEGUARD — HONEST MEASURED SUMMARY")
    print("=" * 70)
    if summary["headline"]:
        h = summary["headline"]
        print(f" HEADLINE (leak-audited, out-of-fold, threshold-free):")
        print(f"   AUPRC {h['auprc']}  (95% CI {h['auprc_ci']['lo']}–{h['auprc_ci']['hi']})")
        print(f"   AUROC {h['auroc']}  (95% CI {h['auroc_ci']['lo']}–{h['auroc_ci']['hi']})")
    if summary["operating_point"]:
        o = summary["operating_point"]
        print(f" HONEST OPERATING POINT (nested CV, out-of-sample):")
        print(f"   P {o['precision']['mean']}±{o['precision']['std']}  "
              f"R {o['recall']['mean']}±{o['recall']['std']}  "
              f"F1 {o['f1']['mean']}±{o['f1']['std']}")
    if summary["optimistic_ceiling"]:
        c = summary["optimistic_ceiling"]
        print(f" (optimistic in-sample ceiling: P {c['precision']} R {c['recall']} — NOT headline)")
    if summary["leak_audit"]:
        la = summary["leak_audit"]
        print(f" LEAK AUDIT: AUPRC drop when calendar features removed = "
              f"{la['auprc_drop_when_removed']} -> {la['calendar_verdict'].split('—')[0].strip()}")
        print(f"   top-5-only AUROC {la['top5_only_auroc']} -> "
              f"{la['concentration_verdict'].split('—')[0].strip()}")
    if summary["resampling_winner"]:
        print(f" RESAMPLING WINNER: {summary['resampling_winner']['winner']}")
    if summary["feature_selection"]:
        print(f" FEATURE SELECTION: keep K = {summary['feature_selection']['recommended_k']}")
    if summary["hp_tuning"]:
        print(f" HP TUNING: {summary['hp_tuning']['tuning_verdict']}")
    print("=" * 70 + "\n")
    log("Wrote reports/HONEST_SUMMARY.json")


if __name__ == "__main__":
    main()
