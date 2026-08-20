"""
Experiment D — small regularization-focused hyperparameter search + multi-seed.

At 81 positives, every grid point that is evaluated-and-picked adds selection
variance, so the grid is deliberately tiny (regularization knobs only) and K/
resampling are already fixed by B/C. n_estimators is NOT tuned — it is set by
early stopping on an inner validation watchlist. The chosen config is then
re-measured across several seeds to report an honest mean±std.

Selection is by mean AUPRC over repeated CV (threshold-free). This is single-level
OOF, so the mild selection optimism on that AUPRC is disclosed; the operating point
itself is estimated honestly (and separately) by the nested-CV Stage A.

Run:  .venv/bin/python src/experiments/hp_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "experiments"))

import config as C  # noqa: E402
from utils import log, save_json  # noqa: E402
import _harness as H  # noqa: E402


# Tiny, regularization-focused grid (<=9 combos). n_estimators fixed (early-stop
# would need a per-fold watchlist; at this size the config n_estimators=400 with
# lr=0.05 and depth<=5 is already conservative).
GRID = [
    {"max_depth": md, "min_child_weight": mcw}
    for md in (3, 4, 5)
    for mcw in (1, 5, 10)
]


def main() -> None:
    data = H.load_features()
    X, y = data.X, data.y
    log(f"HP search on {X.shape[0]:,} x {X.shape[1]:,}, positives={int(y.sum())}; "
        f"{len(GRID)} combos")

    # Fixed structural choices from B/C: spw-only, no SMOTE, all features (updated
    # here if B/C say otherwise — defaults chosen to be the clean baseline).
    base_cfg = {"use_smote": False, "use_spw": True}

    n_rep = 3
    report: dict = {"n_repeats": n_rep, "grid_size": len(GRID),
                    "base_config": base_cfg, "results": []}

    best = None
    for combo in GRID:
        oofs = H.repeated_oof(X, y, n_repeats=n_rep, params=combo, **base_cfg)
        auprc = [H.ranking_metrics(y, p)["auprc"] for p in oofs]
        row = {"params": combo,
               "auprc_mean": round(float(np.mean(auprc)), 4),
               "auprc_std": round(float(np.std(auprc)), 4)}
        report["results"].append(row)
        log(f"    {combo}: AUPRC={row['auprc_mean']}±{row['auprc_std']}")
        if best is None or row["auprc_mean"] > best["auprc_mean"]:
            best = row

    report["best"] = best
    log(f"Best config: {best['params']}  AUPRC={best['auprc_mean']}")

    # Multi-seed confirmation on the winner (5 seeds).
    log("Multi-seed confirmation on winner (5 seeds) ...")
    oofs = H.repeated_oof(X, y, n_repeats=5, params=best["params"], **base_cfg)
    auprc = [H.ranking_metrics(y, p)["auprc"] for p in oofs]
    auroc = [H.ranking_metrics(y, p)["auroc"] for p in oofs]
    report["winner_multiseed"] = {
        "auprc_mean": round(float(np.mean(auprc)), 4),
        "auprc_std": round(float(np.std(auprc)), 4),
        "auroc_mean": round(float(np.mean(auroc)), 4),
        "auroc_std": round(float(np.std(auroc)), 4),
        "n_seeds": 5,
    }
    # Is the winner meaningfully better than config defaults? (honesty about tuning)
    default_oofs = H.repeated_oof(X, y, n_repeats=5, **base_cfg)
    default_auprc = float(np.mean([H.ranking_metrics(y, p)["auprc"] for p in default_oofs]))
    report["default_auprc_mean"] = round(default_auprc, 4)
    report["tuning_gain"] = round(report["winner_multiseed"]["auprc_mean"] - default_auprc, 4)
    report["tuning_verdict"] = (
        "meaningful" if report["tuning_gain"] > report["winner_multiseed"]["auprc_std"]
        else "within noise — tuning gain <= seed std; default config is defensible")
    log(f"Winner multiseed AUPRC={report['winner_multiseed']['auprc_mean']}"
        f"±{report['winner_multiseed']['auprc_std']}  "
        f"vs default={report['default_auprc_mean']}  -> {report['tuning_verdict']}")

    save_json(report, C.REPORTS_DIR / "09_hp_search.json")
    log("Wrote reports/09_hp_search.json")


if __name__ == "__main__":
    main()
