"""
Stage 7/8 — Risk score (0-1000), bands, SHAP reasons, and Account Risk Reports.

The data dictionary turns explanations from "F2506 contributed 0.31" into
"Aadhaar Payment Bridge credit transactions rose sharply in the last week versus
the last month". That is the difference between a SHAP dump and something an
investigator can act on, and it is what an RBI/PMLA audit trail actually
requires: a reason a human can read, tied to a named banking variable.

TWO CORRECTIONS OVER THE PREVIOUS VERSION
-----------------------------------------
1. BAND EDGES ARE NOW THE MODEL'S OWN OPERATING POINTS. `config.py` claimed the
   HIGH cutoff was "re-derived from the precision-recall curve at training
   time", but the code used the constants 400 and 750 and never read the
   thresholds the ensemble had fitted and saved. Now Stage 4/5 writes those two
   thresholds to reports/03_metrics.json and this stage reads them back, so
   "HIGH" means "at or above the threshold that held precision >= 0.90 on inner
   folds" rather than "score above 750, because 750 is a round number".

2. EXPLANATIONS COME FROM THE MODEL THAT PRODUCED THE SCORE. SHAP used to be
   computed from the final ensemble refit on ALL rows, which had already trained
   on the very account it was explaining, while the score came from pooled
   out-of-fold predictions. The reasons therefore did not belong to the number.
   Stage 4/5 now stores per-account out-of-fold SHAP in data/oof_shap.npz and
   this stage reads that.

Outputs:
  reports/05_scoring_report.json      band distribution + band precision
  reports/05_shap_top_features.json   global ranking, with real names
  reports/05_account_reports.json     per-account cards for the riskiest accounts
  data/risk_scores.csv                score + band for every account

Run:  python src/05_score_explain.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
import dictionary as D
from utils import load_frame, log, save_json

ACTION = {
    "LOW": "No action — routine monitoring",
    "MEDIUM": "Enhanced monitoring + step-up authentication (OTP) on transfers",
    "HIGH": "Freeze outward transfers, escalate to AML desk, prepare STR",
}


# --------------------------------------------------------------------------
# Band edges — derived, not decreed
# --------------------------------------------------------------------------
def resolve_band_edges() -> dict:
    """Read the operating points Stage 4/5 fitted, and turn them into 0-1000 edges.

    Falls back to the config constants only when the metrics file is absent, and
    says which of the two happened so a reader is never guessing.
    """
    fallback = {
        "low_max": float(C.BAND_LOW_MAX),
        "medium_max": float(C.BAND_MEDIUM_MAX),
        "source": "config fallback — reports/03_metrics.json not found, so "
                  "these are hand-set constants, not fitted thresholds",
        "derived": False,
    }
    mpath = C.REPORTS_DIR / "03_metrics.json"
    if not C.DERIVE_BANDS_FROM_THRESHOLDS or not mpath.exists():
        return fallback
    try:
        ops = json.loads(mpath.read_text(encoding="utf-8"))["operating_points"]
        high = float(ops["high_band_threshold"]["mean"]) * C.SCORE_MAX
        med = float(ops["medium_band_threshold"]["mean"]) * C.SCORE_MAX
    except (KeyError, ValueError, TypeError):
        return fallback

    # The review-queue threshold must sit below the precision-first one; if a
    # degenerate fold inverts them, ordering is restored rather than silently
    # producing an empty MEDIUM band.
    low_max, medium_max = sorted((med, high))

    # Both thresholds can collapse onto the same value. That happens when the
    # calibrated probabilities saturate: isotonic regression maps almost every
    # account to exactly 0 or exactly 1, so there is no score at which precision
    # trades against recall and both operating points land in the same place.
    #
    # It is a real property of the run, not a bug to paper over, and it is worth
    # saying out loud: a dataset that separates this cleanly is usually a dataset
    # with something wrong in it. So the bands are reported as-is and flagged,
    # rather than being nudged apart to produce a MEDIUM band that the model
    # never actually justified.
    degenerate = abs(medium_max - low_max) < 1e-6
    if degenerate:
        return {
            "low_max": round(low_max, 2),
            "medium_max": round(medium_max, 2),
            "source": "degenerate — both fitted operating points landed on the "
                      "same threshold, so there is no MEDIUM band on this "
                      "dataset. The calibrated probabilities are saturated: "
                      "accounts are scored either clearly in or clearly out, "
                      "with almost nothing in between. Treat a result this "
                      "clean as a reason to re-read the integrity audit.",
            "derived": True,
            "degenerate": True,
        }

    return {
        "low_max": round(low_max, 2),
        "medium_max": round(medium_max, 2),
        "source": "derived from the ensemble's fitted operating points "
                  "(mean over folds, chosen on inner data only)",
        "derived": True,
        "degenerate": False,
    }


def make_bander(edges: dict):
    def band(score: float) -> str:
        if score < edges["low_max"]:
            return "LOW"
        if score < edges["medium_max"]:
            return "MEDIUM"
        return "HIGH"
    return band


def to_score(prob: np.ndarray, graph: np.ndarray | None) -> np.ndarray:
    """Map calibrated probability (optionally blended with graph proximity) to 0-1000."""
    blended = 0.85 * prob + 0.15 * graph if graph is not None else prob
    return np.clip(blended, 0, 1) * C.SCORE_MAX


# --------------------------------------------------------------------------
# Out-of-fold SHAP
# --------------------------------------------------------------------------
def load_oof_shap() -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    path = C.DATA_DIR / "oof_shap.npz"
    if not path.exists():
        log("No out-of-fold SHAP found — run Stage 4/5 with shap installed.")
        return None, None
    with np.load(path, allow_pickle=True) as z:
        sv = z["shap"]
        names = [str(n) for n in z["feature_names"]]
    log(f"Loaded out-of-fold SHAP: {sv.shape[0]:,} accounts x {sv.shape[1]:,} features")
    return sv, names


def global_report(sv: np.ndarray, names: list[str]) -> dict:
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:25]
    return {
        "provenance": "SHAP computed per fold, on validation rows only — each "
                      "account is explained by a model that never trained on it",
        "top_features_by_mean_abs_shap": [
            {
                "feature": names[i],
                "variable": D.real_name(names[i]),
                "meaning": D.explain(names[i]),
                "mean_abs_shap": round(float(mean_abs[i]), 6),
            }
            for i in order
        ],
    }


def reasons_for(sv_row: np.ndarray, names: list[str], k: int = 6) -> list[dict]:
    order = np.argsort(np.abs(sv_row))[::-1][:k]
    out = []
    for j in order:
        if sv_row[j] == 0:
            continue
        code = names[j]
        out.append({
            "variable": D.real_name(code),
            "feature": code,
            "meaning": D.explain(code),
            "effect": "raises risk" if sv_row[j] > 0 else "lowers risk",
            "shap": round(float(sv_row[j]), 6),
        })
    return out


def account_cards(sv, names, scores, bands, y, n: int = 25) -> list[dict]:
    cards = []
    for idx in np.argsort(scores)[::-1][:n]:
        cards.append({
            "account_idx": int(idx),
            "risk_score": int(round(scores[idx])),
            "band": str(bands[idx]),
            "recommended_action": ACTION[str(bands[idx])],
            "confirmed_mule": bool(y[idx]),
            "top_reasons": reasons_for(sv[idx], names) if sv is not None else [],
        })
    return cards


# --------------------------------------------------------------------------
def main() -> None:
    oof = pd.read_csv(C.DATA_DIR / "oof_predictions.csv")
    prob = oof["p_ensemble_calibrated"].to_numpy()
    y = oof["y_true"].to_numpy()

    graph = None
    if C.GRAPH_SCORES_CSV.exists():
        g = pd.read_csv(C.GRAPH_SCORES_CSV).sort_values("account_idx")
        if len(g) == len(prob):
            graph = g["graph_proximity"].to_numpy()
            log("Blending graph proximity into risk score.")

    edges = resolve_band_edges()
    log(f"Band edges: LOW < {edges['low_max']:.1f} <= MEDIUM < "
        f"{edges['medium_max']:.1f} <= HIGH   ({edges['source']})")

    scores = to_score(prob, graph)
    bands = np.array([make_bander(edges)(s) for s in scores])

    dist = {b: int((bands == b).sum()) for b in ("LOW", "MEDIUM", "HIGH")}
    log(f"Band distribution: {dist}")

    band_stats = {}
    for b in ("LOW", "MEDIUM", "HIGH"):
        m = bands == b
        band_stats[b] = {
            "accounts": int(m.sum()),
            "true_mules": int(y[m].sum()),
            "precision": round(float(y[m].mean()), 4) if m.any() else 0.0,
            "recall_of_all_mules": round(float(y[m].sum() / max(y.sum(), 1)), 4),
            "action": ACTION[b],
        }
    log(f"HIGH-band: {band_stats['HIGH']['true_mules']}/"
        f"{band_stats['HIGH']['accounts']} are mules "
        f"(precision {band_stats['HIGH']['precision']:.3f})")

    report = {
        "band_distribution": dist,
        "band_stats": band_stats,
        "score_bands": {"LOW": [0, edges["low_max"]],
                        "MEDIUM": [edges["low_max"], edges["medium_max"]],
                        "HIGH": [edges["medium_max"], C.SCORE_MAX]},
        "band_edge_provenance": edges,
        "note": "Scores come from pooled nested out-of-fold probabilities, so "
                "each account is scored by models that never trained on it.",
    }

    sv, names = load_oof_shap()
    if sv is not None and len(sv) == len(scores):
        save_json(global_report(sv, names),
                  C.REPORTS_DIR / "05_shap_top_features.json")
        save_json({"accounts": account_cards(sv, names, scores, bands, y)},
                  C.REPORTS_DIR / "05_account_reports.json")
        report["explanations_available"] = True
        report["explanation_provenance"] = "out-of-fold"
        log("Wrote SHAP global ranking and per-account risk reports")
    else:
        report["explanations_available"] = False

    save_json(report, C.REPORTS_DIR / "05_scoring_report.json")

    pd.DataFrame({
        "account_idx": np.arange(len(scores)),
        "risk_score": scores.round(0).astype(int),
        "calibrated_probability": prob.round(6),
        "band": bands,
        "recommended_action": [ACTION[b] for b in bands],
        "y_true": y,
    }).to_csv(C.DATA_DIR / "risk_scores.csv", index=False)
    log(f"Wrote {C.DATA_DIR / 'risk_scores.csv'}")


if __name__ == "__main__":
    main()
