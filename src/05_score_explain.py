"""
Stage 7/8 — Risk score (0–1000) + SHAP explanations.

  * Map the calibrated ensemble probability (optionally blended with graph
    proximity, if Stage 6 produced it) onto a 0–1000 risk score.
  * Assign LOW / MEDIUM / HIGH bands.
  * Use SHAP on the tree models to produce a ranked, plain-English reason list
    per account, and emit a few sample Account Risk Reports like the PDF.

Run:  python src/05_score_explain.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from utils import load_frame, log, save_json


def band(score: float) -> str:
    if score <= C.BAND_LOW_MAX:
        return "LOW"
    if score <= C.BAND_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def to_score(prob: np.ndarray, graph: np.ndarray | None) -> np.ndarray:
    """Blend calibrated probability with graph proximity into 0–1000."""
    if graph is not None:
        # 85% model, 15% network corroboration (graph informs, model decides).
        blended = 0.85 * prob + 0.15 * graph
    else:
        blended = prob
    return np.clip(blended, 0, 1) * C.SCORE_MAX


def main() -> None:
    oof = pd.read_csv(C.DATA_DIR / "oof_predictions.csv")
    prob = oof["p_ensemble_calibrated"].values
    y = oof["y_true"].values

    graph = None
    if C.GRAPH_SCORES_CSV.exists():
        g = pd.read_csv(C.GRAPH_SCORES_CSV).sort_values("account_idx")
        if len(g) == len(prob):
            graph = g["graph_proximity"].values
            log("Blending graph proximity into risk score.")

    scores = to_score(prob, graph)
    bands = np.array([band(s) for s in scores])

    dist = {b: int((bands == b).sum()) for b in ("LOW", "MEDIUM", "HIGH")}
    log(f"Band distribution: {dist}")
    # Of the HIGH band, how many are truly mules? (band precision)
    high = bands == "HIGH"
    high_precision = float(y[high].mean()) if high.any() else 0.0
    log(f"HIGH-band precision: {round(high_precision, 4)} "
        f"({int(y[high].sum())}/{int(high.sum())})")

    # --- SHAP explanations on the trained tree model ---
    reasons = _shap_reasons()

    save_json({
        "band_distribution": dist,
        "high_band_precision": round(high_precision, 4),
        "high_band_count": int(high.sum()),
        "high_band_true_mules": int(y[high].sum()),
        "score_bands": {"LOW": [0, C.BAND_LOW_MAX],
                        "MEDIUM": [C.BAND_LOW_MAX, C.BAND_MEDIUM_MAX],
                        "HIGH": [C.BAND_MEDIUM_MAX, C.SCORE_MAX]},
        "sample_reasons_available": reasons is not None,
    }, C.REPORTS_DIR / "05_scoring_report.json")

    # Attach scores back to accounts for downstream use.
    pd.DataFrame({
        "account_idx": np.arange(len(scores)),
        "risk_score": scores.round(0).astype(int),
        "band": bands,
        "y_true": y,
    }).to_csv(C.DATA_DIR / "risk_scores.csv", index=False)
    log(f"Wrote {C.DATA_DIR / 'risk_scores.csv'}")


def _shap_reasons():
    """Compute SHAP values on the saved tree model; write top-feature report."""
    try:
        import joblib
        import shap
    except Exception:  # noqa: BLE001
        log("SHAP unavailable — skipping explanation report.")
        return None

    model_path = C.MODELS_DIR / "muleguard_models.joblib"
    if not model_path.exists():
        log("No saved model found — run Stage 4/5 first for SHAP.")
        return None

    bundle = joblib.load(model_path)
    tree = bundle.get("xgb") or bundle.get("lgbm")
    if tree is None:
        log("No tree model available for SHAP.")
        return None

    df = load_frame(C.FEATURES_PARQUET)
    X = df.drop(columns=[C.TARGET_COL]).select_dtypes(include=[np.number])
    feat_names = bundle["feat_names"]
    X = X[feat_names]

    explainer = shap.TreeExplainer(tree)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # some versions return per-class list
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    top = (pd.Series(mean_abs, index=feat_names)
           .sort_values(ascending=False).head(15).round(5).to_dict())

    save_json({"top_features_by_mean_abs_shap": top},
              C.REPORTS_DIR / "05_shap_top_features.json")
    log("Wrote SHAP top-feature report -> reports/05_shap_top_features.json")
    return top


if __name__ == "__main__":
    main()
