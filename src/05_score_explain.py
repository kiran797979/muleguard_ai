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
    shap_info = _shap_reasons()

    # --- Per-account Account Risk Report cards (closes the page-14 report gap) ---
    _write_risk_report_cards(scores, bands, y, shap_info)

    save_json({
        "band_distribution": dist,
        "high_band_precision": round(high_precision, 4),
        "high_band_count": int(high.sum()),
        "high_band_true_mules": int(y[high].sum()),
        "score_bands": {"LOW": [0, C.BAND_LOW_MAX],
                        "MEDIUM": [C.BAND_LOW_MAX, C.BAND_MEDIUM_MAX],
                        "HIGH": [C.BAND_MEDIUM_MAX, C.SCORE_MAX]},
        "sample_reasons_available": shap_info is not None,
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
    """Compute SHAP values on the saved tree model.

    Writes both the global top-feature report AND per-account Account Risk Report
    cards for the highest-risk accounts (closing the PDF's page-14 report gap).
    Returns the per-row SHAP matrix + feature names for card rendering, or None.
    """
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
    return {"sv": sv, "feat_names": feat_names, "X": X}


# --------------------------------------------------------------------------
# Account Risk Report cards (per-account, human-readable)
# --------------------------------------------------------------------------
def _humanize(feat: str) -> str:
    """Turn an anonymised/engineered feature name into a readable phrase.

    Only maps what the data actually revealed (one-hot categoricals with known
    semantics + our own engineered mg_* features). Anonymised F#### columns are
    left as-is — we do not invent meaning for them (honesty over fabrication).
    """
    engineered = {
        "mg_row_mean": "overall activity level (row mean)",
        "mg_row_std": "activity volatility (row std-dev)",
        "mg_row_max": "peak single-feature value",
        "mg_row_min": "lowest single-feature value",
        "mg_row_nonzero_frac": "breadth of activity (non-zero fraction)",
        "mg_row_skew": "activity skew",
        "mg_row_range": "activity range (max − min)",
    }
    if feat in engineered:
        return engineered[feat]
    # One-hot categoricals: "F3891_student" -> "occupation = student", etc.
    prefixes = {
        "F3886": "account type", "F3889": "recency bucket", "F3890": "account status",
        "F3891": "occupation", "F3892": "gender", "F3893": "customer segment",
    }
    if "_" in feat:
        base, _, val = feat.partition("_")
        if base in prefixes:
            return f"{prefixes[base]} = {val}"
    if feat == "F3888":
        return "account vintage (days since opening)"
    return feat  # anonymised — reported honestly by its code name


def _write_risk_report_cards(scores, bands, y, shap_info) -> None:
    """Emit formatted per-account risk reports for the highest-risk accounts."""
    order = np.argsort(scores)[::-1][:C.N_RISK_REPORT_CARDS]

    cards = []
    lines = []
    lines.append("=" * 60)
    lines.append(" MULEGUARD AI — ACCOUNT RISK REPORTS")
    lines.append(f" Top {len(order)} highest-risk accounts (of {len(scores):,})")
    lines.append(" NOTE: per-account SHAP drivers below are IN-SAMPLE illustrations")
    lines.append(" (full-data model applied to training rows). The headline metrics")
    lines.append(" in reports/03_metrics.json are the honest out-of-sample numbers.")
    lines.append("=" * 60)

    sv = feat_names = X = None
    if shap_info is not None:
        sv, feat_names, X = shap_info["sv"], shap_info["feat_names"], shap_info["X"]

    for rank, idx in enumerate(order, 1):
        idx = int(idx)
        band = bands[idx]
        reasons = []
        if sv is not None:
            row = sv[idx]
            top_pos = np.argsort(row)[::-1][:C.N_REASONS_PER_CARD]  # push risk UP
            for j in top_pos:
                if row[j] <= 0:
                    continue
                fname = feat_names[j]
                reasons.append({
                    "feature": fname,
                    "reason": _humanize(fname),
                    "value": round(float(X.iloc[idx, j]), 4),
                    "shap": round(float(row[j]), 4),
                })

        card = {
            "rank": rank,
            "account_idx": idx,
            "risk_score": int(round(scores[idx])),
            "band": band,
            "recommended_action": C.BAND_ACTIONS.get(band, ""),
            "confirmed_mule": bool(y[idx] == 1),
            "top_reasons": reasons,
        }
        cards.append(card)

        # Human-readable text block
        lines.append("")
        lines.append(f"#{rank}  Account {idx:<6}   Risk {card['risk_score']:>4}/1000   "
                     f"Band: {band}")
        lines.append(f"     Action: {card['recommended_action']}")
        if reasons:
            lines.append("     Top risk drivers:")
            for k, r in enumerate(reasons, 1):
                lines.append(f"       {k}. {r['reason']}  "
                             f"(value={r['value']}, contribution=+{r['shap']})")
        lines.append(f"     [ground-truth label: {'MULE' if card['confirmed_mule'] else 'normal'}]")

    save_json({"n_cards": len(cards), "cards": cards},
              C.REPORTS_DIR / "05_account_risk_reports.json")
    (C.REPORTS_DIR / "account_risk_reports.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {len(cards)} Account Risk Report cards -> "
        f"reports/05_account_risk_reports.json (+ .txt)")


if __name__ == "__main__":
    main()
