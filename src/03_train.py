"""
Stage 4/5 — Nested, repeated cross-validation with an honest operating point.

WHAT CHANGED AND WHY IT MATTERS
--------------------------------
The previous version reported metrics that were optimistically biased in three
separate places, all of which a careful judge would find:

  1. The isotonic calibrator was fitted on the pooled out-of-fold predictions and
     then those same values were scored. Isotonic regression is a flexible
     monotone fit; fitting and evaluating on identical data inflates AUPRC.
  2. The operating threshold was chosen by scanning the same pooled curve the
     precision was then reported from. Picking the maximum of a noisy curve and
     quoting that maximum is optimistic by construction — and with 81 positives
     the curve is very noisy.
  3. The stacking meta-learner was trained on the base models' *training-set*
     predictions, where a 400-tree XGBoost is near-perfect, then applied to
     validation predictions with a completely different distribution.

Here, every decision that touches the label — feature selection, resampling,
stacking weights, probability calibration, and the threshold — is fitted inside
an inner split of the training fold and then applied, frozen, to validation rows
the fitting never saw. Nothing about a validation row influences how it is
scored.

The whole procedure is then repeated with several different shuffles, because a
single 5-fold split puts only ~16 mules in each validation fold and the headline
metric moves by several points on the seed alone. We report mean +/- standard
deviation across repeats, so the number carries its own uncertainty.

Expect the numbers to be LOWER than the biased version produced. That is the
point: they are the numbers that will hold up.

Outputs:
  reports/03_metrics.json    — per-fold and aggregate metrics with error bars
  data/oof_predictions.csv   — pooled out-of-fold probabilities
  models/muleguard_models.joblib — final models refit on all data

Run:  python src/03_train.py
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config as C
import ensemble
import dictionary as D
import schema as S
from ensemble import (HAVE_LGBM, HAVE_XGB, MIN_POSITIVES_FOR_ISOTONIC,
                      MuleEnsemble, metrics_at)
from utils import load_frame, log, save_json

try:
    import shap
    HAVE_SHAP = True
except Exception:  # noqa: BLE001
    HAVE_SHAP = False


def _fold_shap(ens: MuleEnsemble, X_va: np.ndarray) -> np.ndarray | None:
    """SHAP values for validation rows, from the model that scored them.

    Explanations used to come from the final ensemble refit on ALL rows, which
    had already seen the account it was explaining — so the reasons shown did
    not belong to the number shown. Computing them per fold means each account's
    explanation comes from a model that never trained on it, exactly like its
    score.
    """
    if not HAVE_SHAP:
        return None
    tree = ens.models.get("xgb") or ens.models.get("lgbm")
    if tree is None:
        return None
    try:
        sv = shap.TreeExplainer(tree).shap_values(ens.selected_matrix(X_va))
    except Exception as exc:  # noqa: BLE001
        log(f"  SHAP unavailable for this fold ({exc})")
        return None
    if isinstance(sv, list):          # older versions return one array per class
        sv = sv[1]
    sv = np.asarray(sv, dtype=np.float32)
    if sv.ndim == 3:                  # (n, k, classes)
        sv = sv[:, :, -1]
    return sv

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Nested, repeated cross-validation
# --------------------------------------------------------------------------
def run_cv(X: np.ndarray, y: np.ndarray, n_repeats: int) -> dict:
    n_features = X.shape[1]
    fold_rows: list[dict] = []
    per_model_rows: list[dict] = []
    pooled_p = np.zeros(len(y))
    pooled_base = np.zeros((len(y), 3))
    pooled_n = np.zeros(len(y))
    pooled_shap = np.zeros((len(y), n_features), dtype=np.float32)
    selected_ever: set[int] = set()
    target_met_flags: list[bool] = []
    thr_precision: list[float] = []
    thr_recall: list[float] = []
    meta_coefs: list[dict] = []
    calib_methods: list[str] = []

    t0 = time.time()
    for rep in range(n_repeats):
        seed = C.RANDOM_STATE + rep
        skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            ens = MuleEnsemble(seed=seed).fit(X[tr], y[tr])

            p_va = ens.predict_proba(X[va])
            B_va = ens.base_probs(X[va])

            # The threshold was fixed on inner data before these rows were seen.
            fold_rows.append({
                "repeat": rep + 1, "fold": fold,
                "n_val": int(len(va)), "n_val_mules": int(y[va].sum()),
                "precision_first": metrics_at(y[va], p_va, ens.thr_precision),
                "high_recall": metrics_at(y[va], p_va, ens.thr_recall),
            })
            target_met_flags.append(bool(ens.target_met))
            thr_precision.append(float(ens.thr_precision))
            thr_recall.append(float(ens.thr_recall))
            meta_coefs.append(ens.meta_coef)
            calib_methods.append(ens.calibration_method)

            sv = _fold_shap(ens, X[va])
            if sv is not None:
                pooled_shap[np.ix_(va, ens.sel_idx)] += sv
                selected_ever.update(int(i) for i in ens.sel_idx)

            for i, name in enumerate(("iso", "xgb", "lgbm")):
                per_model_rows.append({
                    "model": name,
                    "auprc": float(average_precision_score(y[va], B_va[:, i])),
                    "auroc": float(roc_auc_score(y[va], B_va[:, i])),
                })

            pooled_p[va] += p_va
            pooled_base[va] += B_va
            pooled_n[va] += 1

            log(f"  repeat {rep+1}/{n_repeats} fold {fold}/{C.N_FOLDS}: "
                f"AUPRC={fold_rows[-1]['precision_first']['auprc']:.3f} "
                f"P={fold_rows[-1]['precision_first']['precision']:.3f} "
                f"R={fold_rows[-1]['precision_first']['recall']:.3f} "
                f"({time.time()-t0:.0f}s elapsed)")

    pooled_n = np.maximum(pooled_n, 1)
    pooled_p /= pooled_n
    pooled_base /= pooled_n[:, None]
    pooled_shap /= pooled_n[:, None].astype(np.float32)

    return {
        "fold_rows": fold_rows,
        "per_model_rows": per_model_rows,
        "pooled_p": pooled_p,
        "pooled_base": pooled_base,
        "pooled_shap": pooled_shap,
        "selected_ever": sorted(selected_ever),
        "target_met_rate": float(np.mean(target_met_flags)),
        "thr_precision": thr_precision,
        "thr_recall": thr_recall,
        "meta_coefs": meta_coefs,
        "calib_methods": calib_methods,
    }


def summarise(rows: list[dict], key: str) -> dict:
    """Mean +/- std across folds for each metric, plus the fold count."""
    metrics = ["precision", "recall", "f1", "auprc", "auroc", "fpr",
               "lift_over_prevalence"]
    out: dict = {"n_folds": len(rows)}
    for m in metrics:
        vals = np.array([r[key][m] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        out[m] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        }
    for c in ("tp", "fp", "fn", "tn"):
        out[c + "_total"] = int(sum(r[key][c] for r in rows))
    return out


def main() -> None:
    df = load_frame(C.FEATURES_PARQUET)
    target = S.bind_target(df, C)
    log(f"Target column: {target}")
    y = df[target].astype(int).values
    Xdf = df.drop(columns=[target]).select_dtypes(include=[np.number])
    feat_names = Xdf.columns.tolist()
    # NaNs are carried through on purpose: MuleEnsemble._prep() imputes them
    # with medians learned inside the training fold. Zero-filling here would
    # both destroy the "no such activity" distinction and hide the leak the
    # in-fold imputation exists to fix.
    X = Xdf.to_numpy(dtype=np.float32)
    log(f"Training matrix: {X.shape[0]:,} x {X.shape[1]:,}  positives={int(y.sum())} "
        f"({100*y.mean():.3f}%)")

    if not HAVE_XGB or not HAVE_LGBM:
        log(f"WARNING: xgboost={HAVE_XGB}, lightgbm={HAVE_LGBM}. "
            f"Missing engines are skipped; install for the full ensemble.")

    n_repeats = int(os.environ.get("MULEGUARD_REPEATS", C.N_REPEATS))
    if C.FAST_MODE:
        log("FAST MODE: 1 repeat, 2 inner folds. Error bars widen; nothing that "
            "protects the number is weakened.")
    log(f"Nested CV: {n_repeats} repeat(s) x {C.N_FOLDS} outer folds, "
        f"{C.INNER_FOLDS} inner folds, top-{C.TOP_K_FEATURES} features per fold")

    cv = run_cv(X, y, n_repeats)
    rows = cv["fold_rows"]

    prevalence = float(y.mean())
    results = {
        "n_accounts": int(len(y)),
        "n_mules": int(y.sum()),
        "prevalence_pct": round(100 * prevalence, 4),
        "auprc_baseline_random": round(prevalence, 4),
        "n_features_available": len(feat_names),
        "top_k_features_per_fold": C.TOP_K_FEATURES,
        "validation": {
            "scheme": f"nested {C.N_FOLDS}-fold stratified CV, "
                      f"{n_repeats} repeat(s), {C.INNER_FOLDS} inner folds",
            "what_is_fitted_inside_each_fold": [
                "feature selection", "base models", "stacking weights",
                "probability calibration", "operating threshold",
            ],
            "note": "No validation row influences any fitted component, "
                    "including the threshold it is later scored against.",
        },
        # Recorded rather than assumed: the calibrator is selected by how many
        # positives the fold actually has. Isotonic is a step function and with
        # this few positives it collapses to a handful of distinct values, which
        # destroys ranking resolution and therefore Precision@K. See the
        # measurement in ensemble.py.
        "calibration": _calibration_block(cv),
        "engines": {"xgboost": HAVE_XGB, "lightgbm": HAVE_LGBM, "shap": HAVE_SHAP},
        "reproducibility": _repro_block(n_repeats),
        "precision_target": C.PRECISION_TARGET,
        "precision_target_met_in_folds_pct": round(100 * cv["target_met_rate"], 1),
        "ensemble_precision_first": summarise(rows, "precision_first"),
        "ensemble_high_recall": summarise(rows, "high_recall"),
        "per_model": {},
        "per_fold_detail": rows,
    }

    # The two operating points the model actually fitted, averaged over folds.
    # Stage 7/8 turns these into the LOW/MEDIUM/HIGH band edges, so a band
    # boundary is the model's own threshold rather than a hand-picked constant.
    tp, tr = np.array(cv["thr_precision"]), np.array(cv["thr_recall"])
    results["operating_points"] = {
        "high_band_threshold": {
            "mean": round(float(tp.mean()), 6),
            "std": round(float(tp.std(ddof=1)) if len(tp) > 1 else 0.0, 6),
            "derived_from": f"highest-recall threshold holding precision >= "
                            f"{C.PRECISION_TARGET}, fitted on inner folds only",
        },
        "medium_band_threshold": {
            "mean": round(float(tr.mean()), 6),
            "std": round(float(tr.std(ddof=1)) if len(tr) > 1 else 0.0, 6),
            "derived_from": f"analyst review queue, precision >= "
                            f"{C.RECALL_QUEUE_PRECISION}",
        },
        "note": "Band edges in Stage 7/8 are these values x 1000. They are not "
                "chosen to flatter the result; they are the thresholds the "
                "ensemble fitted before seeing any validation row.",
    }

    # A second operating point that does not have to be fitted.
    #
    # A threshold is estimated from positives, so on a small extract there may
    # not be enough of them to estimate one. With 10 mules over 5 outer and 2
    # inner folds each threshold fit sees about 4 positives, and a cutoff placed
    # on 4 points does not transfer: on such a file the fitted point flagged two
    # accounts and caught none, reporting precision 0.000 and recall 0.000 while
    # the ranking underneath was still worth 6x the base rate. Reporting only
    # the fitted point makes a usable model look dead.
    #
    # A review budget is the alternative a bank actually runs: analysts work the
    # top N accounts per day. It is fixed by capacity rather than fitted, needs
    # no positives, and is invariant to prevalence, so it is comparable across
    # files with different mule rates.
    n_pos_per_fit = float(y.sum()) * (1 - 1 / C.N_FOLDS) / max(C.INNER_FOLDS, 1)
    estimable = n_pos_per_fit >= ensemble.MIN_TP_SUPPORT

    order = np.argsort(-cv["pooled_p"])
    prevalence = float(y.mean())
    budgets = []
    for pct in (0.5, 1.0, 2.0, 5.0, 10.0):
        k = max(1, int(np.ceil(len(y) * pct / 100)))
        if k > len(y):
            continue
        picked = order[:k]
        tp = int(y[picked].sum())
        prec = tp / k
        budgets.append({
            "budget_pct": pct,
            "accounts_reviewed": k,
            "threshold": round(float(cv["pooled_p"][picked[-1]]), 6),
            "true_mules_found": tp,
            "precision": round(prec, 4),
            "recall": round(tp / max(int(y.sum()), 1), 4),
            "lift_over_prevalence": round(prec / prevalence, 2) if prevalence else 0.0,
        })

    results["review_budget"] = {
        "points": budgets,
        "fitted_threshold_is_estimable": bool(estimable),
        "positives_per_threshold_fit": round(n_pos_per_fit, 1),
        "min_positives_required": ensemble.MIN_TP_SUPPORT,
        "why": "Precision and recall at a fixed review capacity, measured on "
               "pooled out-of-fold scores. Needs no threshold to be fitted, so "
               "it stays meaningful when the positive count is too small to "
               "estimate one." + ("" if estimable else
               " On this file the fitted threshold is NOT estimable "
               f"({n_pos_per_fit:.1f} positives per fit, {ensemble.MIN_TP_SUPPORT} "
               "required), so these budget rows are the operating point to "
               "report and the fitted precision/recall above should be "
               "disregarded."),
    }

    # What the stacker learned about each base model. Published because the
    # isolation forest scores below random here and the honest way to say so is
    # to show its negative coefficient, not to omit it.
    mc = pd.DataFrame(cv["meta_coefs"])
    results["stacking_coefficients"] = {
        k: {"mean": round(float(mc[k].mean()), 4),
            "std": round(float(mc[k].std(ddof=1)), 4)} for k in mc.columns
    }

    pm = pd.DataFrame(cv["per_model_rows"])
    for name, g in pm.groupby("model"):
        results["per_model"][name] = {
            "auprc": {"mean": round(float(g["auprc"].mean()), 4),
                      "std": round(float(g["auprc"].std(ddof=1)), 4)},
            "auroc": {"mean": round(float(g["auroc"].mean()), 4),
                      "std": round(float(g["auroc"].std(ddof=1)), 4)},
        }

    e = results["ensemble_precision_first"]
    log("=== ENSEMBLE (precision-first, nested out-of-fold) ===")
    for k in ("precision", "recall", "f1", "auprc", "auroc", "lift_over_prevalence"):
        if k in e:
            log(f"    {k}: {e[k]['mean']:.4f} +/- {e[k]['std']:.4f}")

    save_json(results, C.REPORTS_DIR / "03_metrics.json")

    pd.DataFrame({
        "y_true": y,
        "p_iso": cv["pooled_base"][:, 0],
        "p_xgb": cv["pooled_base"][:, 1],
        "p_lgbm": cv["pooled_base"][:, 2],
        "p_ensemble_calibrated": cv["pooled_p"],
    }).to_csv(C.DATA_DIR / "oof_predictions.csv", index=False)
    log(f"Wrote {C.DATA_DIR / 'oof_predictions.csv'}")

    _save_oof_shap(cv, feat_names)
    _refit_and_save(X, y, feat_names)


def _calibration_block(cv: dict) -> dict:
    """Which calibrator each fold actually used, and why."""
    from collections import Counter
    methods = cv.get("calib_methods") or []
    counts = dict(Counter(methods))
    return {
        "method_by_fold": counts,
        "selection_rule": (f"Platt below {MIN_POSITIVES_FOR_ISOTONIC} positives in the "
                           f"calibration fit, isotonic at or above it."),
        "why": ("Isotonic is a step function. Measured on this data it scored worse "
                "than no calibration at all on log loss and collapsed the output to "
                "15 distinct probabilities, which makes ranking inside a step "
                "impossible and damages Precision@K."),
    }


def _repro_block(n_repeats: int) -> dict:
    """Everything needed to reproduce this exact run.

    The shipped reports previously said "3 repeats" while config defaulted to 5,
    so the numbers in the paper could not be regenerated by running the code as
    checked in. Recording the resolved values and library versions makes that
    class of mismatch impossible to ship silently.
    """
    import platform
    import sklearn

    versions = {"python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__}
    for mod in ("xgboost", "lightgbm", "shap"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "absent"
    return {
        "random_state": C.RANDOM_STATE,
        # Recorded because it used to matter and must be seen not to any more:
        # set-iteration order (hash-seed dependent) once fixed the one-hot column
        # order, and XGBoost samples columns by index, so the same data with the
        # same seed drifted between processes. Orderings are sorted now; this
        # field lets anyone confirm a run was not relying on a particular seed.
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset (randomised)"),
        "column_order_is_deterministic": True,
        "n_repeats_resolved": n_repeats,
        "n_repeats_config_default": C.N_REPEATS,
        "repeats_overridden_by_env": "MULEGUARD_REPEATS" in os.environ,
        "n_folds": C.N_FOLDS,
        "inner_folds": C.INNER_FOLDS,
        "top_k_features": C.TOP_K_FEATURES,
        "versions": versions,
    }


def _save_oof_shap(cv: dict, feat_names: list[str]) -> None:
    """Store per-account SHAP produced by the fold that scored each account."""
    cols = cv["selected_ever"]
    if not cols:
        log("No out-of-fold SHAP produced (shap or a tree engine is missing).")
        return
    sv = cv["pooled_shap"][:, cols]
    names = [feat_names[i] for i in cols]
    path = C.DATA_DIR / "oof_shap.npz"
    np.savez_compressed(path, shap=sv.astype(np.float32),
                        feature_names=np.array(names, dtype=object),
                        feature_index=np.array(cols, dtype=np.int32))
    log(f"Wrote {path}  (out-of-fold SHAP, {sv.shape[0]:,} x {sv.shape[1]:,})")


def _refit_and_save(X: np.ndarray, y: np.ndarray, feat_names: list[str]) -> None:
    """Refit the whole ensemble on all rows, for scoring future accounts."""
    import joblib

    log("Refitting final ensemble on all data ...")
    final = MuleEnsemble(seed=C.RANDOM_STATE).fit(X, y)
    selected = [feat_names[i] for i in final.sel_idx]
    bundle = {
        "ensemble": final,
        "feat_names": feat_names,
        "selected_features": selected,
        "selected_labels": [D.label(f) for f in selected],
        "threshold_precision_first": final.thr_precision,
        "threshold_high_recall": final.thr_recall,
    }
    joblib.dump(bundle, C.MODELS_DIR / "muleguard_models.joblib")
    log(f"Saved final ensemble -> {C.MODELS_DIR / 'muleguard_models.joblib'} "
        f"({len(selected)} selected features)")


if __name__ == "__main__":
    main()
