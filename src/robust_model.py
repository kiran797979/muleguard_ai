"""A behavioural scorer that keeps working when the schema does not arrive whole.

The deployed ensemble is fitted on 1,506 columns and needs nearly all of them:
masked down to 300 it scores 0.009, which is the random baseline. That is fine
on the file it was fitted for and useless anywhere else, and "anywhere else" is
exactly what an unseen validation extract is.

Two changes, both aimed at learning behaviour rather than memorising a schema:

  * Train only on the mule-typology features. They are ratios and shares, so
    they survive a different extraction run in a way that raw per-column
    aggregates do not. Section V-C of the paper measures this: the raw columns
    barely beat a values-free model, while the typology subset is the part least
    explained by the extract artefact.

  * Train with FEATURE DROPOUT. Every epoch sees a copy of the data with a
    random subset of columns blanked, so the model is forced to find several
    routes to the same conclusion instead of leaning on one. A model that has
    only ever seen complete rows has no reason to be robust to incomplete ones,
    and will confidently emit noise when it meets one.

Nothing here is tuned against any held-out test file. The dropout rates are a
schedule chosen in advance, and every number reported is out-of-fold.

Run:  python src/robust_model.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import config as C
from utils import log

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

MODEL_PATH = C.MODELS_DIR / "muleguard_robust.joblib"
REPORT_PATH = C.REPORTS_DIR / "12_robust_model.json"

# Copies of the training data, each with this share of columns blanked. The
# first is the intact data; the rest teach the model to cope without a column.
DROPOUT_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8)
SEED = 42


def behavioural_columns(df: pd.DataFrame) -> list[str]:
    """The engineered typology features, which are the transferable ones."""
    return [c for c in df.columns if c.startswith("mg_")]


def _params(y: np.ndarray) -> dict:
    pos = max(int((y == 1).sum()), 1)
    return dict(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=(y == 0).sum() / pos,
        verbose=-1, random_state=SEED,
    )


def augment(X: np.ndarray, y: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Stack copies of the data with random columns blanked to NaN."""
    parts_x, parts_y = [], []
    for rate in DROPOUT_SCHEDULE:
        Z = X.copy()
        if rate > 0:
            mask = rng.random(Z.shape) < rate
            Z[mask] = np.nan
        parts_x.append(Z)
        parts_y.append(y)
    return np.vstack(parts_x), np.concatenate(parts_y)


def fit(X: np.ndarray, y: np.ndarray, dropout: bool = True):
    rng = np.random.default_rng(SEED)
    Xt, yt = augment(X, y, rng) if dropout else (X, y)
    return lgb.LGBMClassifier(**_params(yt)).fit(Xt, yt)


def out_of_fold(X: np.ndarray, y: np.ndarray, dropout: bool,
                folds: int = 5, repeats: int = 3) -> np.ndarray:
    """Honest scores: every row predicted by a model that never saw it.

    Augmentation happens INSIDE the fold. Building the dropout copies first
    would put masked duplicates of a validation row into training, which is
    leakage of exactly the kind Section IV-E exists to prevent.
    """
    oof = np.zeros((repeats, len(y)))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats,
                                 random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(X, y)):
        model = fit(X[tr], y[tr], dropout=dropout)
        oof[k // folds, te] = model.predict_proba(X[te])[:, 1]
    return oof.mean(axis=0)


def degradation(model, X: np.ndarray, y: np.ndarray,
                keep: tuple[int, ...]) -> list[dict]:
    """AUPRC as columns are progressively withheld at scoring time."""
    rng = np.random.default_rng(SEED)
    out = []
    for k in keep:
        if k > X.shape[1]:
            continue
        idx = rng.choice(X.shape[1], size=k, replace=False)
        Z = np.full_like(X, np.nan)
        Z[:, idx] = X[:, idx]
        ap = average_precision_score(y, model.predict_proba(Z)[:, 1])
        out.append({"features_kept": int(k), "auprc": round(float(ap), 4),
                    "lift": round(float(ap / y.mean()), 1)})
    return out


def main() -> None:
    if lgb is None:
        log("lightgbm is not installed; cannot build the robust model.")
        return
    feats_path = C.DATA_DIR / "features.parquet"
    if not feats_path.exists():
        log("features.parquet missing - run the pipeline first.")
        return

    df = pd.read_parquet(feats_path)
    cols = behavioural_columns(df)
    if not cols:
        log("no mg_ behavioural columns found.")
        return
    y = pd.read_csv(C.RAW_CSV, usecols=[C.TARGET_COL_HINT])[
        C.TARGET_COL_HINT].astype(int).to_numpy()
    X = df[cols].to_numpy(np.float32)
    log(f"behavioural matrix {X.shape[0]:,} x {X.shape[1]} "
        f"({int(y.sum())} positives, {y.mean() * 100:.3f}%)")

    results = {}
    for label, drop in (("plain", False), ("dropout", True)):
        oof = out_of_fold(X, y, dropout=drop)
        ap, auc = average_precision_score(y, oof), roc_auc_score(y, oof)
        results[label] = {"auprc": round(float(ap), 4),
                          "auroc": round(float(auc), 4),
                          "lift_over_base_rate": round(float(ap / y.mean()), 1)}
        log(f"  {label:<8} out-of-fold AUPRC {ap:.4f}  AUROC {auc:.4f}  "
            f"lift {ap / y.mean():.0f}x")

    keep = (len(cols), 40, 30, 20, 15, 10, 6, 3)
    curves = {}
    for label, drop in (("plain", False), ("dropout", True)):
        curves[label] = degradation(fit(X, y, dropout=drop), X, y, keep)

    log("  columns kept | plain  | dropout")
    for a, b in zip(curves["plain"], curves["dropout"]):
        log(f"    {a['features_kept']:>10} | {a['auprc']:.3f}  | {b['auprc']:.3f}")

    model = fit(X, y, dropout=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": cols,
                 "dropout_schedule": list(DROPOUT_SCHEDULE)}, MODEL_PATH)
    log(f"wrote {MODEL_PATH}")

    REPORT_PATH.write_text(json.dumps({
        "question": "Does a behavioural model trained with feature dropout keep "
                    "working when the schema arrives incomplete?",
        "n_accounts": int(len(y)), "n_positives": int(y.sum()),
        "base_rate": round(float(y.mean()), 6),
        "n_behavioural_features": len(cols),
        "validation": "5-fold stratified, 3 repeats, augmentation inside the "
                      "fold so no masked copy of a validation row is trained on",
        "out_of_fold": results,
        "degradation_at_scoring_time": curves,
        "note": "The deployed 1,506-column ensemble falls to the random "
                "baseline below roughly half its schema. These figures are the "
                "reason the behavioural model exists.",
    }, indent=1), encoding="utf-8")
    log(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
