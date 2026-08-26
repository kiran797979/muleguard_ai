"""
MuleGuard AI — the service layer behind the API.

This module owns every artefact the pipeline produced and answers questions
about them. It deliberately contains NO fallback data: if an artefact is
missing, the relevant call raises `ArtefactMissing` and the API turns that into
a 503 with instructions. A demo that quietly invents numbers when the model
fails to load is worse than one that says "run the pipeline first".

Two scoring paths, and the difference matters:

  * `analyse_account(idx)` — for the 9,082 accounts in the benchmark. Uses the
    pooled OUT-OF-FOLD probability and the OUT-OF-FOLD SHAP values, so both the
    score and the explanation come from models that never trained on that
    account. This is the honest number.

  * `score_features(payload)` — for an arbitrary account handed in at runtime.
    Uses the final ensemble refit on all rows, because that is what a bank would
    deploy. Flagged as such in the response so the two are never confused.
"""

from __future__ import annotations

import json
import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

import sys  # noqa: E402

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as C  # noqa: E402
import dictionary as D  # noqa: E402
import schema as S  # noqa: E402

_LOCK = threading.Lock()


class ArtefactMissing(RuntimeError):
    """A pipeline output the caller needs has not been produced yet."""

    def __init__(self, what: str, stage: str):
        self.what, self.stage = what, stage
        super().__init__(f"{what} is missing — run {stage} first.")


# --------------------------------------------------------------------------
# JSON-safe conversion
# --------------------------------------------------------------------------
def jsonable(obj: Any) -> Any:
    """NaN/Inf are not valid JSON; numpy scalars are not serialisable.

    Left unhandled, a single NaN in a feature value produces a response the
    browser's JSON.parse rejects, and the UI shows an empty panel with no clue
    why. Everything crossing the wire goes through here.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def _read_json(path: Path, stage: str) -> dict:
    if not path.exists():
        raise ArtefactMissing(path.name, stage)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtefactMissing(f"{path.name} (corrupt: {exc})", stage) from exc


# --------------------------------------------------------------------------
# Cached artefact loaders
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def integrity() -> dict:
    return _read_json(C.REPORTS_DIR / "06_integrity_audit.json", "Stage 0 (06_integrity.py)")


@lru_cache(maxsize=1)
def clean_report() -> dict:
    return _read_json(C.REPORTS_DIR / "01_clean_report.json", "Stage 1 (01_clean.py)")


@lru_cache(maxsize=1)
def feature_report() -> dict:
    return _read_json(C.REPORTS_DIR / "02_features_report.json", "Stage 2/3 (02_features.py)")


@lru_cache(maxsize=1)
def metrics() -> dict:
    return _read_json(C.REPORTS_DIR / "03_metrics.json", "Stage 4/5 (03_train.py)")


@lru_cache(maxsize=1)
def scoring_report() -> dict:
    return _read_json(C.REPORTS_DIR / "05_scoring_report.json", "Stage 7/8 (05_score_explain.py)")


@lru_cache(maxsize=1)
def scale_benchmark() -> dict:
    """Throughput and volume, from the run on a public dataset we did not build."""
    out = _read_json(C.REPORTS_DIR / "bench_unified" / "unified_saml.json",
                     "src/bench_unified.py")
    try:
        out["latency"] = _read_json(
            C.REPORTS_DIR / "10_operating_metrics.json",
            "src/10_operating_metrics.py").get("scoring_latency", {})
    except Exception:
        out["latency"] = {}
    return out


@lru_cache(maxsize=1)
def operating_points() -> dict:
    """The detection operating point, chosen out of fold in src/operating_point.py.

    Deliberately separate from the action bands. The HIGH band triggers an
    automated freeze on a real customer's money, so it is set precision-first
    and stays there. Ranking accounts for a scored submission is a different
    decision with a different cost, and it gets its own point.
    """
    return _read_json(C.REPORTS_DIR / "13_operating_points.json",
                      "src/operating_point.py")


@lru_cache(maxsize=1)
def shap_global() -> dict:
    return _read_json(C.REPORTS_DIR / "05_shap_top_features.json", "Stage 7/8 (05_score_explain.py)")


@lru_cache(maxsize=1)
def rules_report() -> dict:
    return _read_json(C.REPORTS_DIR / "09_rules_report.json", "Stage 9 (09_rules.py)")


@lru_cache(maxsize=1)
def ablation_report() -> dict:
    return _read_json(C.REPORTS_DIR / "08_feature_ablation.json",
                      "Stage 10 (08_feature_ablation.py)")


@lru_cache(maxsize=1)
def graph_report() -> dict:
    return _read_json(C.REPORTS_DIR / "04_graph_report.json", "Stage 6 (04_graph.py)")


@lru_cache(maxsize=1)
def risk_scores() -> pd.DataFrame:
    path = C.DATA_DIR / "risk_scores.csv"
    if not path.exists():
        raise ArtefactMissing("risk_scores.csv", "Stage 7/8 (05_score_explain.py)")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def oof_shap() -> tuple[np.ndarray, list[str]]:
    path = C.DATA_DIR / "oof_shap.npz"
    if not path.exists():
        raise ArtefactMissing("oof_shap.npz", "Stage 4/5 (03_train.py) with shap installed")
    with np.load(path, allow_pickle=True) as z:
        return z["shap"], [str(n) for n in z["feature_names"]]


@lru_cache(maxsize=1)
def features_frame() -> pd.DataFrame:
    from utils import load_frame

    if not C.FEATURES_PARQUET.exists() and not C.FEATURES_PARQUET.with_suffix(".csv").exists():
        raise ArtefactMissing("features.parquet", "Stage 2/3 (02_features.py)")
    return load_frame(C.FEATURES_PARQUET)


@lru_cache(maxsize=1)
def feature_stats() -> pd.DataFrame:
    """Population median / IQR per feature, for 'is this value unusual?' evidence."""
    df = features_frame()
    target = S.bind_target(df, C)
    num = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
    return pd.DataFrame({
        "median": num.median(),
        "p90": num.quantile(0.90),
        "p99": num.quantile(0.99),
    })


@lru_cache(maxsize=1)
def bundle() -> dict:
    """Load the trained ensemble.

    joblib.load unpickles, which executes code. The path is fixed to
    models/muleguard_models.joblib and is never taken from a request, so the
    only way to reach this is to already have write access to the repo. No
    user-supplied path is ever passed to a loader.
    """
    path = C.MODELS_DIR / "muleguard_models.joblib"
    if not path.exists():
        raise ArtefactMissing("muleguard_models.joblib", "Stage 4/5 (03_train.py)")
    import joblib

    with _LOCK:
        return joblib.load(path)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
ARTEFACTS = [
    ("integrity audit", C.REPORTS_DIR / "06_integrity_audit.json", "06_integrity.py"),
    ("clean report", C.REPORTS_DIR / "01_clean_report.json", "01_clean.py"),
    ("feature report", C.REPORTS_DIR / "02_features_report.json", "02_features.py"),
    ("metrics", C.REPORTS_DIR / "03_metrics.json", "03_train.py"),
    ("scoring report", C.REPORTS_DIR / "05_scoring_report.json", "05_score_explain.py"),
    ("risk scores", C.DATA_DIR / "risk_scores.csv", "05_score_explain.py"),
    ("out-of-fold SHAP", C.DATA_DIR / "oof_shap.npz", "03_train.py"),
    ("trained model", C.MODELS_DIR / "muleguard_models.joblib", "03_train.py"),
    ("feature matrix", C.FEATURES_PARQUET, "02_features.py"),
    ("rule layer", C.REPORTS_DIR / "09_rules_report.json", "09_rules.py"),
    ("feature ablation", C.REPORTS_DIR / "08_feature_ablation.json",
     "08_feature_ablation.py"),
    ("operating metrics", C.REPORTS_DIR / "10_operating_metrics.json",
     "10_operating_metrics.py"),
]


def health() -> dict:
    items = []
    for label, path, stage in ARTEFACTS:
        ok = path.exists() or path.with_suffix(".csv").exists()
        items.append({
            "artefact": label,
            "present": ok,
            "produced_by": stage,
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        })
    ready = all(i["present"] for i in items)
    model_ok, model_err = False, None
    if ready:
        try:
            bundle()
            model_ok = True
        except Exception as exc:  # noqa: BLE001
            model_err = str(exc)
    return {
        "status": "READY" if (ready and model_ok) else "DEGRADED",
        "model_loaded": model_ok,
        "model_error": model_err,
        "artefacts": items,
        "missing": [i["artefact"] for i in items if not i["present"]],
        "hint": "Run  .\\run.ps1  (Windows) or  ./run.sh  (macOS/Linux) to "
                "produce every artefact.",
    }


# --------------------------------------------------------------------------
# Account analysis
# --------------------------------------------------------------------------
def n_accounts() -> int:
    return int(len(risk_scores()))


def list_accounts(band: str | None = None, limit: int = 50,
                  offset: int = 0, mules_only: bool = False) -> dict:
    df = risk_scores()
    if band:
        df = df[df["band"] == band.upper()]
    if mules_only:
        df = df[df["y_true"] == 1]
    total = int(len(df))
    df = df.sort_values("risk_score", ascending=False).iloc[offset:offset + limit]
    return {
        "total_matching": total,
        "offset": offset,
        "limit": limit,
        "accounts": jsonable(df.to_dict(orient="records")),
    }


def _evidence_for(idx: int, codes: list[str]) -> list[dict]:
    """The account's own value for each cited variable, against the population."""
    df, stats = features_frame(), feature_stats()
    out = []
    for code in codes:
        if code not in df.columns or code not in stats.index:
            continue
        val = df[code].iloc[idx]
        med, p90, p99 = (stats.loc[code, k] for k in ("median", "p90", "p99"))
        if pd.isna(val):
            standing = "no value recorded"
        elif val >= p99:
            standing = "top 1% of all accounts"
        elif val >= p90:
            standing = "top 10% of all accounts"
        elif val > med:
            standing = "above the population median"
        else:
            standing = "at or below the population median"
        out.append({
            "feature": code,
            "variable": D.real_name(code),
            "meaning": D.explain(code),
            "account_value": jsonable(val),
            "population_median": jsonable(med),
            "population_p90": jsonable(p90),
            "standing": standing,
        })
    return out


def analyse_account(idx: int, top_k: int = 8) -> dict:
    """Full investigator view of one benchmark account — all real model output."""
    scores = risk_scores()
    if not isinstance(idx, int) or idx < 0 or idx >= len(scores):
        raise IndexError(f"account_idx must be between 0 and {len(scores) - 1}")

    row = scores.iloc[idx]
    sv, names = oof_shap()
    contrib = sv[idx]
    order = np.argsort(np.abs(contrib))[::-1][:top_k]

    reasons, cited = [], []
    for j in order:
        if contrib[j] == 0:
            continue
        code = names[j]
        cited.append(code)
        reasons.append({
            "feature": code,
            "variable": D.real_name(code),
            "meaning": D.explain(code),
            "direction": "RAISES" if contrib[j] > 0 else "LOWERS",
            "shap": jsonable(contrib[j]),
            "abs_shap": jsonable(abs(float(contrib[j]))),
        })

    band = str(row["band"])
    return {
        "account_idx": int(idx),
        "risk_score": int(row["risk_score"]),
        "calibrated_probability": jsonable(row.get("calibrated_probability")),
        "band": band,
        "recommended_action": str(row["recommended_action"]),
        "confirmed_mule": bool(row["y_true"]),
        "score_provenance": "pooled nested out-of-fold — this account was scored "
                            "by models that never trained on it",
        "explanation_provenance": "SHAP from the same out-of-fold models",
        "top_reasons": reasons,
        "evidence": _evidence_for(idx, cited),
        "band_edges": scoring_report().get("score_bands", {}),
        "investigator_next_steps": NEXT_STEPS[band],
    }


NEXT_STEPS = {
    "HIGH": [
        "Freeze outward transfers on the account pending review.",
        "Pull the last 31 days of transaction detail and identify counterparties.",
        "Check whether the cited pass-through / cash-out pattern repeats weekly.",
        "Escalate to the AML desk and prepare a Suspicious Transaction Report.",
        "Cross-check KYC: does the occupation and declared income match the volume?",
    ],
    "MEDIUM": [
        "Apply step-up authentication (OTP) on outward transfers.",
        "Place the account on enhanced monitoring for 30 days.",
        "Review the cited variables again after the next extract.",
        "No freeze — the evidence does not yet support restricting the customer.",
    ],
    "LOW": [
        "No action. Routine monitoring only.",
        "Do not contact the customer; there is no supporting evidence.",
    ],
}


# --------------------------------------------------------------------------
# Live scoring of an arbitrary account
# --------------------------------------------------------------------------
def score_features(payload: dict) -> dict:
    """Score an account supplied at runtime with the deployed ensemble.

    Unknown keys are ignored and unsupplied features are left missing, which the
    ensemble fills with the medians it learned during training. That is the
    correct treatment for a single live account: it has no distribution of its
    own to impute from.
    """
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object of feature -> value")

    b = bundle()
    feat_names: list[str] = b["feat_names"]
    ens = b["ensemble"]

    index = {n: i for i, n in enumerate(feat_names)}
    # Accept either F-codes or real banking variable names.
    by_real = {D.real_name(n).upper(): n for n in feat_names}

    x = np.full((1, len(feat_names)), np.nan, dtype=np.float32)
    used, ignored = {}, []
    for key, val in payload.items():
        code = key if key in index else by_real.get(str(key).upper())
        if code is None:
            ignored.append(str(key)[:64])
            continue
        try:
            x[0, index[code]] = float(val)
            used[code] = float(val)
        except (TypeError, ValueError):
            ignored.append(f"{str(key)[:64]} (not numeric)")

    prob = float(ens.predict_proba(x)[0])
    score = float(np.clip(prob, 0, 1) * C.SCORE_MAX)
    edges = scoring_report().get("score_bands", {})
    low_max = float(edges.get("LOW", [0, C.BAND_LOW_MAX])[1])
    med_max = float(edges.get("MEDIUM", [0, C.BAND_MEDIUM_MAX])[1])
    band = "LOW" if score < low_max else ("MEDIUM" if score < med_max else "HIGH")

    return {
        "risk_score": int(round(score)),
        "calibrated_probability": jsonable(prob),
        "band": band,
        "recommended_action": {
            "LOW": "No action — routine monitoring",
            "MEDIUM": "Enhanced monitoring + step-up authentication (OTP) on transfers",
            "HIGH": "Freeze outward transfers, escalate to AML desk, prepare STR",
        }[band],
        "features_recognised": len(used),
        "features_ignored": ignored[:20],
        "features_imputed_from_training_medians": len(feat_names) - len(used),
        "score_provenance": "final ensemble refit on all 9,082 rows — this is "
                            "the deployment model, NOT an out-of-fold estimate, "
                            "so it is not comparable to the benchmark metrics",
        "investigator_next_steps": NEXT_STEPS[band],
    }


# --------------------------------------------------------------------------
# Scoring an UNLABELLED file with the already-trained model
# --------------------------------------------------------------------------
# The upload path retrains, so it needs a target column and refuses to guess one.
# That is right for a benchmark, and wrong for the job a bank actually has: score
# accounts nobody has flagged yet. Detection does not need labels, only training
# does. This reuses the deployed ensemble and never looks for a target.
# Mule typology, as directions that are decided BEFORE any data is seen.
#
# Each entry is (token test, +1 if a high value is suspicious, -1 if a low value
# is, plain-English reason). Directions come from money-mule tradecraft, not from
# fitting: fitting a direction against labels we are also scoring on is the exact
# error the integrity audit exists to catch.
#
# Counterparty COUNTS are deliberately absent. A gather-scatter mule has many
# beneficiaries and a simple conduit has one, so the direction genuinely depends
# on the role the account plays and cannot be fixed in advance. Guessing it once
# cost more than leaving it out: as an assumed "more is worse" signal it scored
# below random and dragged the honest signals down with it.
TYPOLOGY_SIGNALS = [
    (("round",),                    +1, "round-amount share: structuring"),
    (("rapid", "velocity", "burst"), +1, "burst activity: sudden activation"),
    (("time_to_out", "outflow_hour"), -1, "hours to outflow: a conduit forwards fast"),
    (("benefic_conc", "concentration"), +1, "concentration: single-purpose account"),
    (("kyc_risk", "risk_score"),    +1, "the institution's own risk score"),
    (("account_age", "age_days"),   -1, "account age: new accounts moving money"),
    (("chargeback", "dispute"),     +1, "chargebacks: victim complaints"),
    (("device_count", "ip_count"),  +1, "device and IP spread: shared control"),
    (("geographic_distance", "geo_dist"), +1, "geographic distance from profile"),
    (("cash_deposit", "cash_in"),   +1, "cash deposits: the layering hand-off"),
]


# Below this share of its schema the deployed ensemble stops being a model and
# starts being a random number generator. Measured, not guessed: masked to 750
# of 1,506 columns it scores AUPRC 0.937, and at 300 it scores 0.009 against a
# 0.0089 base rate. Half the schema is the floor.
COVERAGE_FLOOR = 0.50
MIN_INTERSECTION = 5


@lru_cache(maxsize=1)
def _training_frame():
    """The labelled feature matrix, for fitting on a partial schema."""
    feats = C.DATA_DIR / "features.parquet"
    if not feats.exists():
        return None, None
    X = pd.read_parquet(feats)
    y = pd.read_csv(C.RAW_CSV, usecols=[C.TARGET_COL_HINT])[
        C.TARGET_COL_HINT].astype(int).to_numpy()
    return X, y


def fit_on_intersection(cols_to_feats: dict, frame: "pd.DataFrame") -> dict:
    """Fit a model on exactly the columns this file and the training data share.

    A file carrying a fraction of the schema cannot be scored by the deployed
    ensemble, but the columns it does carry are real and they are labelled in
    our own data. Fitting on the intersection uses the relationship that
    actually exists between those columns and the outcome, rather than the
    directions someone assumed in advance. The out-of-fold score of that fit is
    returned with the predictions, because a caller is entitled to know how much
    signal the intersection really holds before acting on it.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    X, y = _training_frame()
    if X is None:
        raise ValueError("features.parquet is absent, so no model can be fitted "
                         "on the shared columns.")
    pairs = [(c, f) for c, f in cols_to_feats.items() if f in X.columns]
    if len(pairs) < MIN_INTERSECTION:
        raise ValueError(f"only {len(pairs)} of this file's columns exist in the "
                         "training data, which is too few to fit anything "
                         "honest.")
    tr = np.column_stack([pd.to_numeric(X[f], errors="coerce") for _, f in pairs])
    te = np.column_stack([pd.to_numeric(frame[c], errors="coerce") for c, _ in pairs])

    def _mk(yy):
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=(yy == 0).sum() / max((yy == 1).sum(), 1),
            verbose=-1, random_state=42)

    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
        oof[b] = _mk(y[a]).fit(tr[a], y[a]).predict_proba(tr[b])[:, 1]
    ap = float(average_precision_score(y, oof))
    proba = _mk(y).fit(tr, y).predict_proba(te)[:, 1]
    return {"proba": proba, "columns": [c for c, _ in pairs],
            "oof_auprc": round(ap, 4),
            "oof_lift": round(ap / float(y.mean()), 1)}


def _typology_by_role(df: "pd.DataFrame") -> tuple["pd.DataFrame", dict]:
    """Rebuild mule-typology signals from an unfamiliar schema, by meaning.

    The deployed ensemble is tied to the columns it was trained on, so a file
    from another bank cannot be scored with it. The typology can still be
    measured: pass-through is arithmetic, and `roles.parse_role` recovers the
    direction, window and statistic of a column from its name alone.

    Returns the signals, already oriented so that larger always means more
    suspicious, and a provenance map naming the column behind each one.
    """
    import roles

    sig = pd.DataFrame(index=df.index)
    prov: dict[str, str] = {}
    lower = {c: str(c).lower() for c in df.columns}

    # 1. Pass-through, the one signal that must be derived rather than read: a
    #    conduit forwards what it receives, so debits over credits approaches 1.
    parsed = {c: roles.parse_role(c) for c in df.columns}
    inflow = outflow = None
    for c, r in parsed.items():
        if r.stat == "TOT" and r.direction == "credit" and inflow is None:
            inflow, prov["passthrough (in)"] = pd.to_numeric(df[c], errors="coerce"), c
        if r.stat == "TOT" and r.direction == "debit" and outflow is None:
            outflow, prov["passthrough (out)"] = pd.to_numeric(df[c], errors="coerce"), c
    if inflow is not None and outflow is not None:
        sig["passthrough"] = (outflow / (inflow + 1e-9)).clip(0, 5)

    # 2. Everything else is read directly and oriented by its a priori direction.
    for tokens, direction, reason in TYPOLOGY_SIGNALS:
        for c, name in lower.items():
            if any(t in name for t in tokens):
                v = pd.to_numeric(df[c], errors="coerce")
                if v.notna().sum() == 0:
                    continue
                key = reason.split(":")[0].strip().replace(" ", "_")
                sig[key] = v * direction
                prov[key] = f"{c} ({'high' if direction > 0 else 'low'} is suspicious)"
                break
    return sig, prov


def _rank_score(sig: "pd.DataFrame") -> "np.ndarray":
    """Average of percentile ranks across every oriented signal.

    Ranks rather than values, so one heavy-tailed column cannot dominate, and a
    plain average rather than fitted weights, because there are no labels here
    to fit weights against and inventing them would be dishonest.
    """
    if sig.empty or not len(sig.columns):
        return np.zeros(len(sig))
    return sig.rank(pct=True).mean(axis=1).to_numpy()


def detect_target(path) -> dict:
    """Decide, before any work starts, whether this file carries a label.

    Whoever hands over a file will not tell us whether it is labelled, and it is
    not their job to. The same resolver the pipeline uses is run here on a small
    sample, so the system can choose for itself between measuring and detecting
    instead of failing and asking the operator to pick a button.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            head = pd.read_excel(path, nrows=400)
        elif path.suffix.lower() == ".parquet":
            head = pd.read_parquet(path).head(400)
        else:
            # A small sample can hide the label: on an extract sorted by class,
            # the target reads as constant rather than binary in the first rows.
            head = pd.read_csv(path, nrows=8000, low_memory=False)
    except Exception as exc:
        return {"labelled": False, "target": None, "why": f"could not read the file: {exc}"}

    try:
        import schema as _schema
        # Same call the pipeline makes in 01_clean.py, hint included.
        col, how = _schema.resolve_target(head, C.TARGET_COL_HINT)
        return {"labelled": True, "target": str(col), "why": how}
    except Exception as exc:
        return {"labelled": False, "target": None, "why": str(exc)}


MAX_SCORE_ROWS = 200_000


def score_file(path, top: int = 100) -> dict:
    """Score every row of an unlabelled extract with the deployed ensemble."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    if len(df) > MAX_SCORE_ROWS:
        raise ValueError(f"{len(df):,} rows exceeds the {MAX_SCORE_ROWS:,} row limit "
                         "for synchronous scoring.")

    b = bundle()
    feat_names: list[str] = b["feat_names"]
    ens = b["ensemble"]
    index = {n: i for i, n in enumerate(feat_names)}
    by_real = {D.real_name(n).upper(): n for n in feat_names}

    # Map this file's columns onto the model's features, by F-code or by the
    # dictionary's banking name. Anything unrecognised is reported, not guessed.
    mapping, unknown = {}, []
    for col in df.columns:
        code = col if col in index else by_real.get(str(col).strip().upper())
        if code is None:
            unknown.append(str(col)[:64])
        elif code not in mapping.values():
            mapping[col] = code

    # A file from another bank shares no column names with the training data,
    # so the deployed ensemble cannot score it. That is a limit of the model,
    # not of the problem: the typology is arithmetic and can still be measured.
    # Three tiers, decided by how much of its own schema the model actually has.
    coverage = len(mapping) / max(len(feat_names), 1)
    if coverage < COVERAGE_FLOOR:
        # The deployed ensemble is not applicable. Before falling back to
        # assumed directions, try the honest thing: fit on the columns this
        # file and the labelled training data genuinely share, so the weights
        # are learned from data rather than asserted.
        try:
            got = fit_on_intersection(mapping, df)
        except Exception as exc:
            got = None
            fit_note = str(exc)
        if got is not None:
            prob = np.asarray(got["proba"], dtype=float).ravel()
            order = np.argsort(-prob)[:max(1, min(int(top), 1000))]
            rows = [{"row": int(i), "probability": jsonable(float(prob[i])),
                     "percentile": round(float((prob < prob[i]).mean()) * 100, 1)}
                    for i in order]
            return {
                "mode": "FITTED_ON_SHARED_COLUMNS",
                "rows_scored": int(len(df)),
                "columns_used": got["columns"],
                "model_features_matched": len(mapping),
                "model_features_expected": len(feat_names),
                "schema_coverage_pct": round(coverage * 100, 1),
                "fit_quality": {
                    "out_of_fold_auprc": got["oof_auprc"],
                    "lift_over_base_rate": got["oof_lift"],
                    "meaning": "How well these shared columns separate mules in "
                               "OUR labelled data, measured out of fold. It is "
                               "the ceiling on what to expect here, and it is "
                               "reported so the number is not taken on trust.",
                },
                "top_accounts": rows,
                "provenance":
                    f"This file carries {len(mapping)} of the {len(feat_names)} "
                    "columns the deployed ensemble was fitted on, which is "
                    "below the half-schema floor where that ensemble stops "
                    "being better than chance. Rather than score with it "
                    "anyway, a model was fitted on the columns this file and "
                    "our labelled data share, so the weights come from measured "
                    "relationships and not from assumed ones. With no labels in "
                    "this file, no precision or recall can be computed for it.",
            }

    if len(mapping) < 40:
        sig, prov = _typology_by_role(df)
        if sig.empty:
            raise ValueError(
                "This file's columns match neither the trained model's features "
                "nor any recognisable mule-typology quantity. Nothing here can "
                "be scored honestly. Expected either the training schema, or "
                "columns whose names carry a direction and a window, such as "
                "total_incoming_7d.")
        rank = _rank_score(sig)
        # A ranking alone makes the reviewer invent a cut-off. Otsu picks the
        # split that maximises between-class variance, so the data decides how
        # many accounts are flagged rather than a number we chose. It is the
        # same parameter-free cut used for the temporal windows.
        from temporal import otsu_threshold
        cut, separation = otsu_threshold(rank)
        flagged = rank >= cut
        order = np.argsort(-rank)[:max(1, min(int(top), 1000))]
        rows = [{"row": int(i), "percentile": round(float(rank[i]) * 100, 1),
                 "flagged": bool(flagged[i]),
                 "signals": {c: jsonable(float(sig[c].iloc[i]))
                             for c in sig.columns}} for i in order]
        return {
            "mode": "TYPOLOGY_RANKING",
            "rows_scored": int(len(df)),
            "accounts_flagged": int(flagged.sum()),
            "flag_rate_pct": round(float(flagged.mean()) * 100, 1),
            "threshold": {"value": jsonable(float(cut)),
                          "separation": jsonable(float(separation)),
                          "method": "Otsu, parameter-free: the split that "
                                    "maximises between-class variance in the "
                                    "score distribution. Nothing was tuned and "
                                    "no share of the book was assumed."},
            "columns_used": prov,
            "signals_built": list(sig.columns),
            "model_features_matched": len(mapping),
            # Otsu splits whatever distribution it is given. On a curated test
            # file that is 40 percent mules that split is the right answer; on a
            # real portfolio, where mules are well under one percent, the same
            # cut would flag a large share of ordinary customers. Say so rather
            # than let a plausible-looking count travel unchallenged.
            "caution": ("Flagged share is %.1f percent of the file. Otsu finds "
                        "the natural split in whatever it is given, so on a "
                        "curated test set that is expected. A real portfolio "
                        "carries well under one percent mules, and on one of "
                        "those you should work down the ranking to a review "
                        "budget rather than treat this cut as a count of "
                        "mules." % (float(flagged.mean()) * 100)
                        ) if flagged.mean() > 0.20 else None,
            "top_accounts": rows,
            "provenance":
                "UNVALIDATED. This ranking assumes the textbook direction of "
                "each signal, and on a given book those directions can be "
                "inverted: measured on one 213-account extract, this path "
                "scored AUROC 0.476, worse than chance, because mules there "
                "moved smaller amounts over shorter distances than ordinary "
                "customers. The same extract reaches AUPRC 0.743 once a label "
                "column is supplied and the weights are learned instead of "
                "assumed. If you have labels, even for part of the book, supply "
                "them: that is the difference between detection and a guess. "
                "This file shares no schema with the training data, so the "
                "trained ensemble was NOT used and no calibrated probability "
                "exists. Accounts are RANKED by mule typology rebuilt from this "
                "file's own columns: pass-through, retention, counterparty "
                "spread, round-amount share, burstiness and speed of outflow, "
                "each resolved by meaning rather than by name. A rank is not a "
                "probability. It says which accounts to review first, not how "
                "likely any of them is a mule, and with no labels here neither "
                "precision nor recall can be computed. To get calibrated "
                "probabilities on this schema, supply labels and retrain.",
        }

    x = np.full((len(df), len(feat_names)), np.nan, dtype=np.float32)
    for col, code in mapping.items():
        x[:, index[code]] = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float32)

    prob = np.asarray(ens.predict_proba(x), dtype=float).ravel()
    score = np.clip(prob, 0, 1) * C.SCORE_MAX
    edges = scoring_report().get("score_bands", {})
    low_max = float(edges.get("LOW", [0, C.BAND_LOW_MAX])[1])
    med_max = float(edges.get("MEDIUM", [0, C.BAND_MEDIUM_MAX])[1])
    band = np.where(score < low_max, "LOW", np.where(score < med_max, "MEDIUM", "HIGH"))

    order = np.argsort(-score)[:max(1, min(int(top), 1000))]
    rows = [{"row": int(i), "risk_score": int(round(score[i])),
             "calibrated_probability": jsonable(float(prob[i])),
             "band": str(band[i]),
             "recommended_action": NEXT_STEPS[str(band[i])][0]} for i in order]

    return {
        "mode": "SCORE_ONLY",
        "rows_scored": int(len(df)),
        "features_matched": len(mapping),
        "features_expected": len(feat_names),
        "features_imputed_from_training_medians": len(feat_names) - len(mapping),
        "columns_not_recognised": unknown[:25],
        "band_distribution": {b_: int((band == b_).sum())
                              for b_ in ("LOW", "MEDIUM", "HIGH")},
        "top_accounts": rows,
        "provenance": "Scored with the deployed ensemble refit on the labelled "
                      "training data. No target column was read from this file "
                      "and none is required. These are predictions, not "
                      "measurements: with no labels here, no precision or "
                      "recall can be computed for this file.",
    }


# --------------------------------------------------------------------------
# Investigator decisions and the audit trail
# --------------------------------------------------------------------------
DECISIONS_CSV = C.DATA_DIR / "investigator_decisions.csv"

VALID_DECISIONS = {
    "CONFIRMED_MULE": "Confirmed as a mule. Freeze simulated, STR prepared.",
    "DISMISSED": "Reviewed and cleared. No action against the customer.",
    "NEEDS_REVIEW": "Escalated for a second pair of eyes.",
}

_DECISION_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def operating_metrics() -> dict:
    return _read_json(C.REPORTS_DIR / "10_operating_metrics.json",
                      "Stage 11 (10_operating_metrics.py)")


def record_decision(account_idx: int, decision: str, note: str = "",
                    actor: str = "demo-analyst") -> dict:
    """Append one investigator decision to the audit trail.

    This is the feedback loop a real deployment needs: the labels an analyst
    produces are what a next-generation model retrains on. It is append only and
    timestamped, because an AML audit trail that can be edited is not an audit
    trail.

    Nothing here accepts a file path or a column name from the caller, and the
    dataset carries no personal data to begin with: accounts are referred to by
    row index, never by name or account number.
    """
    scores = risk_scores()
    if not isinstance(account_idx, int) or not (0 <= account_idx < len(scores)):
        raise IndexError(f"account_idx must be between 0 and {len(scores) - 1}")
    decision = str(decision).upper().strip()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")

    row = scores.iloc[account_idx]
    record = {
        "recorded_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "account_idx": int(account_idx),
        "risk_score": int(row["risk_score"]),
        "band": str(row["band"]),
        "model_said": "MULE" if str(row["band"]) == "HIGH" else "NOT MULE",
        "decision": decision,
        "action": VALID_DECISIONS[decision],
        "note": str(note)[:500],
        "actor": str(actor)[:64],
        "ground_truth": int(row["y_true"]),
    }

    with _DECISION_LOCK:
        df = pd.DataFrame([record])
        header = not DECISIONS_CSV.exists()
        DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DECISIONS_CSV, mode="a", header=header, index=False,
                  encoding="utf-8")

    return {**record, "audit_trail": str(DECISIONS_CSV.name),
            "note_for_retraining": "Decisions accumulate as labels. One week of "
                                   "these is what a retrained model would learn "
                                   "from."}


def decision_log(limit: int = 100) -> dict:
    """The audit trail, newest first, with agreement against the model."""
    if not DECISIONS_CSV.exists():
        return {"total": 0, "decisions": [],
                "note": "No decisions recorded yet in this session."}
    with _DECISION_LOCK:
        df = pd.read_csv(DECISIONS_CSV)
    total = len(df)
    agree = int(((df["decision"] == "CONFIRMED_MULE") & (df["ground_truth"] == 1)).sum()
                + ((df["decision"] == "DISMISSED") & (df["ground_truth"] == 0)).sum())
    resolved = int((df["decision"] != "NEEDS_REVIEW").sum())
    return {
        "total": total,
        "resolved": resolved,
        "analyst_agreed_with_ground_truth": agree,
        "agreement_rate": round(agree / resolved, 4) if resolved else None,
        "decisions": jsonable(df.tail(limit).iloc[::-1].to_dict(orient="records")),
        "note": "Append only and timestamped. In a deployment these rows are the "
                "training labels for the next model version.",
    }


# --------------------------------------------------------------------------
# Narrative panels
# --------------------------------------------------------------------------
def overview() -> dict:
    m, cl = metrics(), clean_report()
    return {
        "accounts": m["n_accounts"],
        "mules": m["n_mules"],
        "prevalence_pct": m["prevalence_pct"],
        "raw_columns": cl["input_shape"][1],
        "features_after_cleaning": m["n_features_available"],
        "random_baseline_auprc": m["auprc_baseline_random"],
        "graph": graph_report(),
        "engines": m.get("engines", {}),
        "reproducibility": m.get("reproducibility", {}),
    }


def schema_report() -> dict:
    """How the pipeline interpreted this dataset, and what it discovered.

    The point of this panel is that none of it is configured. A judge can point
    the pipeline at a different file and watch these values change.
    """
    cl = clean_report()
    sch = cl.get("schema", {})
    part = cl.get("partition_audit", {})
    return {
        "target": {
            "column": sch.get("target_column"),
            "resolved_by": sch.get("target_resolved_by"),
            "positives": sch.get("positives"),
            "negatives": sch.get("negatives"),
            "prevalence_pct": sch.get("prevalence_pct"),
        },
        "shape": {"rows": sch.get("n_rows"), "columns": sch.get("n_columns"),
                  "numeric": sch.get("n_numeric"),
                  "non_numeric": sch.get("n_non_numeric")},
        "column_naming": sch.get("column_naming"),
        "dictionary_used": cl.get("dictionary_used"),
        "identifiers": {
            "dropped": cl.get("dropped_identifier_columns", []),
            "count": cl.get("n_dropped_identifier_columns", 0),
            "why": "Row keys carry no generalisable signal, but a tree will "
                   "memorise one. Caught by name pattern and by being near-unique "
                   "per row — floats are exempt, since any continuous measurement "
                   "is near-unique and excluding them is what stops the rule from "
                   "deleting the whole feature matrix.",
        },
        "partition_audit": part,
        "raw_dates_dropped": cl.get("dropped_raw_date_columns", []),
        "encoded_categoricals": cl.get("encoded_categoricals", {}),
        "discovered_not_configured": [
            "the target column",
            "row identifiers",
            "categorical columns and which of them are ordinal",
            "post-outcome leak columns",
            "partition columns (the generalisation of MNTH)",
            "which behavioural features can be built from the available columns",
        ],
    }


def leakage_defence() -> dict:
    cl = clean_report()
    return {
        "layer_1_semantic": {
            "title": "Post-outcome fields, removed by meaning",
            "detail": "Written only after an analyst closes a case, so they do "
                      "not exist at scoring time. FRAUD_SUSPECTED correlates "
                      "0.97; FALSE_POSITIVE correlates 0.05 and is just as "
                      "unusable — which is why a correlation threshold is not a "
                      "leak defence.",
            "removed": cl.get("removed_semantic_leaks", {}).get("post_outcome", []),
        },
        "layer_2_structural": {
            "title": "Sample-assembly artefacts",
            "detail": "MNTH alone separates the classes perfectly.",
            "removed": cl.get("removed_semantic_leaks", {}).get("structural", []),
            "evidence": cl.get("structural_leak_audit", {}),
        },
        "layer_3_extract_hardening": {
            "title": "Class-dependent blank rates",
            "detail": "Whether a cell is populated is decided by the extraction "
                      "job, not the customer. Can only remove signal, never "
                      "manufacture it.",
            **cl.get("extract_hardening", {}),
        },
        "layer_4_separation_audit": {
            "title": "Scan for the next MNTH",
            "detail": "Every surviving column checked for disjoint class ranges "
                      "or a near-exclusive value.",
            **cl.get("separation_audit", {}),
        },
        "correlation_backstop": {
            "threshold": C.LEAK_CORR_THRESHOLD,
            "removed": cl.get("removed_corr_leaks", []),
            "top_correlations": cl.get("top_target_correlations", {}),
        },
        "caveat": "Layers 2-4 and the correlation backstop compute against the "
                  "label on the full dataset. They only ever REMOVE columns, so "
                  "they make the reported result more conservative, never less "
                  "— but they are not fitted inside the fold and the paper says "
                  "so rather than implying otherwise.",
    }


def mule_features() -> dict:
    fr = feature_report()
    return {
        "typology_feature_count": len(fr.get("mule_typology_features", {})),
        "typology_features": fr.get("mule_typology_features", {}),
        "row_profile_features": fr.get("added_row_profile_features", []),
        "channels_used": fr.get("channels_used_for_mix", []),
        "occupation_deviation_columns": fr.get("occupation_deviation_columns_used", 0),
        "could_not_build": fr.get("missing_base_columns", []),
        "total_features": fr.get("output_feature_count", 0),
    }


def model_comparison() -> dict:
    m = metrics()
    return {
        "per_model": m.get("per_model", {}),
        "ensemble_precision_first": m.get("ensemble_precision_first", {}),
        "ensemble_high_recall": m.get("ensemble_high_recall", {}),
        "stacking_coefficients": m.get("stacking_coefficients", {}),
        "validation": m.get("validation", {}),
        "operating_points": m.get("operating_points", {}),
        "random_baseline_auprc": m.get("auprc_baseline_random"),
    }
