"""
MuleGuard AI — central configuration (single source of truth).

Every stage imports paths, column names, and hyper-parameters from here so
there is exactly one place to change behaviour. No magic numbers scattered
across scripts.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../muleguard
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Raw dataset (place the hackathon file here)
RAW_CSV = DATA_DIR / "DataSet.csv"

# Intermediate artefacts written between stages
CLEAN_PARQUET = DATA_DIR / "clean.parquet"             # after Stage 1
FEATURES_PARQUET = DATA_DIR / "features.parquet"       # after Stage 2/3
GRAPH_SCORES_CSV = DATA_DIR / "graph_scores.csv"       # after Stage 6

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset schema (from the submission PDF)
# --------------------------------------------------------------------------
TARGET_COL = "F3924"          # binary: 0 = normal, 1 = mule
LEAK_COL = "F3912"            # known target leak (~0.97 corr) — remove before modelling

# Any column whose |Pearson corr| with the target exceeds this is treated as a
# suspected leak and flagged (F3912 should surface here automatically).
LEAK_CORR_THRESHOLD = 0.90

# Non-feature identifier columns to drop before modelling. The real CSV is
# exported by R's write.csv, which prepends an unnamed row-id column ("Unnamed: 0"
# after pandas parses it). It is a unique 1..N counter — useless as a feature and
# a mild ordering leak (the 81 mules occupy the final rows), so it must go.
ID_COL_PREFIXES = ("Unnamed:",)   # any column whose name starts with these is dropped

# --------------------------------------------------------------------------
# Cleaning parameters
# --------------------------------------------------------------------------
MISSING_DROP_FRAC = 0.50      # drop columns missing more than this fraction
COLLINEAR_CORR_THRESHOLD = 0.98  # de-duplicate near-identical rolling-window copies

# Categorical (string) columns are encoded rather than silently coerced to NaN.
# Low-cardinality string columns (<= this many distinct values) become one-hot
# dummies; higher-cardinality string columns are kept only if they parse as
# dates (-> numeric account "vintage" in days), otherwise dropped as ID/free-text.
CATEGORICAL_MAX_CARDINALITY = 30
DATE_PARSE_MIN_FRAC = 0.80    # a high-card column is a date if >= this frac parses

# Categorical leak guard. A categorical column is a target leak if knowing the
# category almost perfectly predicts the label. We measure this as the reduction
# in majority-class error achieved by conditioning on the category; >= this
# fraction means the column essentially *is* the label and is dropped BEFORE
# one-hot encoding (so no subset of its dummies can smuggle the leak in).
# This catches F2230 (sampling month: every normal is 'Oct25', every mule isn't
# — a 100% dataset-assembly artifact, not a generalizable fraud signal).
CATEGORICAL_LEAK_ERROR_REDUCTION = 0.98

# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 42

# Honest operating-point estimation (Stage 4/5). The precision-first threshold and
# isotonic calibrator are chosen inside a NESTED cross-validation and applied to
# untouched outer-test folds, so the reported operating point is out-of-sample —
# not the same pooled predictions it is scored on. Repeats reduce the (large) split
# variance at only 81 positives; the pooled single-level number is kept as a
# labelled "optimistic ceiling" for comparison.
INNER_FOLDS = 5
N_CV_REPEATS = 10             # repeated nested CV — the biggest variance-reducer at N=81
BOOTSTRAP_N = 2000            # stratified bootstrap resamples for headline CIs

# Resampling policy (set from the measured ablation, reports/07_resampling_ablation.json).
# USE_SMOTE + USE_SPW together = the old double-correction (SMOTE ~50/50 THEN
# scale_pos_weight ~111 on top). The ablation picks the honest winner; default here
# is the clean, calibration-preserving spw-only configuration.
USE_SMOTE = False             # in-fold SMOTE-Tomek resampling
USE_SPW = True                # per-fold scale_pos_weight = n_neg/n_pos

# Feature-selection policy (set from reports/08_feature_selection.json). "all" keeps
# every feature; an int K keeps the in-fold top-K by XGBoost gain (selection fitted
# strictly inside each training fold).
FEATURE_SELECT_K = "all"

# --------------------------------------------------------------------------
# Risk score bands (0–1000). Cutoffs are defaults; the HIGH cutoff is
# re-derived from the precision–recall curve at training time to hold the
# target precision (see PRECISION_TARGET).
# --------------------------------------------------------------------------
SCORE_MIN = 0
SCORE_MAX = 1000
BAND_LOW_MAX = 400            # LOW:    0–400   → no action
BAND_MEDIUM_MAX = 750         # MEDIUM: 400–750 → enhanced monitoring + OTP
# HIGH: 750–1000 → auto-freeze + STR

PRECISION_TARGET = 0.90       # HIGH-band cutoff chosen to hold precision >= this

# Per-account "Account Risk Report" cards (Stage 7/8). How many of the
# highest-risk accounts to emit formatted reports + SHAP reason lists for, and
# how many top reasons to show per account.
N_RISK_REPORT_CARDS = 20
N_REASONS_PER_CARD = 5

# Recommended action per band, shown on the report cards.
BAND_ACTIONS = {
    "LOW": "No action — routine monitoring.",
    "MEDIUM": "Enhanced monitoring + step-up authentication (OTP) on high-value transfers.",
    "HIGH": "Auto-freeze outward transfers + file Suspicious Transaction Report (STR).",
}

# --------------------------------------------------------------------------
# Graph / label-propagation (Stage 6). Only used if edge data is detected.
# --------------------------------------------------------------------------
HOP_DECAY = {1: 0.85, 2: 0.70, 3: 0.55}   # first / second / third hop scores
MAX_HOPS = 3

# --------------------------------------------------------------------------
# Ensemble composition
# --------------------------------------------------------------------------
# Isolation Forest is an UNSUPERVISED anomaly detector. On this dataset it is
# uninformative for mule-vs-normal (measured out-of-fold AUROC ~0.26 — worse
# than random, because "statistical outlier" != "mule" here) and it drags the
# supervised stack down. We therefore exclude it by default; flip to True only
# if a future dataset shows anomaly signal actually helps.
USE_ISO_FOREST = False

# LightGBM + logistic stacking. Measured across 3 CV seeds, blending LightGBM in
# (via the meta-learner) HURTS the operating metric that matters here — recall at
# precision >= 0.90:
#     XGBoost solo : 0.840 +- 0.027     (wins every seed)
#     0.7*xgb+lgbm : 0.778
#     stacked ens. : 0.667  (the old default)
# LightGBM's AUPRC (~0.79) is well below XGBoost's (~0.91), so averaging it in
# pulls the strong model down. We therefore run XGBoost as the sole calibrated
# model by default. Flip USE_LGBM=True to restore the two-model stack.
USE_LGBM = False

# --------------------------------------------------------------------------
# Model hyper-parameters (imbalance-aware; regularization MEASURED, not guessed).
# scale_pos_weight is computed at runtime from the actual class ratio, but a
# fallback is provided here for reference (~9001/81 ≈ 111).
#
# max_depth=3 (was 5) and min_child_weight=1 come from the inner-CV grid search in
# reports/09_hp_search.json: across 5 seeds, {max_depth:3, min_child_weight:1}
# scores AUPRC 0.9301 ± 0.0028 vs 0.9133 for the old depth-5 default — a +0.017
# gain that clears the seed std, so it is "meaningful", not tuning noise. Shallower
# trees regularize better with only 81 positives (~20:1 feature-to-signal ratio).
# --------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "min_child_weight": 1,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}

LGBM_PARAMS = {
    "n_estimators": 400,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

ISO_FOREST_PARAMS = {
    "n_estimators": 300,
    "contamination": "auto",
    "random_state": RANDOM_STATE,
}
