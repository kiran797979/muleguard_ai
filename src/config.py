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

# --------------------------------------------------------------------------
# Cleaning parameters
# --------------------------------------------------------------------------
MISSING_DROP_FRAC = 0.50      # drop columns missing more than this fraction
COLLINEAR_CORR_THRESHOLD = 0.98  # de-duplicate near-identical rolling-window copies

# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 42

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

# --------------------------------------------------------------------------
# Graph / label-propagation (Stage 6). Only used if edge data is detected.
# --------------------------------------------------------------------------
HOP_DECAY = {1: 0.85, 2: 0.70, 3: 0.55}   # first / second / third hop scores
MAX_HOPS = 3

# --------------------------------------------------------------------------
# Model hyper-parameters (conservative, imbalance-aware defaults)
# scale_pos_weight is computed at runtime from the actual class ratio, but a
# fallback is provided here for reference (~9001/81 ≈ 111).
# --------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
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
