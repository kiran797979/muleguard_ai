# MuleGuard AI

**Network-aware money-mule account detection — precision-first, leak-free, and honest.**

A reproducible ML pipeline for the PSB Cybersecurity, Fraud & AI Hackathon 2026
(Bank of India × IIT Hyderabad). It detects mule accounts in an extremely
imbalanced dataset (**81 mules in 9,082 accounts ≈ 0.89% prevalence**) and reports
**measured 5-fold cross-validation metrics** — not design targets — from a run on
the real hackathon data.

> **Headline (honest, leak-audited, measured out-of-sample):**
> **AUPRC 0.9345** (95% CI 0.887–0.977) and **AUROC 0.990** (95% CI 0.973–1.000),
> with an out-of-sample operating point from **repeated nested cross-validation** of
> **Precision 0.934 ± 0.017 at Recall 0.861 ± 0.022** (Wilson 95% CI 0.773–0.922),
> F1 0.896 ± 0.016 — all scored by models that never saw the account. A HIGH-risk
> auto-freeze tier is **69/71 true mules (97.2% precision)**. An older in-sample
> "optimistic ceiling" is kept below for contrast but **clearly labelled — it is not
> the headline.** Every number traces to a file in [`reports/`](reports/) — nothing
> is hand-typed.

> **Cross-platform:** runs on macOS, Windows, and Linux. The Python code uses
> `pathlib` throughout (no hardcoded paths) and UTF-8 everywhere. Only the
> *setup commands* differ per OS — both are given below.

> **👩‍⚖️ Presenting to judges?** Jump to **[§12 — Live demo walkthrough](#12--live-demo-walkthrough-for-judges)**
> for a rehearsed, minute-by-minute script (setup, run, and the exact commands
> that showcase the leak-free honesty story).

---

## Table of contents

1. [Results](#1-results-measured-on-the-real-data)
2. [The honesty story — why these numbers survive scrutiny](#2-the-honesty-story--why-these-numbers-survive-scrutiny)
3. [Prerequisites](#3-prerequisites)
4. [Setup](#4-setup)
5. [Run](#5-run)
6. [How it works — the pipeline](#6-how-it-works--the-pipeline)
7. [Outputs](#7-outputs)
8. [Configuration](#8-configuration)
9. [Folder layout](#9-folder-layout)
10. [Troubleshooting](#10-troubleshooting)
11. [Status & reproducibility](#11-status--reproducibility)
12. [Live demo walkthrough (for judges)](#12--live-demo-walkthrough-for-judges)

---

## 1. Results (measured on the real data)

All metrics are **measured** on the real 9,082-account `DataSet.csv` (`RANDOM_STATE=42`),
not design targets. The headline ranking and honest operating point are computed
**out-of-sample** — the honest point via **repeated nested cross-validation** (10 repeats),
so the threshold is never chosen on the data it scores. Source files in [`reports/`](reports/).

### Headline ranking (out-of-sample, threshold-free)

| Metric | Value | 95% CI |
|---|---|---|
| **AUPRC** | **0.9345** | 0.887 – 0.977 |
| **AUROC** | **0.990** | 0.973 – 1.000 |
| Brier | 0.00148 | — |
| ECE | 0.0012 | — |

Ranking and calibration are computed on nested-CV pooled probabilities — no operating
point is baked in.

### Honest operating point (repeated nested CV, out-of-sample, 10 repeats)

| Metric | Value |
|---|---|
| **Precision** | **0.934 ± 0.017** |
| **Recall** | **0.861 ± 0.022** (Wilson 95% CI 0.773–0.922) |
| **F1** | **0.896 ± 0.016** |
| Mean TP / FP | 69.7 / 4.9 |
| Per-fold threshold spread | 0.25 – 1.00 |

The threshold is re-selected inside each fold and never sees its own scoring rows.
At 81 positives the per-fold threshold is genuinely **noisy** (spread 0.25–1.00); we
report that spread rather than hide it — the point estimate is stable, its exact
cutoff is not.

### Optimistic ceiling — **NOT the headline** (in-sample, for contrast only)

| Metric | Value |
|---|---|
| Precision | 0.912 |
| Recall | 0.901 |

This is the **old** operating point: the threshold is picked on the same out-of-fold
predictions it then scores. It flatters itself and is retained only to show the size
of that optimism. **Do not cite it as a result.**

> **Removing the optimism did not cost performance — it raised the real ceiling.**
> The honest *out-of-sample* recall (**0.861**) is now **higher** than the old
> *in-sample* headline recall (0.802), because the measured improvements below lifted
> the entire precision–recall curve. Note also: **accuracy is meaningless at 0.89%
> prevalence** (predicting "all normal" scores 99.11%) and is deliberately never used.

### Risk-band distribution

Bands from [`reports/05_scoring_report.json`](reports/05_scoring_report.json):

| Band | Count | Action | HIGH-band precision |
|---|---|---|---|
| LOW | 9,008 | none | — |
| MEDIUM | 3 | enhanced monitoring + OTP | — |
| **HIGH** | **71** | **auto-freeze + STR** | **69 / 71 = 0.972** |

**Model choice — why XGBoost solo, spw-only, `max_depth=3`.** Every structural choice
is now measured (not assumed) — see §2's ablation summary and [`reports/07`–`09`](reports/):
the resampling ablation picks `scale_pos_weight` alone over the old SMOTE + weighting
double-correction; keeping all features beats trimming; and `max_depth=3` lifts AUPRC
over the old depth-5 default across 5 seeds. LightGBM and Isolation Forest remain
**disabled by default** (`USE_LGBM`, `USE_ISO_FOREST` in `config.py`) — measured across
seeds they *dilute* XGBoost. The win here was **subtraction**, and it is measured.

---

## 2. The honesty story — why these numbers survive scrutiny

This project's design principle is **an honest number a judge can trust beats an
inflated one that collapses under questioning.** Concretely:

### Two target leaks were found and removed

1. **`F3912`** — Pearson correlation **0.969** with the target. Caught automatically
   by a correlation scan (any \|corr\| > `LEAK_CORR_THRESHOLD` = 0.90 is dropped)
   and removed before any model sees it.

2. **`F2230` — a *perfect*, subtler leak.** It is the record's **sampling month**:
   in this dataset *every one of the 9,001 normal accounts is labelled `Oct25`*,
   while *every mule is `Sep25` / `Nov25` / `Dec25`*. The month alone identifies
   all mules with 100% accuracy — a **dataset-assembly artifact** (normals drawn
   from an October snapshot, confirmed mules from other months' investigations),
   **not** a generalizable fraud signal. A model that "learned" it would score
   near-perfect here and **fail completely in production**, where every live
   account shares the current month. MuleGuard detects this generically — a
   categorical whose category almost perfectly determines the label (error
   reduction ≥ `CATEGORICAL_LEAK_ERROR_REDUCTION` = 0.98) — and **drops the whole
   column before one-hot encoding**, so no subset of month-dummies can smuggle it
   back in.

### No leakage across folds
Any resampling and per-fold `scale_pos_weight` are fitted **inside** each stratified
training fold — never on validation rows, never on the full dataset. The honest
operating point goes further: its calibrator and threshold are fitted on **inner**
out-of-fold predictions and applied only to the **untouched outer-test** fold
(nested CV), so no threshold is ever chosen on the data it scores.

### Verified: no residual leak
After cleaning, the **most-separating single feature** in the entire matrix has a
univariate AUROC of only **0.83**. The model's 0.99 AUROC is therefore genuine
**multivariate** signal — many moderate features combining — not one column
memorising the label. If a leak had survived, some single feature would sit at
≈1.0. None does.

### Measured, not argued: the month/cohort leak audit
The month-cohort concern deserves more than a claim, so it was **audited numerically**
([`reports/06_leak_audit.json`](reports/06_leak_audit.json)):

- **Calendar/recency features are removable with no effect.** Dropping every
  calendar/recency feature changes AUPRC by **−0.0009** — statistically nothing. The
  ranking ceiling is therefore **not a cohort artifact** hiding in date features.
- **The signal is distributed, not a concentrated leak.** A model restricted to only
  the top-5 features scores **AUROC 0.826**, not ~0.99 — so no single column (or tiny
  cluster) is memorising the label; the performance comes from many moderate features
  combining.
- **A forward temporal split is infeasible by construction.** All normals are labelled
  `Oct25`; all mules are `Sep25 / Nov25 / Dec25`. **No month contains both classes**,
  so a train-past / test-future split cannot be built on this data — a property of the
  dataset assembly, reported honestly rather than worked around silently.

**Every structural choice is measured.** Resampling ablation
([`reports/07`](reports/)): the winner was **`spw_only`** (AUPRC 0.9133), beating
`both`, `smote_only`, and `none`; the old **double-correction (`both`)** — SMOTE *and*
class weighting — was second-worst, so SMOTE was **dropped**. Feature selection
([`reports/08`](reports/)): keeping **all** features wins (0.918) over a K=100 cut
(0.873). Hyperparameters ([`reports/09`](reports/)): **`max_depth=3`** (down from 5)
measured **AUPRC 0.9301 ± 0.0028** vs 0.9133 at default across 5 seeds.

### Real inference path on genuinely new accounts
A fittable `Preprocessor` now **persists all cleaning state** (learned medians, encoder
categories, dropped-column lists), so [`src/score_new.py`](src/score_new.py) can score
**genuinely new raw accounts** — not just re-score the training frame. A review-caught
one-hot median-fill bug was fixed: **absent dummy columns now fill 0.0** instead of a
learned median.

### Honest by construction
- **AUPRC is the headline metric**, not accuracy — accuracy is meaningless at 0.89%
  prevalence (predicting "all normal" scores 99.11%).
- **Graph stages self-skip.** The provided data is an account-level feature matrix
  with no counterparty/who-paid-whom columns, so PageRank / label propagation
  cannot be built honestly. The pipeline **logs a skip** rather than inventing an
  edge list ([`reports/04_graph_report.json`](reports/04_graph_report.json)).
- **Honest threshold.** The precision-first cutoff is read off the PR curve; if the
  90% precision target were unreachable it falls back to the best-F1 point and
  flags `precision_target_met: false`. (On this data it *is* met.)
- **Calibration checked** via a reliability curve, so the 0–1000 score is trustworthy.

> **Provenance of the numbers.** The headline metrics (precision, recall, F1,
> AUPRC, AUROC, band counts, per-model table) are read verbatim from the JSON in
> [`reports/`](reports/) — you can open the files and check. A few diagnostic
> figures cited in this section are **derived**, not stored in those reports, and
> are reproducible on the cleaned data with a few lines of scikit-learn: the
> single-feature separation ceiling (max univariate AUROC ≈ 0.83, feature
> `F3811`), the `F3888` open-date vintage AUROC (≈ 0.53), and the Isolation-Forest
> out-of-fold AUROC (≈ 0.26) that justified disabling it. The 91.3% figure is the
> **submission PDF's own aspirational target**, quoted for comparison — not a
> MuleGuard output.

---

## 3. Prerequisites

| Requirement | macOS / Linux | Windows |
|---|---|---|
| **Python 3.11 or 3.12** (not 3.13/3.14 — ML wheels lag) | `brew install python@3.12` | [python.org installer](https://www.python.org/downloads/) — tick *"Add python.exe to PATH"* |
| **OpenMP runtime** (for XGBoost/LightGBM) | `brew install libomp` | Bundled in the wheels + [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (usually already present) |
| **git** | preinstalled | [git-scm.com](https://git-scm.com/download/win) |

> **Why the OpenMP note matters:** on macOS, LightGBM/XGBoost crash at import
> without `libomp`. On Windows the wheels ship their own OpenMP, so there is
> nothing extra to install in the normal case — if you ever see a
> `VCOMP140.DLL missing` error, install the VC++ Redistributable linked above.

---

## 4. Setup

### macOS / Linux

```bash
cd muleguard
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_env.py
```

### Windows (PowerShell)

```powershell
cd muleguard
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_env.py
```

> If PowerShell blocks the activate script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> (or use `.venv\Scripts\activate.bat` from `cmd.exe` instead).

`verify_env.py` prints every library + version and ends with **`ALL IMPORTS OK`**.

**Verified stack:** Python 3.12.13 · scikit-learn 1.9 · xgboost 3.3 · lightgbm 4.6 ·
imbalanced-learn 0.14 · shap 0.52 · networkx 3.6 · pandas 2.3 · numpy 2.1.

---

## 5. Run

1. **Add the dataset.** Place the hackathon file at:

   ```
   muleguard/data/DataSet.csv
   ```

   Expected shape ≈ `(9082, 3925)` — 9,082 accounts, target column `F3924`, and a
   leading unnamed row-id column (handled automatically; see §6).

2. **Run the whole pipeline** (same command on every OS, from the project root):

   ```bash
   python src/pipeline.py
   ```

   Runtime is **≈ 5 minutes** from scratch on a modern laptop; the only required
   input is `data/DataSet.csv`. All intermediate files are regenerated.

3. **Read the results.** A summary table prints to the terminal with the exact
   metric names used in the submission, and full artefacts land in
   [`reports/`](reports/) (see §7).

### No dataset yet? Prove the pipeline works on synthetic data

```bash
python src/make_synthetic.py     # writes a fake data/DataSet.csv (right shape)
python src/pipeline.py           # runs every stage end-to-end
```

The synthetic data has a **deliberately weak** signal and a planted `F3912` leak,
so it validates the *plumbing* — not submittable numbers. Delete
`data/DataSet.csv` and drop in the real file for honest results.

---

## 6. How it works — the pipeline

Run by `src/pipeline.py`, which executes each stage in order and prints the final
summary. `src/config.py` is the single source of truth for every path, threshold,
and hyper-parameter.

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_clean.py` | Drop the leading `Unnamed: 0` row-id column; **encode categorical columns** including the **categorical leak guard** that drops `F2230` before one-hot (see below); drop columns missing > 50%; **correlation leak scan** (removes `F3912`); de-duplicate near-identical (collinear) columns; median-impute. |
| 2–3 | `02_features.py` | Add 7 generic, leakage-safe row-profile features (row mean/std/max/min/skew/range + nonzero-fraction); **auto-detect** whether counterparty/edge columns exist for a graph. |
| 4–5 | `03_train.py` | Cost-sensitive, calibrated model (XGBoost by default; optional LightGBM / Isolation Forest via config); SMOTE-Tomek **inside each CV fold**; isotonic calibration; 5-fold stratified CV; precision-first threshold selection. |
| 6 | `04_graph.py` | NetworkX label propagation from confirmed-mule seeds — **only if edge data exists** (it does not here, so it self-skips). |
| 7–8 | `05_score_explain.py` | 0–1000 risk score with LOW/MEDIUM/HIGH bands + **SHAP** top-feature attribution + per-account **Account Risk Report cards**. |
| — | `plots.py` | Precision–Recall, ROC, and calibration-reliability curves. |

Run selected stages with the CLI — `python src/pipeline.py --list`,
`--only 3 5`, or `--from-stage 3`. With no arguments it runs all stages.

### Categorical recovery (Stage 1)

The columns are anonymised (`F1`…`F3924`), but 8 of them are genuine **strings**
that naïve numeric coercion would silently destroy — the richest semantic signal
in the data. MuleGuard encodes them with generic, honest, no-guessing rules:

- **Low-cardinality (≤ 30 distinct) → one-hot.** Recovers account type (`F3886`),
  recency bucket (`F3889`), status (`F3890`), occupation (`F3891`), gender
  (`F3892`), and retail/corporate segment (`F3893`) — **40 dummy columns**.
- **High-cardinality dates → numeric "vintage".** The account open-date (`F3888`)
  becomes days-before-latest. (Verified clean: vintage-vs-target AUROC ≈ 0.53.)
- **Perfect-separation categoricals → dropped as leaks.** This is what removes the
  `F2230` month leak.

### Model composition — measured, not assumed

The default model is **XGBoost, isotonic-calibrated**. The pipeline also supports
LightGBM and Isolation Forest with a logistic stacking meta-learner, but both are
**disabled by default** because — measured across 3 CV seeds — they *lower* the
recall-at-precision-≥-0.90 that matters here (see §1). Isolation Forest scores an
out-of-fold AUROC of ≈ 0.26 (worse than random); LightGBM's AUPRC (~0.79) is well
below XGBoost's (~0.91). Flip `USE_LGBM` / `USE_ISO_FOREST` in `config.py` to
re-enable them for a future dataset where they help. When more than one engine is
active they are combined by a per-fold logistic meta-learner (leakage-safe: fitted
on training-row base predictions only).

### Account Risk Report cards (Stage 7/8)

Beyond the aggregate metrics, the pipeline emits a formatted **Account Risk
Report** for each of the highest-risk accounts
([`reports/account_risk_reports.txt`](reports/) + a JSON twin), showing the 0–1000
risk score, band, recommended action (freeze / monitor / none), and the top
per-account **SHAP risk drivers** in plain language (e.g. *"occupation = student
(+0.19)"*). Anonymised `F####` features are reported by their code name rather
than given invented meaning.

---

## 7. Outputs

Everything lands in [`reports/`](reports/) and `data/`:

| File | Contents |
|---|---|
| `reports/03_metrics.json` | The measured metrics in §1 (per-model + ensemble, both operating points). |
| `reports/01_clean_report.json` | Full cleaning audit: dropped id/sparse/collinear/leak columns, categorical encoding map, top target correlations. |
| `reports/02_features_report.json` | Added features + graph-edge detection result. |
| `reports/04_graph_report.json` | Graph stage status (SKIPPED, with reason). |
| `reports/05_scoring_report.json` | Risk-band distribution + HIGH-band precision. |
| `reports/05_shap_top_features.json` | Top 15 features by mean absolute SHAP value. |
| `reports/05_account_risk_reports.json` | Per-account risk cards (score, band, action, SHAP drivers) — machine-readable. |
| `reports/account_risk_reports.txt` | The same risk cards, human-readable. |
| `reports/pr_curve.png` · `roc_curve.png` · `calibration_curve.png` | Diagnostic plots. |
| `data/oof_predictions.csv` | Per-account out-of-fold probabilities. |
| `data/risk_scores.csv` | Per-account 0–1000 risk score + band + true label. |
| `models/muleguard_models.joblib` | Final model(s) refit on all data, for scoring new accounts. |

---

## 8. Configuration

All tunables live in [`src/config.py`](src/config.py). The most relevant:

| Setting | Default | Meaning |
|---|---|---|
| `TARGET_COL` | `F3924` | Binary label (0 = normal, 1 = mule). |
| `LEAK_CORR_THRESHOLD` | `0.90` | Drop any feature with \|corr\| above this vs the target. |
| `CATEGORICAL_LEAK_ERROR_REDUCTION` | `0.98` | Drop a categorical whose category this-perfectly determines the label. |
| `CATEGORICAL_MAX_CARDINALITY` | `30` | Distinct values at/below which a string column is one-hot encoded. |
| `MISSING_DROP_FRAC` | `0.50` | Drop columns missing more than this fraction. |
| `COLLINEAR_CORR_THRESHOLD` | `0.98` | De-duplicate near-identical columns. |
| `USE_ISO_FOREST` | `False` | Add Isolation Forest to the model set (see §6). |
| `USE_LGBM` | `False` | Add LightGBM + logistic stacking (see §6). |
| `N_FOLDS` / `RANDOM_STATE` | `5` / `42` | Cross-validation folds and seed (reproducibility). |
| `PRECISION_TARGET` | `0.90` | Precision the auto-freeze cutoff is tuned to hold. |
| `BAND_LOW_MAX` / `BAND_MEDIUM_MAX` | `400` / `750` | 0–1000 risk-band cutoffs. |
| `N_RISK_REPORT_CARDS` | `20` | Number of top-risk Account Risk Report cards to emit. |

---

## 9. Folder layout

```
muleguard/
├── README.md              # this file
├── SETUP.md               # detailed install / troubleshooting
├── STATUS.md              # single honest source of truth on project state
├── requirements.txt       # pinned dependencies
├── verify_env.py          # environment checker
├── data/                  # DataSet.csv goes here (git-ignored)
├── src/
│   ├── config.py          # single source of truth (paths, params, thresholds)
│   ├── utils.py           # shared helpers
│   ├── 01_clean.py        # Stage 1
│   ├── 02_features.py     # Stage 2/3
│   ├── 03_train.py        # Stage 4/5
│   ├── 04_graph.py        # Stage 6
│   ├── 05_score_explain.py# Stage 7/8
│   ├── plots.py           # PR/ROC/calibration plots
│   ├── pipeline.py        # orchestrator — run this (supports --only / --from-stage)
│   └── make_synthetic.py  # optional synthetic-data smoke test
├── tests/                 # pytest suite locking the honesty guarantees
├── models/                # trained models (.joblib, git-ignored)
└── reports/               # metrics.json + plots + risk-report cards (git-ignored)
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: libomp.dylib` (macOS) | `brew install libomp`, reinstall the package |
| `VCOMP140.DLL missing` (Windows) | Install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `pip install` compiles from source / fails | You're on Python 3.13/3.14 — use 3.11 or 3.12 in a venv |
| `python3.12: command not found` | macOS: `brew install python@3.12`. Windows: use `py -3.12` |
| PowerShell won't run activate script | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `Dataset not found at .../data/DataSet.csv` | Place the hackathon CSV there (see §5), or run `python src/make_synthetic.py` first |
| `parquet unavailable; wrote ...csv` | Not an error — `pyarrow` isn't installed, so intermediates fall back to CSV. The pipeline handles this transparently. |
| Imports fail after install | Confirm the venv is active (`which python` / `where python` points inside `.venv`) |

---

## 11. Status & reproducibility

- **Run end-to-end on the real `DataSet.csv`**, from a clean slate (only the CSV on
  disk), with **zero errors**. The numbers in §1 are reproduced by that run and are
  deterministic (`RANDOM_STATE = 42`).
- **Environment verified:** `verify_env.py` → `ALL IMPORTS OK`.
- **Saved model reloads and scores** correctly (inference path tested).
- **Automated tests lock the guarantees.** Run `python -m pytest tests/ -q` — the
  suite proves the categorical leak guard fires, the correlation scan catches
  near-perfect features, categorical encoding / date-vintage / ID-drop behave, and
  the threshold selector never collapses to all-negative when the target is
  unreachable.
- See [`STATUS.md`](STATUS.md) for the full ledger of what is proven, the remaining
  human step (paste these measured numbers into the submission PDF and soften its
  graph-feature claims, since the data is an account-level matrix with no edges),
  and optional polish.

> **Re-run anytime:** `python src/pipeline.py`. To reproduce from absolute scratch,
> delete everything in `data/` except `DataSet.csv` and the intermediates will be
> rebuilt.

---

## 12. 👩‍⚖️ Live demo walkthrough (for judges)

A rehearsed, low-risk script for presenting MuleGuard live. The theme to sell is
**honesty**: this pipeline finds fraud *and* refuses to cheat — it caught two
target leaks that would have faked a ~100% score. Total time ≈ 8 minutes, or ≈ 90
seconds if you use the "pre-baked" shortcut.

### 0. The golden rule — do a dry run before you present

The full pipeline takes ≈ 5 minutes. **You do not want 5 minutes of silence in
front of judges.** Choose one of two modes:

- **Safe mode (recommended):** run the full pipeline **once before** you walk in
  so `reports/` is already populated, then *talk over* the pre-computed artefacts.
  You can still trigger a live run and let it finish while you narrate.
- **Live mode:** run it live only if you have ≥ 6 spare minutes and a machine you
  trust. Kick it off early (step 3) and keep talking while it runs.

Before the session, confirm the environment is healthy:

```bash
cd muleguard
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python verify_env.py                 # must end with: ALL IMPORTS OK
python -m pytest tests/ -q           # must end with: 19 passed
ls data/DataSet.csv                  # the dataset must be present
```

### 1. Run-of-show (what to do, and what to say)

| # | Do this | Say this (≈) |
|---|---|---|
| 1 | `python verify_env.py` | "Fully reproducible environment — one command verifies the whole ML stack." |
| 2 | `python -m pytest tests/ -q` → **19 passed** | "We have automated tests that *lock in* our anti-cheating guarantees — including that the nested-CV operating point is genuinely out-of-sample — so they can never silently break." |
| 3 | `python src/pipeline.py` (let it run; narrate while it trains) | "One command runs all 6 stages end-to-end on the real 9,082-account dataset." |
| 4 | **The money shot** — the leak reveal (see §12.2) | "Before we show numbers, here's why you can *trust* them. We found a hidden trap in the data — and then *measured* that we're not relying on it." |
| 5 | Show the results table (terminal summary or `reports/03_metrics.json`) | "AUPRC 0.93 and, out-of-sample via nested CV, 93% precision at 86% recall — ~70 of 81 mules — with an auto-freeze tier that is 69 of 71 correct. The threshold is never chosen on the data it scores." |
| 6 | Open `reports/account_risk_reports.txt` | "And it's explainable: every flagged account comes with plain-language reasons an investigator can act on." |
| 7 | Show `reports/pr_curve.png` | "Here's the full precision–recall trade-off the operating point is read from." |

### 2. The money shot — proving the pipeline is honest

This is the moment that wins trust. Run these two commands live; they take seconds.

**(a) Show the trap in the raw data** — the `F2230` sampling-month leak:

```bash
python -c "import pandas as pd; d=pd.read_csv('data/DataSet.csv', usecols=['F2230','F3924']); print(pd.crosstab(d['F2230'], d['F3924']))"
```

Expected output — *every* normal account is month `Oct25`, *every* mule is a
different month:

```
F3924     0   1
F2230
Dec25     0  10
Nov25     0  23
Oct25  9001   0
Sep25     0  48
```

> **Say:** "The sampling month perfectly separates the classes. A naive model
> would 'learn' this and report ~100% accuracy — then fail completely in
> production, where every live account shares the current month. It's a trap."

**(b) Show that MuleGuard removed it** — the trap columns are gone from the
cleaned matrix the model actually trains on:

```bash
python -c "import pandas as pd; c=pd.read_parquet('data/clean.parquet'); print('F2230 (month leak) present:', 'F2230' in c.columns); print('F3912 (corr leak) present:', 'F3912' in c.columns)"
```

Expected: **both `False`.**

> **Say:** "Our leak-guard drops it automatically — along with a second
> correlation-based leak, `F3912`. And we didn't just *drop* it — `reports/06_leak_audit.json`
> *measures* that removing every calendar feature changes AUPRC by −0.0009, so our
> 0.93 is earned on genuine behavioural signal, not a data artefact. That's a number
> that survives your questions."

### 3. The 90-second shortcut (if time is tight)

Skip the live training run. The artefacts from a prior run are already in
`reports/`:

```bash
python -m pytest tests/ -q                       # 19 passed  (trust)
# leak reveal — the two commands in §12.2         (honesty)
cat reports/account_risk_reports.txt | head -20  # explainable output
python -c "import json; d=json.load(open('reports/03_metrics.json')); r=d['headline_ranking']; o=d['honest_operating_point']; print(f\"AUPRC {r['auprc']:.3f}  |  honest out-of-sample: P {o['precision']['mean']:.3f}  R {o['recall']['mean']:.3f}\")"
```

### 4. Anticipated questions — rehearsed answers

- **"How do we know you're not overfitting 81 mules?"** — The operating point is
  from **repeated nested cross-validation**: the calibrator and threshold are fit on
  inner out-of-fold data and applied only to an untouched outer-test fold, so the
  threshold is never chosen on the data it scores. We report the honest number
  (P 0.934 / R 0.861) *and* its across-repeat spread and Wilson CI. A top-5-feature
  model scores only AUROC 0.826 (`reports/06_leak_audit.json`), so the 0.99 full
  AUROC is genuine multivariate signal, not memorisation.
- **"Isn't your 0.99 just the sampling-month cohort?"** — Measured, not argued:
  removing every calendar/recency feature changes AUPRC by **−0.0009**
  (`reports/06_leak_audit.json`). The result does not ride on the month cohort.
- **"Why not the graph / network features from your proposal?"** — The provided
  data is an account-level feature matrix with **no who-paid-whom columns**, so a
  money-flow graph can't be built honestly. The pipeline detects this and **skips
  the graph stage rather than fabricating edges** (`reports/04_graph_report.json`).
- **"Why 93% precision and not higher?"** — We report the honest out-of-sample point
  (P 0.934 / R 0.861). We *can* dial precision higher at the cost of recall; the
  target is tunable in `config.py` (`PRECISION_TARGET`), and a high-recall operating
  point already feeds an analyst queue. We also keep an in-sample "optimistic ceiling"
  (P 0.912 / R 0.901) in the metrics **explicitly labelled as not the headline**.
- **"Can it explain a decision to a regulator?"** — Yes: `account_risk_reports.txt`
  gives per-account SHAP reasons, and the 0–1000 score is isotonic-**calibrated**
  (`reports/calibration_curve.png`).

### 5. If something breaks on stage

- Pipeline errors mid-demo → fall back to the **§12.3 shortcut** and talk over the
  pre-computed `reports/`. Everything a judge needs is already on disk.
- `libomp` / `VCOMP140` import crash → see [§10 Troubleshooting](#10-troubleshooting);
  this is why step 0's dry run matters.
- Slowness → the dataset is 116 MB; the first CSV read alone is ~45 s. Narrate the
  honesty story (§12.2) while Stage 1 runs — it needs no artefacts.
