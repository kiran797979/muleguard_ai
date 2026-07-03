# MuleGuard AI

**Network-aware money-mule account detection — precision-first.**

A reproducible ML pipeline for the PSB Cybersecurity, Fraud & AI Hackathon 2026
(Bank of India × IIT Hyderabad). It detects mule accounts in an extremely
imbalanced dataset (**81 mules in 9,082 accounts ≈ 0.89%**) and produces
**honest, measured 5-fold cross-validation metrics** — not design targets — that
replace the placeholder figures in the submission document.

> **Cross-platform:** runs on macOS, Windows, and Linux. The Python code uses
> `pathlib` throughout (no hardcoded paths) and UTF-8 everywhere. Only the
> *setup commands* differ per OS — both are given below.

---

## 1. Prerequisites

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

## 2. Setup

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

---

## 3. Run

1. **Add the dataset.** Place the hackathon file at:

   ```
   muleguard/data/DataSet.csv
   ```

2. **Run the whole pipeline** (same command on every OS, from the project root):

   ```bash
   python src/pipeline.py
   ```

3. **Read the results.** A summary table prints to the terminal with the exact
   metric names used in the submission, and full artefacts land in `reports/`:
   `03_metrics.json`, `pr_curve.png`, `roc_curve.png`, `calibration_curve.png`,
   `05_shap_top_features.json`.

### No dataset yet? Prove the pipeline works on synthetic data

```bash
python src/make_synthetic.py     # writes a fake data/DataSet.csv (right shape)
python src/pipeline.py           # runs all 8 stages end-to-end
```

The synthetic data has a **deliberately weak** signal and a planted `F3912`
leak, so it validates the *plumbing* — not submittable numbers. Delete
`data/DataSet.csv` and drop in the real file for honest results.

---

## 4. What it does (8 stages)

| Stage | Script | Purpose |
|---|---|---|
| 1 | `src/01_clean.py` | Drop >50%-missing columns, **auto-detect & remove the F3912 target leak** (correlation scan), de-duplicate collinear rolling-window copies, impute |
| 2–3 | `src/02_features.py` | Generic behavioural fingerprint features + **auto-detect** whether graph/edge data exists |
| 4–5 | `src/03_train.py` | Cost-sensitive, calibrated ensemble (Isolation Forest + XGBoost + LightGBM), SMOTE-Tomek **inside each CV fold**, logistic stacking, isotonic calibration, 5-fold stratified CV |
| 6 | `src/04_graph.py` | NetworkX label propagation from confirmed-mule seeds (**only if edge data exists**) |
| 7–8 | `src/05_score_explain.py` | 0–1000 risk score with LOW/MEDIUM/HIGH bands + **SHAP** reason lists |
| — | `src/pipeline.py` | Runs every stage in order and prints the final summary |

---

## 5. Honesty guarantees (why the numbers survive scrutiny)

- **Leak removed first.** `F3912` (~0.97 corr with target) is caught by a
  correlation scan and dropped before any model sees it.
- **No leakage across folds.** All SMOTE-Tomek resampling is fitted *inside*
  each of the 5 stratified folds — never on validation rows or the full data.
- **AUPRC is the headline metric**, not accuracy (accuracy is meaningless at
  0.89% prevalence — predicting "all normal" scores 99.11%).
- **Nothing fabricated.** The graph stages activate only if the data actually
  contains counterparty account IDs. When it doesn't, the pipeline **logs a
  skip** instead of inventing an edge list.
- **Honest threshold.** The precision-first cutoff is read off the precision–
  recall curve; if the 90% target is unreachable it falls back to the best-F1
  point and **flags that in the report** (`precision_target_met: false`).
- **Calibration checked** via a reliability curve, so the 0–1000 score is trustworthy.

> ⚠️ **Data-shape caveat.** The provided data is an *account-level feature
> matrix* (9,082 × 3,924). A true money-flow graph (PageRank, label propagation)
> needs a *transaction ledger* of who-paid-whom. If the columns don't encode
> counterparty IDs, the graph stages are skipped automatically and the tabular
> model carries the result — stated openly rather than faked.

---

## 6. Folder layout

```
muleguard/
├── README.md              # this file
├── SETUP.md               # detailed install / troubleshooting
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
│   ├── pipeline.py        # orchestrator — run this
│   └── make_synthetic.py  # optional synthetic-data smoke test
├── models/                # trained models (.joblib, git-ignored)
└── reports/               # metrics.json + plots (git-ignored)
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: libomp.dylib` (macOS) | `brew install libomp`, reinstall the package |
| `VCOMP140.DLL missing` (Windows) | Install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `pip install` compiles from source / fails | You're on Python 3.13/3.14 — use 3.11 or 3.12 in a venv |
| `python3.12: command not found` | macOS: `brew install python@3.12`. Windows: use `py -3.12` |
| PowerShell won't run activate script | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| Imports fail after install | Confirm the venv is active (`which python` / `where python` points inside `.venv`) |

---

## 8. Status

The pipeline has been **run end-to-end successfully** on Python 3.12 with the
full stack (xgboost 3.3, lightgbm 4.6, shap 0.52, imbalanced-learn 0.14,
networkx 3.6). All 8 stages execute, the F3912 leak is auto-removed, and reports
are generated. Feed it the real `DataSet.csv` to produce the final submission
numbers.
