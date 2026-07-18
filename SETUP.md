# MuleGuard AI — Setup & Installation Guide

Everything you need to install and configure to run the MuleGuard AI mule-account
detection pipeline and produce **real, measured** precision/recall/AUPRC numbers.

> **Goal of this doc:** get a clean, reproducible environment on your Mac so that
> running the pipeline on `DataSet.csv` yields honest 5-fold cross-validation
> results — the numbers that replace the design-target placeholders in the
> submission PDF.

---

## 0. TL;DR (copy-paste, in order)

```bash
# 1. Install system prerequisites (Homebrew must already be installed)
brew install python@3.11 libomp git

# 2. Go to the project folder
cd ~/Desktop/muleguard

# 3. Create + activate an isolated Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Upgrade pip and install all Python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 5. Verify everything imported correctly
python verify_env.py

# 6. Drop DataSet.csv into ~/Desktop/muleguard/data/ then run the pipeline
#    (pipeline scripts are built in the next step of the project)
```

---

## 1. Why a specific Python version

The submission stack (XGBoost, LightGBM, SHAP, imbalanced-learn, NetworkX) is
**not guaranteed to have prebuilt wheels for Python 3.14**, which is what your
machine currently has (`Python 3.14.5`). Installing on 3.14 can fail with
compiler errors or force slow source builds.

**Use Python 3.11 (recommended) or 3.12.** This matches the PDF's stated
"Python 3.11" stack and has full, stable wheel support for every library below.

| What you have | What we'll use | Why |
|---|---|---|
| Python 3.14.5 (system) | Python 3.11 in a `venv` | Guaranteed ML wheel compatibility |

Your system Python 3.14 is left untouched — the virtual environment is fully
isolated.

---

## 2. System prerequisites (macOS / Apple Silicon)

Install these once with [Homebrew](https://brew.sh):

```bash
brew install python@3.11 libomp git
```

| Package | Why it's required |
|---|---|
| `python@3.11` | The interpreter we build the environment on |
| `libomp` | **OpenMP runtime** — LightGBM *and* XGBoost need it on macOS, or they crash at import/train time with `Library not loaded: libomp.dylib` |
| `git` | Version control (already present on your machine) |

> **Apple Silicon note:** `libomp` is the single most common reason LightGBM
> fails on a Mac. Do not skip it.

Confirm Homebrew is installed first:

```bash
which brew || echo "Install Homebrew from https://brew.sh first"
```

---

## 3. Create and activate the virtual environment

```bash
cd ~/Desktop/muleguard
python3.11 -m venv .venv
source .venv/bin/activate
```

After activation your prompt shows `(.venv)`. Confirm the right Python:

```bash
python --version      # should print Python 3.11.x
which python          # should point inside .../muleguard/.venv/bin/python
```

To leave the environment later: `deactivate`.
To re-enter it in a new terminal: `cd ~/Desktop/muleguard && source .venv/bin/activate`.

---

## 4. Python dependencies

All packages are pinned in [`requirements.txt`](./requirements.txt). Install with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### What each package does (maps to the PDF tool stack)

| Package | Role in MuleGuard | PDF section |
|---|---|---|
| `pandas` | Data loading, cleaning, feature computation | Stage 1–2 |
| `numpy` | Numerical / matrix operations | throughout |
| `scipy` | Stats, sparse matrices (dependency of sklearn) | throughout |
| `scikit-learn` | Stacking meta-learner, isotonic calibration, stratified K-fold, PR-curve metrics | Stage 5, 7 |
| `xgboost` | Primary gradient-boosting classifier | Stage 5 |
| `lightgbm` | Leaf-wise boosting classifier | Stage 5 |
| `imbalanced-learn` | SMOTE-Tomek in-fold resampling for 0.89% prevalence | Stage 4 |
| `shap` | Feature-attribution explanations (RBI/PMLA audit trail) | Stage 8 |
| `networkx` | Directed transaction graph, PageRank, centrality, label propagation | Stage 3, 6 |
| `matplotlib` | PR/ROC curves, calibration plots | Section 9 |
| `seaborn` | Feature-importance and network visualisations | Section 9 |
| `jupyterlab` | (Optional) interactive notebooks for EDA | — |
| `joblib` | Save/load trained models | — |
| `tqdm` | Progress bars for long steps | — |

---

## 5. Verify the installation

Run the checker script (created below):

```bash
python verify_env.py
```

Expected output — a table of every library with its version and a final
`ALL IMPORTS OK` line. If anything fails, jump to **Troubleshooting**.

---

## 6. Project folder layout

Target structure once the pipeline is built:

```
muleguard/
├── SETUP.md                 # this file
├── requirements.txt         # pinned dependencies
├── verify_env.py            # import + version checker
├── data/
│   └── DataSet.csv          # <-- YOU must place the hackathon dataset here
├── src/
│   ├── 01_clean.py          # Stage 1: cleaning + F3912 leak removal
│   ├── 02_features.py       # Stage 2–3: behavioural + graph features
│   ├── 03_train.py          # Stage 4–5: imbalance handling + ensemble
│   ├── 04_graph.py          # Stage 6: label propagation
│   ├── 05_score_explain.py  # Stage 7–8: risk score + SHAP
│   └── pipeline.py          # runs all stages end-to-end
├── models/                  # saved trained models (.joblib)
├── reports/                 # metrics, PR curves, SHAP plots
└── .venv/                   # virtual environment (not committed)
```

Create the empty folders now:

```bash
cd ~/Desktop/muleguard
mkdir -p data src models reports
```

---

## 7. The dataset (required to get real numbers)

- The pipeline expects the hackathon file at: **`~/Desktop/muleguard/data/DataSet.csv`**
- Expected shape per the PDF: **9,082 rows × 3,924 feature columns + target `F3924`**.
- ⚠️ It is **not currently in the folder.** Everything installs and the code can
  be written without it, but no measured metric can be produced until it's placed
  in `data/`.

Quick sanity check once you add it:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/DataSet.csv'); print(df.shape); print('F3924' in df.columns)"
# Expect roughly: (9082, 3925)  True
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: @rpath/libomp.dylib` (LightGBM/XGBoost) | `brew install libomp`, then reinstall the package |
| `pip install` compiles from source / fails on Python 3.14 | You skipped the venv — use `python3.11 -m venv .venv` and reinstall |
| `command not found: python3.11` | `brew install python@3.11` then reopen the terminal |
| `shap` install is slow | Normal — it builds some extensions; let it finish |
| SSL / certificate errors during pip | `python -m pip install --upgrade certifi` |
| Everything installed but imports fail | Confirm the venv is active: `which python` should be inside `.venv` |
| Mac blocks `brew`/scripts | Ensure Homebrew is installed and on PATH (`which brew`) |

---

## 9. What comes after setup

Once `verify_env.py` passes **and** `DataSet.csv` is in `data/`, the next phase is
building and running the 8-stage pipeline to produce the honest metrics:

1. **Clean** — drop >50%-missing columns, remove the **F3912 target leak**.
2. **Engineer** — 8 behavioural + 6 graph features.
3. **Train** — cost-sensitive ensemble (Isolation Forest + XGBoost + LightGBM),
   SMOTE-Tomek *inside* each of 5 stratified CV folds.
4. **Calibrate** — isotonic calibration of the stacked probability.
5. **Propagate** — NetworkX label propagation from the 81 confirmed mules.
6. **Score + explain** — 0–1000 risk score with SHAP reason lists.
7. **Report** — precision, recall, F1, AUPRC, AUROC, PR curve → `reports/`.

These measured 5-fold CV numbers then **replace the bold placeholder figures**
on pages 3 and 13 of the submission PDF.

---

## 10. One-command environment reset

If the environment ever gets into a bad state, nuke and rebuild:

```bash
cd ~/Desktop/muleguard
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_env.py
```
