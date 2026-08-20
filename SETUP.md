# MuleGuard AI — Setup & Installation Guide

Detailed install, troubleshooting, and environment notes. For the quick path see
[README.md](./README.md); for what the numbers mean see
[reports/00_INTEGRITY.md](./reports/00_INTEGRITY.md).

This project runs on **macOS, Windows, and Linux** from the same source tree. The
Python code uses `pathlib` throughout and forces UTF-8 on every file it writes,
so only the *setup commands* differ per platform.

---

## 0. TL;DR

### Windows (PowerShell)

```powershell
cd $HOME\OneDrive\Desktop\muleguard
.\run.ps1                 # creates .venv-win, installs, runs the pipeline
```

If PowerShell blocks the script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned      # once
# or, without changing policy:
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### macOS / Linux

```bash
cd ~/Desktop/muleguard
chmod +x run.sh           # once
./run.sh                  # creates .venv, installs, runs the pipeline
```

Both launchers are idempotent: they build the environment only if it is missing,
then run `src/pipeline.py`.

---

## 1. Two virtual environments, on purpose

A virtual environment is **not portable across operating systems** — macOS uses
`.venv/bin/` with Mach-O binaries, Windows uses `.venv\Scripts\` with PE
binaries. Because this project folder syncs between machines through OneDrive,
both would otherwise land in the same `.venv` and destroy each other.

| Platform | Directory | Interpreter |
|---|---|---|
| macOS / Linux | `.venv/` | `.venv/bin/python` |
| Windows | `.venv-win/` | `.venv-win\Scripts\python.exe` |

Both are git-ignored. Deleting one has no effect on the other.

> **OneDrive note:** a virtual environment inside a synced folder is slow and
> generates constant sync churn. If you can, keep the project outside OneDrive.
> If you cannot, exclude `.venv` and `.venv-win` from sync in the OneDrive
> settings — the pipeline works either way, it is purely a speed matter.

---

## 2. Python version

Use **Python 3.11 or 3.12**. XGBoost, LightGBM, SHAP and imbalanced-learn do not
reliably ship prebuilt wheels for 3.13/3.14, so installing there either fails
with compiler errors or silently falls back to slow source builds.

| Platform | Install |
|---|---|
| Windows | [python.org installer](https://www.python.org/downloads/) — tick **"Add python.exe to PATH"**. `run.ps1` then finds it via `py -3.12`. |
| macOS | `brew install python@3.12` |
| Linux | `sudo apt install python3.12 python3.12-venv` |

Your system Python is left untouched; everything is installed into the venv.

---

## 3. System prerequisites

| Platform | What | Why |
|---|---|---|
| macOS | `brew install libomp` | **Required.** LightGBM and XGBoost link against the OpenMP runtime and crash at import without it (`Library not loaded: libomp.dylib`). The single most common Mac failure. |
| Windows | usually nothing | The wheels bundle their own OpenMP. If you see `VCOMP140.DLL missing`, install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
| Linux | `sudo apt install libgomp1` | Same OpenMP requirement. |

---

## 4. Dependencies

Pinned in [`requirements.txt`](./requirements.txt).

| Package | Role |
|---|---|
| `pandas`, `numpy`, `scipy` | Loading, cleaning, numerics |
| `scikit-learn` | Stacking, isotonic calibration, stratified CV, PR metrics |
| `xgboost`, `lightgbm` | The two gradient-boosted base models |
| `imbalanced-learn` | Optional resampling (the pipeline defaults to cost-sensitive weighting) |
| `shap` | Feature attribution for the audit trail |
| `networkx` | Graph propagation — only used if edge data exists |
| `matplotlib`, `seaborn` | PR/ROC/calibration plots |
| `openpyxl` | Reads `Description.xlsx`, the data dictionary |
| `pyarrow` | Parquet intermediates (falls back to CSV if absent) |
| `joblib`, `tqdm` | Model persistence, progress |
| `fastapi`, `uvicorn` | The command-center API and static host (§9) |

Verify with:

```powershell
.\run.ps1 -Verify        # Windows
./run.sh verify          # macOS / Linux
```

Expect a table of versions ending in **`ALL IMPORTS OK`**.

---

## 5. Input files

The pipeline searches several locations, so you can leave files where they land:

| File | Looked for in | Override |
|---|---|---|
| Dataset | `data/DataSet.csv`, then the project root (`DataSet.csv`, `DataSet (2).csv`) | `MULEGUARD_DATA` |
| Data dictionary | `data/Description.xlsx`, then the project root | `MULEGUARD_DICT` |

```powershell
$env:MULEGUARD_DATA = "D:\somewhere\DataSet.csv"; .\run.ps1
```

```bash
MULEGUARD_DATA=/path/to/DataSet.csv ./run.sh
```

Expected dataset shape: **9,082 rows x 3,925 columns** (3,924 features `F1..F3924`
plus an unnamed row-index column). The target is `F3924` (`FRAUD_TGT`).

The dictionary is optional — the pipeline runs without it — but strongly
recommended: it is what turns leak detection, feature engineering, and SHAP
reason lists from opaque `F` codes into named banking variables.

Sanity check once the file is in place:

```bash
python -c "import pandas as pd; d=pd.read_csv('DataSet (2).csv'); print(d.shape, d['F3924'].sum())"
# expect roughly: (9082, 3925) 81
```

---

## 6. A dataset handed over live

Two ways. **Use the browser one** when someone is watching.

### From the browser (easiest)

```powershell
.
un.ps1 -Serve
```

Open `http://127.0.0.1:8000`, go to **00 UPLOAD DATASET**, drag the file in, press
RUN PIPELINE. Stages tick past with a progress bar and a live log, then the
integrity verdict, what it worked out about the schema, and the measured result
appear on the page. **Measured: 101 seconds** for a 5,000 x 62 file.

Accepted: `.csv .tsv .txt .xlsx .xls .parquet`, up to 500 MB. Each upload gets
its own folder under `runs/`, so the submission's own results are never touched.

Upload safety, since this accepts a file from outside: extensions are
allow-listed, the client's filename is discarded and a new one generated (so
path traversal has nothing to traverse), the stream is written in chunks against
a hard size cap, and the pipeline runs as a separate process that can be killed.
Nothing uploaded is ever executed, only parsed.

### From the command line

```powershell
.
un.ps1 -Dataset D:	heirs.csv
```

```bash
./run.sh dataset /path/theirs.csv
```



This is the path to use when somebody puts a file in front of you and asks you
to run it.

```powershell
.
un.ps1 -Dataset D:	heirs.csv
```

```bash
./run.sh dataset /path/theirs.csv
```

**Measured: 83 seconds** on an unseen 6,000 x 82 file, start to finished report.
It writes everything under `runs/<name>/`, so your own submission results are
never touched, and it prints a per-stage timing table at the end.

Add the target column only if auto-detection cannot find it:
`.
un.ps1 -Dataset D:	heirs.csv -Target is_fraud`

### What demo mode changes, and what it refuses to change

| Changed | Not changed |
|---|---|
| 1 CV repeat instead of 3 | Every leak defence |
| 2 inner folds instead of 3 | Partition-column detection |
| 200 trees instead of 400 | The nested structure |
| 3-fold integrity audit instead of 5 | Calibration and the fitted threshold |
| Feature ablation skipped | Explainability |
| Inputs over 60,000 rows sampled, all positives kept | |

The tree count was measured rather than guessed. Over identical folds, 400 trees
gives AUPRC 0.8502 +/- 0.0686 in 182s; 200 trees gives 0.8443 +/- 0.0600 in 107s.
That is a cost of 0.006 AUPRC, about a twelfth of the standard deviation, for a
1.7x speedup. Cutting the feature count to 150 was also tried and **rejected**: it
dropped AUPRC to 0.8156, which is a real loss rather than a rounding one.

Pass `-Full` (PowerShell) or `MULEGUARD_FULL=1` (bash) for full precision.

---

## 7. Running on a different dataset

Nothing is hardcoded to the hackathon file. Point the pipeline at another CSV
and it works out the target, the identifiers, the categoricals, the leak
columns and the partition columns for itself.

```bash
MULEGUARD_DATA=/path/to/other.csv \
MULEGUARD_WORKDIR=runs/other \
./run.sh
```

```powershell
$env:MULEGUARD_DATA = "D:\other.csv"
$env:MULEGUARD_WORKDIR = "runs\other"
.\run.ps1
```

| Variable | Effect | Default |
|---|---|---|
| `MULEGUARD_DATA` | dataset path | `data/DataSet.csv`, then the project root |
| `MULEGUARD_DICT` | data dictionary, `.xlsx` or `.csv` | `data/Description.xlsx`, then root |
| `MULEGUARD_TARGET` | name the target explicitly | auto-detected |
| `MULEGUARD_WORKDIR` | where `data/ models/ reports/` are written | the project root |
| `MULEGUARD_REPEATS` | CV repeats | 3 |

`MULEGUARD_WORKDIR` is what lets several datasets coexist in one checkout — each
run keeps its own artefacts, so processing a second file does not overwrite the
first one's results or trained model.

**If target detection fails** it stops rather than guessing, and tells you what
it saw. Set `MULEGUARD_TARGET=<column>` and re-run.

### Demonstrate the graph stage

```bash
python src/graph_demo.py
```

Builds a synthetic transaction ledger with planted mule rings and runs the real
`propagate()` from `04_graph.py` against it, writing
`reports/demo_graph/`. The supplied dataset has no counterparty column so the
stage disables itself there; this shows the same code working when edges exist.
Outputs are namespaced and watermarked as a demonstration, never a result.

### Run the tests

```bash
python tests/test_schema.py     # 12 tests: target, identifiers, partitions, leaks
python tests/test_roles.py      # 13 tests: role parsing and cross-schema matching
```

Both run standalone with no test dependency, and under pytest if you prefer.

### Verify dataset-independence

```bash
python src/make_synthetic.py --schema alien
MULEGUARD_DATA=data/alien_dataset.csv MULEGUARD_DICT=/nonexistent \
MULEGUARD_WORKDIR=runs/alien MULEGUARD_REPEATS=1 python src/pipeline.py
```

The alien file shares no column name with the hackathon data and ships no
dictionary. Check `runs/alien/reports/01_clean_report.json` → `schema` and
`partition_audit` to see what was discovered.

---

## 8. Running individual stages

```powershell
.\run.ps1 -Stage 01_clean.py
```

```bash
./run.sh stage 01_clean.py
```

Stages write intermediates to `data/` and must run in order the first time:
`06_integrity.py` → `01_clean.py` → `02_features.py` → `03_train.py` →
`04_graph.py` → `05_score_explain.py` → `09_rules.py` →
`08_feature_ablation.py` → `plots.py`.

`09_rules.py` and `08_feature_ablation.py` only read artefacts the earlier stages
wrote, so they are cheap to re-run on their own while you are exploring.

Training is the slow one. Control the cost with:

```powershell
$env:MULEGUARD_REPEATS = "1"     # 1 repeat x 5 folds — quick check (~3 min)
$env:MULEGUARD_REPEATS = "3"     # default — mean +/- std across 15 folds (~10 min)
```

Whichever value is used is recorded in `reports/03_metrics.json` under
`reproducibility`, together with the resolved fold counts and every library
version, so a published number can always be traced back to the run that
produced it.

---

## 9. The command center (web UI)

Once the pipeline has run at least once:

```powershell
.\run.ps1 -Serve                 # http://127.0.0.1:8000
.\run.ps1 -Serve -Port 8080      # different port
```

```bash
./run.sh serve
./run.sh serve 8080
```

Needs `fastapi` and `uvicorn`, both in `requirements.txt`. **No Node, no npm, no
build step** — the frontend is plain HTML/CSS/JS in `app/static/`, so edits take
effect on a browser refresh.

It opens on the **Command Center** landing page: the thesis, a live system
readout, the four numbers the argument rests on, and three routes into the rest.
Sixteen sections in total, numbered 00 to 15 down the left rail.

The UI reads pipeline artefacts, so it degrades honestly rather than silently:

| Symptom | Cause | Fix |
|---|---|---|
| Status bar shows `DEGRADED` | Some artefacts are missing | The Pipeline section lists exactly which, and which stage makes each |
| A panel shows `ARTEFACT_MISSING` | That stage has not run | Run the named stage, then refresh |
| Status bar shows `API DOWN` | The server is not running | Start it with the command above |
| Explainability is empty | `shap` was not installed when Stage 4/5 ran | `pip install shap`, re-run `03_train.py` |

`GET /api/health` returns the same information as JSON, and `/api/docs` gives an
interactive API browser.

The server binds to `127.0.0.1` on purpose. It unpickles a model file and serves
account-level risk data, so it is a local analyst tool — do not expose it on a
shared network without putting real authentication in front of it.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: @rpath/libomp.dylib` (macOS) | `brew install libomp`, then reinstall lightgbm/xgboost |
| `VCOMP140.DLL missing` (Windows) | Install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `run.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `pip install` compiles from source / fails | You are on Python 3.13+. Use 3.11 or 3.12. |
| `py: command not found` (Windows) | Reinstall Python with "Add python.exe to PATH" ticked |
| `Dataset not found at ...` | Put the CSV in `data/`, or set `MULEGUARD_DATA` |
| `Data dictionary not found` (warning only) | Put `Description.xlsx` in `data/`, or set `MULEGUARD_DICT` |
| Mojibake / `UnicodeEncodeError` in the console | The launchers set `PYTHONIOENCODING=utf-8`; set it manually if invoking Python directly |
| Imports fail after install | Confirm the venv is active — `where python` / `which python` should point inside `.venv-win` / `.venv` |
| Training takes too long | Lower `MULEGUARD_REPEATS`, or `TOP_K_FEATURES` in `src/config.py` |

---

## 11. Environment reset

```powershell
Remove-Item -Recurse -Force .venv-win ; .\run.ps1 -Setup      # Windows
```

```bash
rm -rf .venv && ./run.sh setup                                 # macOS / Linux
```

To also clear generated artefacts:

```bash
rm -f data/*.parquet data/*.csv models/*.joblib reports/*.json reports/*.png
```

(Leave `data/DataSet.csv` in place, and keep `reports/00_INTEGRITY.md` — it is
regenerated by the pipeline but is worth reading before you delete anything.)
