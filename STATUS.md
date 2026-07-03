# MuleGuard AI — Project Status

**What is proven, what is pending, and what stands between the code and a
submission-ready result.** Updated 2026-07-02.

Read this first. It is the single honest source of truth for where the project
stands — no inflated claims.

---

## ✅ PROVEN (built, run, and verified on this machine)

These are not aspirations — they were executed end-to-end on macOS / Python 3.12
with the full ML stack (xgboost 3.3, lightgbm 4.6, shap 0.52,
imbalanced-learn 0.14, networkx 3.6, scikit-learn 1.9).

| # | Item | Evidence |
|---|---|---|
| 1 | **Environment installs cleanly** | `verify_env.py` → `ALL IMPORTS OK` (13/13 libraries) |
| 2 | **All 8 stages run end-to-end** | `python src/pipeline.py` completed; summary table printed |
| 3 | **F3912 target leak auto-detected & removed** | Stage 1 log: `Removed 1 suspected leak column(s): ['F3912']` |
| 4 | **Sparse + collinear columns dropped** | Stage 1 log: dropped 40 sparse + 6 near-duplicate cols |
| 5 | **5-fold stratified CV with in-fold SMOTE-Tomek** | Stage 4/5 log: 5 folds, resampling fitted inside each fold |
| 6 | **Calibrated 3-model ensemble** (IsoForest + XGB + LGBM + isotonic) | `reports/03_metrics.json` written with per-model + ensemble metrics |
| 7 | **Graph stage self-skips honestly on account-level data** | Stage 6 log: `Graph stage SKIPPED (no edge data)` — not fabricated |
| 8 | **SHAP explanations generate** | `reports/05_shap_top_features.json` written |
| 9 | **PR / ROC / calibration plots generate** | `pr_curve.png`, `roc_curve.png`, `calibration_curve.png` |
| 10 | **A real bug was found by running (not just compiling) and fixed** | Threshold fallback no longer returns all-zeros; falls back to best-F1 and flags `precision_target_met: false` |
| 11 | **Cross-platform code audit passed** | No hardcoded POSIX paths; UTF-8 forced on all file I/O; no shell calls |

> Proof was on **synthetic** data (right shape, deliberately weak signal + planted
> F3912 leak). That validates the *plumbing*, not submission numbers. See
> `src/make_synthetic.py`.

---

## 🔴 BLOCKER — clears everything below

| # | Item | Owner | Notes |
|---|---|---|---|
| B1 | **`data/DataSet.csv` is not present** | **You** | The real hackathon file (9,082 × 3,924). Nothing real can be measured without it. Everything in "PENDING" is fast once this lands. |

Drop it at: `muleguard/data/DataSet.csv`

---

## 🟡 PENDING — needs the dataset, then ~10 min of work

| # | Item | Depends on | Effort |
|---|---|---|---|
| P1 | Run pipeline on real data → measured precision/recall/F1/AUPRC/AUROC | B1 | 1 command |
| P2 | Replace PDF placeholder numbers (pages 3 & 13) with measured results | P1 | mechanical |
| P3 | Sanity-check real results (leak caught? numbers plausible, not too-good?) | P1 | review |

---

## 🟠 GAPS — where the PDF claims more than the code currently does

Honest inconsistencies a judge could spot. Resolution often depends on what the
real data actually contains.

| # | PDF claims | Code currently does | Resolution |
|---|---|---|---|
| G1 | Graph features / PageRank / label propagation is the headline differentiator | Auto-skips — an account-level matrix has no who-paid-whom edges | If real data has counterparty ID columns → it runs automatically. If not → soften the PDF's graph claims. **Unknown until B1.** |
| G2 | Named behavioural features (money velocity, occupation–income divergence, named F-columns) | Uses generic row-profile features (columns are anonymized) | If Bank of India provides a data dictionary, build the real named features. |
| G3 | Formatted "Account Risk Report" card (page 14) | Produces SHAP top-features JSON, not the formatted per-account report | Add a report-card formatter (nice-to-have for demo). |

---

## ⚪ OPTIONAL POLISH (only if time allows)

| # | Item | Value |
|---|---|---|
| O1 | Hyperparameter tuning to push toward 90%+ precision | Currently sensible defaults, not tuned |
| O2 | Confirm a real run on an actual Windows machine | Code is Windows-safe by audit, unverified on hardware |
| O3 | Add the report-card formatter (G3) | Demo polish |

---

## The one honest caveat on accuracy/precision

This is a **rigorous, leak-free, reproducible** pipeline. It is **not** guaranteed
to hit the PDF's 91.3% precision — those were the document's own aspirational
*targets*, not measured results. Real numbers depend on real signal in the real
data. If they come out lower, the pipeline reports that honestly (it even flags
when the 90% precision target is unreachable) rather than inflating it. An honest
number a judge can trust beats an inflated one that collapses under questioning.

---

## Next action

**Do you have `DataSet.csv`?**
- **Yes** → put it in `muleguard/data/` and run `python src/pipeline.py`. We then do P1–P3.
- **No** → getting it is the whole critical path. Nothing else matters until then.
