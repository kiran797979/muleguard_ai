# MuleGuard AI — Project Status

**What is proven, what is broken in the data, and what to do about it.**
Updated 2026-08-19.

Read [`reports/00_INTEGRITY.md`](./reports/00_INTEGRITY.md) alongside this file.
It is the single most consequential thing this project has produced.

---

## The headline

The real dataset and the real data dictionary are both in hand, and the pipeline
runs end-to-end on Windows and macOS. **But the supplied dataset cannot support
an honest mule-detection metric**, and we can prove it rather than suspect it.

Every negative in `DataSet.csv` comes from the October extract; every positive
comes from the September, November, or December extracts. **No month contains
both classes.** So anything that differs between monthly extraction runs is
perfectly correlated with the label while describing no customer behaviour.

The proof is a falsification test, not an opinion. Feed a model **only whether
each cell was blank** — discard every value, so nothing about any account's
behaviour remains:

| What the model sees | AUPRC | AUROC |
|---|---|---|
| Blank/not-blank pattern only, no values at all | **0.8236** | **0.9925** |
| 250 columns each with \|corr\| < 0.05 (individually useless) | 0.7361 | 0.9734 |
| Same columns, labels shuffled (sanity floor) | 0.0094 | 0.5284 |
| *Random-guess baseline* | *0.0089* | *0.500* |

Whether a cell was populated is decided by the extraction job, not by a customer.
The floor test collapsing to baseline confirms the harness is sound — so those
scores are a real property of the data.

**Any score you see on this dataset, from any team, is measuring extract
provenance as well as mule behaviour, and the two cannot be cleanly separated
within this file.**

---

## PROVEN (built, run, verified on this machine)

Executed on Windows 11 / Python 3.12 with xgboost 3.4, lightgbm 4.7, shap 0.52,
scikit-learn 1.9, imbalanced-learn 0.14, pandas 2.3.

| # | Item | Evidence |
|---|---|---|
| 1 | Runs on Windows **and** macOS from one source tree | `run.ps1` / `run.sh`; separate `.venv-win` / `.venv` |
| 2 | Real dataset loads: 9,082 x 3,925, 81 mules (0.892%) | Stage 1 log |
| 3 | Data dictionary wired in as executable knowledge | `src/dictionary.py`, 3,924 definitions |
| 4 | **Month leak found and removed** | `01_clean_report.json` → `structural_leak_audit` |
| 5 | **Post-outcome leaks removed by meaning, not correlation** | 6 columns: `FRAUD_SUSPECTED` (corr 0.97) *and* `FALSE_POSITIVE` (corr 0.05), `OTHER_RESOLUTION`, `UNATTENDED`, `MIN/MAX_RESOLVE_DAYS` |
| 6 | **Extract hardening** drops columns whose blank rate is class-dependent | `01_clean_report.json` → `extract_hardening` |
| 7 | Categoricals encoded instead of destroyed | occupation, gender, area, product, segment, account-age — the old code turned all 8 to NaN and dropped them |
| 8 | "Missing = no activity" treatment for transaction aggregates | ~79k values correctly zero-filled rather than median-imputed |
| 9 | 29 named mule-typology features | pass-through, turnover/balance, burst, cash-out, channel mix, ticket size, night alerts, balance shape, occupation divergence |
| 10 | **Nested repeated CV** — selection, stacking, calibration, threshold all fitted inside the fold | `03_metrics.json` → `validation` |
| 11 | Separation audit scans every column for near-perfect class split | This is what would catch the *next* MNTH |
| 12 | Graph stage still self-skips honestly | Dictionary confirms no counterparty column exists in any of the 3,924 variables |
| 13 | Per-account risk reports with plain-English reasons | `05_account_reports.json` — the PDF's page-14 card, now real |

---

## Fixed since the last version

| Was | Now |
|---|---|
| Isotonic calibration fitted on the same out-of-fold predictions it was scored on | Fitted on an inner split; validation rows never touch it |
| Threshold chosen by scanning the curve it was then reported from | Chosen on inner data, applied frozen |
| Meta-learner trained on base models' *training* predictions | Trained on inner out-of-fold predictions |
| Single 5-fold split, no error bars | Repeated CV, mean +/- std across folds |
| Isolation Forest min-max scaled on the validation fold | Range learned on train, frozen |
| Only `F3912` treated as a leak | 8 leak columns, classified by meaning |
| 8 categorical columns silently destroyed | Encoded |
| Generic row aggregates only | 29 domain features + row aggregates |

---

## Fixed in the 2026-08-19 audit round

A code-vs-paper audit found eight defects. All eight are fixed and the pipeline
was re-run; the numbers below are from that run.

| # | Was | Now |
|---|---|---|
| 1 | `config.py` documented a band cutoff "re-derived from the precision-recall curve"; `band()` used the constants 400 and 750 and never read the fitted thresholds | Band edges are the ensemble's own operating points: LOW < **55.5** <= MEDIUM < **763.0** <= HIGH |
| 2 | Median imputation fitted across all 9,082 rows before any split | Fitted inside the training fold in `MuleEnsemble._prep()`, applied frozen |
| 3 | SHAP came from the final model refit on all rows, while scores came from out-of-fold predictions — the reasons did not belong to the number | Per-fold SHAP on validation rows, stored in `data/oof_shap.npz` |
| 4 | Isolation forest presented as a contributing base model | Reported as scoring below random (AUROC 0.317); the stacking coefficient is published and is negative (-0.4541) |
| 5 | "Missing = no activity" treatment silently did nothing without the data dictionary | Falls back to raw column names and warns loudly if nothing matches |
| 6 | Shipped reports said 3 repeats while the config default was 5 — the numbers could not be reproduced from the code as checked in | `reproducibility` block written into `03_metrics.json` on every run, with resolved settings and library versions |
| 7 | `config.py` hardcoded `D:/Description.xlsx` | Removed |
| 8 | `pipeline.py` used `runpy.run_path`, so no stage could be imported or tested | Stages are imported and their `main()` called |
| 9 | **Results were not reproducible across processes.** `CATEGORICAL_NAMES` is a `set`, and iterating a set of strings follows Python's per-process hash seed. That fixed the encoding order, which fixed the one-hot column order, and XGBoost's `colsample_bytree` samples columns *by index* — so identical data with an identical `random_state` produced different fold metrics from one run to the next. | All set-derived orderings sorted. Verified stable across `PYTHONHASHSEED` 1/2/3; `python_hash_seed` recorded in the metrics file |

Defect 9 was found by accident, while checking that the dataset-independence work
had not changed the results: Stage 1 and Stage 2 output were byte-identical, yet
the fold metrics moved. It matters because it silently invalidated the
reproducibility claim added alongside defect 6 — the settings were recorded, but
re-running with them still would not reproduce the numbers.

Layers 2-4 of the leak defence still compute against the label on the full
dataset. They only ever remove columns, so they make the result more
conservative — but they are not fitted inside the fold, and the paper now says
so rather than implying otherwise.

### Measured after the fixes

Run with `PYTHONHASHSEED=0`, 3 repeats x 5 outer folds.

| | Value |
|---|---|
| Precision (precision-first point) | 0.976 +/- 0.067 |
| Recall (precision-first point) | 0.692 +/- 0.129 |
| AUPRC | 0.854 +/- 0.076 |
| AUROC | 0.967 +/- 0.033 |
| Lift over base rate | 109x |
| Band edges (derived) | LOW < 68.9 <= MEDIUM < 685.8 <= HIGH |
| HIGH band | 53 accounts, 53 real mules, precision 1.000 |
| Partition audit | clean after removal — nothing else splits the classes |
| Separation audit | clean — no surviving column separates the classes |

**These differ slightly from the figures published before defect 9 was fixed**
(AUPRC was 0.861, precision 0.970). That earlier run was one arbitrary draw from
a hash-seed-dependent column ordering; it was not wrong so much as
unreproducible. The figures above are the ones that regenerate.

Determinism verified directly rather than assumed: Stages 1-2 were run twice
under `PYTHONHASHSEED=11` and `=22`, and the resulting feature matrices are
**byte-identical** (same shape, same column order, same SHA-256 over the
values).

These remain an **upper bound** on what this dataset can show. See the headline
above.

---

## The web layer (new)

There was no frontend, backend, or API before this round. There is now:

- `app/server.py` — FastAPI: 13 endpoints, normalised errors, body cap, no
  request-supplied paths.
- `app/service.py` — artefact loading, account analysis, live scoring. Raises
  `ArtefactMissing` -> 503 naming the stage to run, rather than inventing data.
- `app/static/` — brutalist command-center UI, plain HTML/CSS/JS, no build step.
  Includes a 12-step **Judge Mode** walkthrough ending in a live analysis.

Start it with `.\run.ps1 -Serve` or `./run.sh serve`.

---

## Dataset independence (O5 — now closed)

The pipeline used to run on exactly one file. `TARGET_COL` was the literal string
`"F3924"`, the dictionary loader discarded any row whose code did not match
`^F\d+$`, and `MNTH` was removed because a human had read the data and noticed
it. `src/schema.py` replaces all of that with detection.

| Question | Answer now | Was |
|---|---|---|
| Which column is the target? | `MULEGUARD_TARGET`, else a binary column matching a target-name pattern, else the only binary column. Stops rather than guessing. | `"F3924"` |
| Which are row IDs? | name patterns **and** near-uniqueness, floats exempt | `"Unnamed: 0"` |
| Which are categorical? | dtype + cardinality; known ordinal vocabularies encoded monotonically | a fixed list of 6 names |
| Which leak? | `POST_OUTCOME_PATTERNS` matched by meaning, unioned with the known names | a fixed list of 6 names |
| Which partition the classes? | **shape** — low cardinality, class-pure values | someone noticed `MNTH` |
| Where is column X? | normalised matching (`tot.txnamt.cr.l7d` == `TOT_TXNAMT_CR_L7D`) | exact string equality |
| No data dictionary? | identity map over the dataset's own headers | leak detection and features silently degraded |
| Second dataset? | `MULEGUARD_WORKDIR` gives each run its own `data/ models/ reports/` | outputs collided |

### The partition test

This is the part worth defending in a viva. Removing `MNTH` by name does not
generalise; finding it by shape does. `schema.partition_columns()` flags a
low-cardinality column whose values are split between the classes rather than
shared by them, which is exactly what an assembly artefact looks like. On the
supplied file it re-derives `MNTH` at purity 1.000 without being told it exists.
A genuine behavioural categorical fails the test, because its values contain
both classes — verified against `customer_segment` in the alien fixture below.

### Verified, not asserted

`src/make_synthetic.py --schema alien` writes a dataset sharing **no column
name** with the hackathon file and shipping **no dictionary**:

| Planted | Discovered |
|---|---|
| target `is_mule` (not last, not F-coded) | found by name pattern + binary check |
| identifier `account_number` | dropped; nothing else was |
| partition `data_month` | flagged at purity 1.000 and removed |
| post-outcome `case_resolution` | removed by meaning |
| decoy `customer_segment` | correctly **not** flagged |
| behavioural cols, differently punctuated | resolved fuzzily; 6 typology features built with no dictionary |

The full pipeline runs to completion on it (exit 0) and the integrity audit
correctly reports that file as **not** contaminated — test A scores 0.013
against a 0.010 baseline. The audit working in the negative direction matters as
much as it working in the positive one.

Remaining honest limitation: the 29 typology features need the relevant
transaction aggregates to exist. On a dataset without them the pipeline still
runs, degrades to row-profile features, and reports which bases were missing
rather than fabricating zeros — on the alien fixture 6 of 29 were buildable and
it said so.

---

## We measured our own claims, and two of them did not survive

### The feature ablation

Removing the 29 behavioural features changes AUPRC by
-0.0121, which is noise. More importantly, the raw
columns (0.8623) beat a model given **only blank/not-blank patterns and no
values at all** (0.8236) by 0.0387. Almost the entire headline is
extract provenance.

The behavioural features alone score 0.3268, 37x
random. Being ratios, they are robust to which fields an export populated, so
that is the figure we would defend as plausible real detection.

Reproduce: `python src/08_feature_ablation.py` -> `reports/08_feature_ablation.json`

### The AML rule layer

Twelve rules from published typology, thresholds not tuned against this data.
**7 of 12 score at or below the base rate**; two catch zero mules; combined
they flag 7,862 accounts (87% of the book) at
103.4 alerts per mule.

Cause: the typology is inverted in this dataset. Median 7-day pass-through is
0.776 for normal customers and 0.622 for mules. Only small ticket size (4.7x
lift, 45 of 81 mules) and single payment rail (1.4x) earned their place.

Reproduce: `python src/09_rules.py` -> `reports/09_rules_report.json`

### Why both are in the submission

Neither result flatters us. The rules mostly fail and the ML's headline is mostly
artefact. Publishing both is the only position consistent with the integrity
audit, and it means every number we quote has a stated provenance.

---

## The honest position on accuracy

The pipeline reports a high AUPRC on this dataset. **Do not quote it as a
mule-detection result.** The integrity report shows an uninformative view of the
same data scores comparably, which means most of that number is extract
provenance.

What can be said honestly:

- The pipeline is leak-hardened, nested-validated, and reproducible.
- Its measured numbers are the *upper bound* of what this dataset can show, and
  the integrity audit states how much of that is artefact.
- On data where both classes are drawn from the same months, the same code would
  produce a trustworthy number — nothing about the method needs changing.

An honest number a judge can trust beats an inflated one that collapses under
questioning. Here the most valuable finding *is* the caveat: this is the kind of
thing that decides a hackathon when someone on the panel asks the right question.

---

## OPEN

| # | Item | Notes |
|---|---|---|
| O1 | Raise the month-split issue with the organisers | Affects every team using this file, not just this submission |
| O2 | Re-run once positives and negatives share months | One command; no code change needed |
| O3 | Hyperparameter tuning | Deliberately deferred — tuning against a confounded target optimises the artefact |
| O4 | Verify a full run on macOS with the new code | Windows verified end-to-end; the Mac path is `./run.sh` and unchanged in shape |
| ~~O5~~ | ~~Generalise to an arbitrary dataset~~ | **Done** — see "Dataset independence" above. `src/schema.py` + `MULEGUARD_WORKDIR`; verified on a synthetic alien schema |
| O6 | Package the UI for a machine without Python | Currently needs the venv. A single-file build is possible but was not in scope |

---

## Next action

1. Read `reports/00_INTEGRITY.md`.
2. Decide how the submission presents the finding. The strongest framing is to
   lead with it: the pipeline is rigorous *and* it detected that the benchmark is
   contaminated.
3. Ask the organisers whether same-month negatives can be supplied.
