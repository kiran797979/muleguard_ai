# MuleGuard AI

**Network-aware money-mule account detection — precision-first, leak-hardened,
and honest about what its own benchmark can prove.**

A reproducible ML pipeline for the PSB Cybersecurity, Fraud & AI Hackathon 2026
(Bank of India × IIT Hyderabad). It detects mule accounts in an extremely
imbalanced dataset (**81 mules in 9,082 accounts = 0.892%**) using the supplied
data dictionary as executable domain knowledge, and produces measured nested
cross-validation metrics — not design targets.

> **Runs on Windows, macOS, and Linux** from one source tree. `pathlib`
> throughout, UTF-8 forced on all I/O, no shell calls, no hardcoded paths.

> **👩‍⚖️ Presenting to judges?** Jump to **[§12 — Live demo walkthrough](#12--live-demo-walkthrough-for-judges)**
> for a rehearsed, minute-by-minute script (setup, run, and the exact commands
> that showcase the leak-free honesty story).

---

## ⚠️ Read this first

**The supplied dataset cannot support an honest mule-detection score, and this
pipeline proves it rather than papering over it.**

Every negative comes from the **October** extract; every positive comes from the
**September, November, or December** extracts. No month contains both classes.
So any difference between monthly extraction runs correlates perfectly with the
label while describing nothing about any customer.

Give a model **only whether each cell was blank** — throw away every value, so no
account behaviour survives at all:

| What the model sees | AUPRC | AUROC | vs random |
|---|---|---|---|
| **Blank/not-blank pattern only, no values** | **0.8236** | **0.9925** | 92× |
| 250 columns each with \|corr\| < 0.05 (individually useless) | 0.7361 | 0.9734 | 83× |
| Same columns, labels shuffled *(sanity floor)* | 0.0094 | 0.5284 | 1× |
| *Random-guess baseline* | *0.0089* | *0.500* | 1× |

Whether a cell is populated is decided by the extraction job, not by a customer.
The shuffled-label floor collapsing to baseline proves the harness is sound — so
these are real properties of the data.

Full analysis, reproducible via `src/06_integrity.py`:
**[`reports/00_INTEGRITY.md`](./reports/00_INTEGRITY.md)**.

The pipeline removes this artefact wherever it can be identified (see §10). What
remains unidentifiable — a genuine behavioural feature that *also* drifts month
to month — cannot be separated within this file by any method. Fixing it needs
negatives and positives sampled from the **same months**; that is a
data-collection change, and it applies to every team working from this file.

---

## 1. Quick start

### Windows (PowerShell)

```powershell
cd $HOME\OneDrive\Desktop\muleguard
.\run.ps1
```

### macOS / Linux

```bash
cd ~/Desktop/muleguard
chmod +x run.sh && ./run.sh
```

**Given a dataset live?** Start the UI and drag the file in at
**00 UPLOAD DATASET** (101 seconds for a 5,000 row file), or from the terminal:

```powershell
.
un.ps1 -Dataset D:	heirs.csv       # writes to runs	heirs\, never touches your own results
```

```bash
./run.sh dataset /path/theirs.csv
```

It works out the target, the identifiers, the leak columns and the partition
columns for itself. See [SETUP.md §6](./SETUP.md) for what demo mode trades.

Then start the command center:

```powershell
.\run.ps1 -Serve          # Windows   -> http://127.0.0.1:8000
```

```bash
./run.sh serve            # macOS/Linux
```

Each launcher creates its own virtual environment if missing, installs pinned
dependencies, and runs all stages. Windows uses `.venv-win\`, macOS/Linux uses
`.venv\` — a venv is not portable across operating systems, and this folder syncs
between machines through OneDrive, so they are kept separate on purpose.

Prerequisites: **Python 3.11 or 3.12** (ML wheels lag on 3.13+), and on macOS
`brew install libomp` (LightGBM/XGBoost crash at import without it). Details and
troubleshooting in [SETUP.md](./SETUP.md).

### Input files

Both are found automatically in `data/` or the project root; override with
`MULEGUARD_DATA` / `MULEGUARD_DICT`.

| File | What it is |
|---|---|
| `DataSet.csv` | 9,082 × 3,925. Target `F3924` (`FRAUD_TGT`). |
| `Description.xlsx` | The data dictionary — 3,924 `F`-code → banking-variable definitions. **Optional**: without it the pipeline uses the dataset's own column names, which is the right answer for any file whose headers are already readable. |

---

## 2. It runs on any dataset

Nothing in the pipeline is configured for the hackathon file. `src/schema.py`
works out what it is looking at, and every stage asks it rather than assuming.

| Question | How it is answered | Was |
|---|---|---|
| Which column is the target? | `MULEGUARD_TARGET`, else a binary column whose name matches a target pattern, else the only binary column | `TARGET_COL = "F3924"` |
| Which columns are row IDs? | Name patterns **and** near-uniqueness (floats exempt — every continuous measurement is near-unique) | `"Unnamed: 0"` |
| Which are categorical? | dtype + cardinality; ordinal vocabularies get a monotonic encoding | a fixed list of 6 names |
| Which leak? | `POST_OUTCOME_PATTERNS` matched by meaning | a fixed list of 6 names |
| Which columns partition the classes? | **Shape**: low cardinality + class-pure values | someone noticed `MNTH` |
| Where is `TOT_TXNAMT_CR_L7D`? | normalised match — `tot.txnamt.cr.l7d`, `Tot Txnamt Cr L7D` all resolve | exact string equality |

### The partition test is the interesting one

`MNTH` was removed from this dataset because a human read the file and spotted
it. That does not generalise. `schema.partition_columns()` finds the same column
by its **shape** — low cardinality, values split between the classes rather than
shared by them — so the defence fires on a file nobody has inspected. On the
supplied data it re-derives `MNTH` (purity 1.000) unaided. A genuine behavioural
categorical like occupation fails the test, because its values contain both
classes.

### Prove it

```bash
python src/make_synthetic.py --schema alien          # a dataset sharing no column name
MULEGUARD_DATA=data/alien_dataset.csv \
MULEGUARD_DICT=/nonexistent \
MULEGUARD_WORKDIR=runs/alien \
python src/pipeline.py
```

That file has a target called `is_mule` (not last, not `F`-coded), an
`account_number` identifier, a `data_month` partition column, a
`case_resolution` post-outcome leak, readable-but-differently-punctuated
behavioural columns, **and no data dictionary at all**. The pipeline finds all
four traps, builds features by fuzzy name resolution, and — correctly — reports
that file as *not* contaminated, because it isn't.

### Environment overrides

| Variable | Effect |
|---|---|
| `MULEGUARD_DATA` | dataset path |
| `MULEGUARD_DICT` | data dictionary (`.xlsx` or `.csv`), optional |
| `MULEGUARD_TARGET` | name the target explicitly, skipping detection |
| `MULEGUARD_WORKDIR` | write `data/ models/ reports/` under here, so several datasets coexist |
| `MULEGUARD_REPEATS` | CV repeats |

---

## 3. The dictionary is the differentiator

Without `Description.xlsx`, this data is 3,924 opaque numbers. With it, every
leak decision, feature, and SHAP reason is a named banking variable.
`src/dictionary.py` turns the spreadsheet into structured knowledge:

- `F3891` → `CUST_OCCP` → *"Occupation code of customer"*
- Name decomposition: **stat** (`R`/`RA`/`D`/`DA`/`AVG`/`MAX`/`MIN`/`TOT`) ×
  **channel** (CASH/UPI/ATM/APB/…) × **direction** (CR/DB) × **window**
  (7D/14D/31D/7-14/7-31/14-31)
- The bank's own **18 shortlisted variables** (`Bank_Finalized_Variables`) — a
  free domain-expert feature selection, used as a prior
- **Leak classification by meaning**, which is what makes §10 possible

---

## 4. The command center

A local FastAPI service serves the trained model and every pipeline report to a
brutalist single-page UI. **No Node, no npm, no build step** — the frontend is
plain HTML/CSS/JS, so there is nothing to compile five minutes before a demo.

```
.\run.ps1 -Serve     ./run.sh serve     -> http://127.0.0.1:8000
```

| # | Section | What it shows |
|---|---|---|
| 00 | **Command Center** | The landing page. Thesis, live system readout, the four numbers that carry the argument, and three routes into the rest. |
| 01 | **Judge Mode** | The whole project in 12 steps, ~90 seconds, arrow-key driven, ending in a live account analysis. |
| 02 | **Upload Dataset** | Drag in a CSV, TSV, Excel or parquet file and run the full pipeline against it. |
| 03–04 | Overview / Dataset | Scale, cleaning ledger, encoding, missing-value treatment, and how the schema was inferred. |
| 05 | **Integrity audit** | The partition column, the three falsification tests, the verdict and the grounds it rests on. |
| 06 | Leakage defence | All four layers with their evidence, plus the correlation backstop. |
| 07 | Mule features | All 29, grouped by family, each with why it is diagnostic. |
| 08 | Models | Per-model scores, stacking coefficients, both operating points, reproducibility. |
| 09 | Explainability | Global SHAP ranking with real banking variable names. |
| 10 | **Rules & Ablation** | The twelve-rule AML baseline measured against the base rate, and how much of our own score is the artefact. |
| 11 | Risk triage | Band edges and their provenance, band precision and recall. |
| 12 | **Account analysis** | Score, calibrated probability, band, SHAP reasons, evidence against the population, next steps, and the investigator decision panel. |
| 13 | **Operating cost** | Precision@K, false alarms per thousand, review budget, latency, extract drift. |
| 14 | **Audit trail** | Every investigator decision, timestamped and append-only. The retraining set. |
| 15 | Pipeline | Stage flow and artefact health. |

Three rules the UI holds to:

1. **It never invents a number.** If an artefact is missing the panel renders an
   error naming the artefact and the stage that produces it — a demo that shows
   plausible figures when the model failed to load would be the worst possible
   failure for a project whose thesis is "do not trust unverified numbers".
2. **Score and explanation share a provenance.** Benchmark accounts are served
   out-of-fold probabilities and out-of-fold SHAP. The live `POST /api/score`
   path uses the deployment model and says so in its response, so the two are
   never confused.
3. **The integrity verdict travels with everything.** It is on the landing page,
   in the status bar of every section, and embedded in every exported alert. No
   screen shows a score without showing how much to trust it.

API: `GET /api/health`, `/api/overview`, `/api/integrity`, `/api/clean`,
`/api/leakage`, `/api/features`, `/api/models`, `/api/metrics`, `/api/shap`,
`/api/bands`, `/api/accounts`, `/api/account/{idx}`, and `POST /api/score`.
Interactive docs at `/api/docs`.

The server binds to `127.0.0.1` by default. It loads a pickled model, so it is a
local analyst tool and is not hardened for exposure on a shared network. No
request supplies a path, filename, or module name.

---

## 5. What the pipeline measures about itself

Two stages exist purely to test our own claims, and both produced uncomfortable
answers that we publish rather than bury.

### Feature ablation (`src/08_feature_ablation.py`)

How much of our score is the extract artefact rather than detection? Three runs,
identical folds, identical seed:

| Condition | Features | AUPRC | vs random |
|---|---|---|---|
| Everything | 1,506 | 0.850 | 96x |
| **Raw columns only** | 1,477 | **0.862** | 97x |
| *Blank patterns only, no values at all* | *3,772 bits* | *0.824* | *92x* |
| **Behavioural features only** | 29 | **0.327** | 37x |

Read rows 2 and 3 together. The raw columns, with every value present, beat a
model that has **no values at all** by only 0.039.
Almost everything they contribute on this dataset is provenance.

The 29 behavioural features score 0.327 alone,
37x better than random. They are ratios, so they survive a
change in which fields an export populated. That makes them the part least
explained by the artefact, and the number we would actually defend.

### The AML rule layer (`src/09_rules.py`)

Twelve rules from published money-mule typology, thresholds **not tuned** against
this dataset. Tuning them here would fit the confound and fail elsewhere.

- **7 of 12 perform at or below the base rate.**
- Two catch zero mules.
- All twelve combined flag 7,862 accounts, 87% of the book, at 103.4 alerts per mule.

The typology is inverted here: ordinary customers have a median 7-day
pass-through of **0.776**, mules **0.622**. Mules pass through *less* money than
everybody else.

Two rules survived: **small ticket sizes** (4.7x lift, finds 45 of 81 mules) and
**single payment rail** (1.4x). We would keep those two and delete the other ten.

---

## 6. EFRMS and AML platform integration

The pipeline emits alerts, case packs and an audit trail in a **documented,
vendor-neutral schema** whose fields map onto the concepts every AML case
management system uses.

```
reports/exports/alerts.json            full alert payload
reports/exports/alerts.csv             batch exchange format
reports/exports/case_pack_<id>.json    one investigator case pack
reports/exports/integration_contract.json   the field mapping
```

Or over HTTP: `/api/export/alerts` (add `?format=csv`),
`/api/export/casepack/{id}`, `/api/export/contract`, and an OpenAPI 3
specification at `/api/openapi.json`.

| Field | Maps to |
|---|---|
| `alert_id` | Case reference. Deterministic, so a re-run updates rather than duplicates |
| `entity_type` / `entity_ref` | Always ACCOUNT; this scores accounts, not transactions |
| `risk_score` / `risk_probability` | 0-1000 and the calibrated probability behind it |
| `priority` / `risk_band` | Queue ordering, from fitted operating points |
| `scenario_codes` | Typology that fired. **Our taxonomy, not a regulator code list** |
| `reasons` | Ranked SHAP attributions with direction |
| `model_id` / `score_provenance` | Provenance, for model risk management |
| `data_integrity_warning` | Travels with every alert when the source data failed its audit |

### What we claim, and what we do not

**We do not claim certified compatibility with any named platform.** We have not
tested against Oracle FCCM, SAS, NICE Actimize, Clari5, Amlock or anything else,
and we do not have their integration specifications.

What is true: the schema is documented, the mapping is published, JSON and CSV
are both provided, and the API is described by OpenAPI. Wiring this into a
specific EFRMS is a field-mapping exercise, not a rebuild. To do it properly you
would need that platform's spec, a scenario catalogue to map onto, an agreed
entity key, and a test environment.

**On STR filing:** the case pack assembles what an analyst needs to *prepare* a
Suspicious Transaction Report. It does not file one and must not. FIU-IND
submission carries its own schema and its own legal responsibility, and that
decision belongs to a human.

---

## 7. The graph stage, and why it is switched off

Mule detection is usually a network problem, so the absence of a graph is the
first thing anyone asks about.

**This dataset cannot support one.** All 3,924 variables were checked against the
data dictionary: not one names a counterparty. No beneficiary, no payee, no VPA,
no IFSC. Every column aggregates a single account's own activity. Without edges
there is no graph, so `src/04_graph.py` detects the absence, writes the reason to
`reports/04_graph_report.json`, and exits rather than fabricating an edge list.

### Proving the code works anyway

"It would work if the data allowed it" is a claim, so we made it demonstrable:

```bash
python src/graph_demo.py     # -> reports/demo_graph/
```

This builds a synthetic ledger (2,000 accounts, 4,320
transfers, 8 planted rings) and runs the **real**
`propagate()` from `04_graph.py` against it, imported rather than copied. Two
members of each ring are treated as already confirmed; the rest have to be found.

| Stop after | Queue size | Members found | Recall | Precision | Alerts per find |
|---|---|---|---|---|---|
| **hop 1** | **93** | **33** | **92%** | **35%** | **2.8** |
| hop 2 | 442 | 36 | 100% | 8% | 12.3 |
| hop 3 | 1,222 | 36 | 100% | 3% | 33.9 |

**One hop does almost all the work.** It recovers 92% of
the unknown ring members at under three alerts per find. Hops two and three add
3 more members and
1,129 innocent accounts. That collapse is the
honest argument against untuned propagation, and it is why the pipeline consumes
this as a **0.15 blend weight** on the risk score rather than as an alert.

> **This is a capability demonstration, not a result.** The rings were planted by
> the script and then found by the script. It says nothing about mule detection
> on the supplied data. Every output lives under `reports/demo_graph/` and the
> figure is watermarked so it can never be mistaken for a real finding.

---

## 8. Pipeline stages

Run by `src/pipeline.py`, which executes each stage in order and prints the final
summary. `src/config.py` is the single source of truth for every path, threshold,
and hyper-parameter.

| Stage | Script | What it does |
|---|---|---|
| 0 | `06_integrity.py` | **Dataset integrity audit.** Three falsification tests; writes `reports/00_INTEGRITY.md`. Run and read first. |
| 1 | `01_clean.py` | Semantic leak removal, categorical encoding, activity-aware imputation, extract hardening, separation audit |
| 2/3 | `02_features.py` | 29 named mule-typology features + row-profile aggregates + honest graph-feasibility check |
| 4/5 | `03_train.py` | Nested repeated CV: in-fold selection, IsolationForest + XGBoost + LightGBM, stacking, isotonic calibration, threshold |
| 6 | `04_graph.py` | NetworkX label propagation — **only if** counterparty data exists |
| 7/8 | `05_score_explain.py` | 0–1000 risk score, LOW/MEDIUM/HIGH bands, SHAP reasons, per-account risk reports |
| 9 | `09_rules.py` | Deterministic AML rule layer, measured against the base rate, thresholds untuned |
| 10 | `08_feature_ablation.py` | How much of the score survives removing the behavioural features |
| 12 | `integration.py` | Vendor-neutral alert / case-pack export for EFRMS and AML platforms |
| — | `plots.py` | PR / ROC / calibration curves |
| — | `pipeline.py` | Runs everything, prints the summary |

---

## 9. The features encode a mule's actual behaviour

A mule *receives* money and pushes it straight back out, holds almost nothing,
in bursts, through digital rails, often at odd hours, on an account whose owner
profile does not match the volume. Each family measures one clause:

| Family | Feature | Why it is diagnostic |
|---|---|---|
| **Pass-through** | `mg_passthrough_7d/14d/31d` | credit ≈ debit → the account is a conduit, not a wallet |
| **Turnover / balance** | `mg_turnover_over_balance_*` | moves many multiples of what it ever holds |
| **Burst** | `mg_amount_burst_7v31` | weekly rate ≫ monthly rate → sudden activation |
| **Cash-out** | `mg_cash_out_share_7d`, `mg_digital_in_cash_out_7d` | digital in, cash out — the layering handoff |
| **Channel mix** | `mg_channel_hhi_7d`, `mg_channel_active_7d` | single-purpose accounts ride one rail |
| **Ticket size** | `mg_avg_ticket_*` | many small tickets → structuring |
| **Alert timing** | `mg_alert_share_night`, `mg_alert_time_entropy` | mules' night-alert share runs ~3× higher (0.198 vs 0.065) |
| **Balance shape** | `mg_balance_volatility_*` | spike-and-drain rather than held balance |
| **Profile mismatch** | `mg_occ_deviation_*` | the dataset already ships **444** `D_*_OCC` columns — the PDF's "occupation–income divergence" needed no invention |

Real signal confirmed in the data: **Aadhaar Payment Bridge (APB)** deviation
features are the strongest legitimate correlates; students show a 1.94% mule rate
against a 0.89% base, rural 1.44%, Savings accounts 1.28% vs Current 0.20%.

---

## 10. Leak defences

Four layers, because correlation thresholds alone are not a defence:

1. **Semantic, correlation-independent.** Post-outcome fields are removed by what
   they *mean*: `FRAUD_SUSPECTED` (corr **0.97**), `OTHER_RESOLUTION`,
   `FALSE_POSITIVE` (corr **0.05**), `UNATTENDED`, `MIN/MAX_RESOLVE_DAYS`. All are
   written only after an analyst closes a case, so none exists at scoring time —
   and `FALSE_POSITIVE` at 0.05 would sail past any correlation threshold.
2. **Structural.** `MNTH` alone separates the classes perfectly. Dropped, with the
   crosstab published in the report as evidence.
3. **Extract hardening.** Any column whose *blank rate* differs between classes by
   more than 10% is dropped outright. This can only remove signal, never
   manufacture it — the conservative direction.
4. **Separation audit.** Every remaining column is scanned for disjoint class
   ranges or a near-exclusive value. This is what would catch the next `MNTH`.

---

## 11. Why the metrics are trustworthy *as metrics*

Everything that touches the label is fitted **inside** the training fold and
applied frozen to validation rows: feature selection, base models, stacking
weights, isotonic calibration, **and the operating threshold**. Median imputation
is fitted there too — it does not touch the label, but fitting it across all
9,082 rows would still let validation rows shape how training rows are filled.
No validation row influences how it is later scored.

This corrects three biases in the previous version — calibration fitted on the
predictions it was scored on, a threshold chosen by scanning the curve it was
reported from, and a meta-learner trained on training-set predictions where
XGBoost is near-perfect.

The whole procedure repeats across several shuffles, because 81 positives across
5 folds means ~16 mules per fold and the metric moves several points on the seed
alone. Results are reported **mean ± std**, and `reports/03_metrics.json` carries
per-fold detail plus lift over the base rate — the number an AML desk actually
uses.

---

## 12. Layout

```
muleguard/
├── run.ps1 / run.sh          # one-command launchers (Windows / Unix)
├── README.md, SETUP.md, STATUS.md
├── requirements.txt, verify_env.py
├── data/                     # DataSet.csv + intermediates (git-ignored)
│   └── oof_shap.npz          # per-account out-of-fold SHAP
├── src/
│   ├── config.py             # single source of truth
│   ├── schema.py             # dataset adaptation — target, IDs, leaks, partitions
│   ├── roles.py              # resolves columns by MEANING, not by name
│   ├── dictionary.py         # the data dictionary as executable knowledge
│   ├── utils.py
│   ├── 06_integrity.py       # Stage 0 — read this output first
│   ├── 01_clean.py … 05_score_explain.py
│   ├── plots.py, pipeline.py, make_synthetic.py
├── app/                      # web layer
│   ├── server.py             # FastAPI routes + error handling
│   ├── service.py            # artefact loading, account analysis, live scoring
│   └── static/               # index.html, brutal.css, app.js — no build step
├── tests/                    # test_schema.py, test_roles.py (25 tests)
├── models/                   # trained ensemble (git-ignored)
└── reports/
    ├── 00_INTEGRITY.md       # the finding that frames every number
    ├── 01_clean_report.json … 06_integrity_audit.json
    └── pr_curve.png, roc_curve.png, calibration_curve.png
```

---

## 13. Honest limitations

- **The benchmark is contaminated** (§4). The reported AUPRC is an upper bound on
  what this dataset can show, not a mule-detection result.
- **No transaction graph.** All 3,924 variables aggregate an account's own
  activity; not one names a counterparty. PageRank and label propagation need a
  who-paid-whom ledger, so Stage 6 self-skips and says so rather than inventing
  an edge list.
- **Hyperparameters are deliberately untuned.** Tuning against a confounded target
  optimises the artefact.
- **The 29 typology features need the right columns to exist.** On a dataset with
  no transaction aggregates the pipeline still runs, but degrades to row-profile
  features and says which bases were missing rather than inventing zeros.
- **81 positives is small.** Every metric carries a wide error bar; that is why
  they are reported with one.

---

## 14. Reproducing

```powershell
.\run.ps1 -Verify                    # environment check
.\run.ps1 -Stage 06_integrity.py     # just the integrity audit
$env:MULEGUARD_REPEATS = "1"         # faster training (default 5)
```

```bash
./run.sh verify
./run.sh stage 06_integrity.py
MULEGUARD_REPEATS=1 ./run.sh
```

Results are deterministic: identical data and `random_state` give identical
numbers, independent of `PYTHONHASHSEED`. That was **not** true before — set
iteration order fixed the one-hot column order and XGBoost's `colsample_bytree`
samples columns by index, so runs drifted. Verified by running Stages 1-2 under
two different hash seeds and comparing the feature matrices byte for byte.

No dataset? `python src/make_synthetic.py` writes a correctly-shaped fake file
with a planted `F3912` leak, to exercise the plumbing only.
