# MuleGuard AI

**Money-mule account detection — precision-first, leak-hardened, and honest
about what its own benchmark can prove.**

A reproducible ML pipeline for the PSB Cybersecurity, Fraud & AI Hackathon 2026
(Bank of India × IIT Hyderabad). It detects mule accounts in an extremely
imbalanced dataset (**81 mules in 9,082 accounts = 0.892%**), explains every
alert in named banking variables, and reports measured nested cross-validation
metrics — not design targets.

> **Runs on Windows, macOS, and Linux** from one source tree. `pathlib`
> throughout, UTF-8 forced on all I/O, no shell calls, no hardcoded paths.

> **Presenting to judges?** Start at **[§1 — Quick start](#1-quick-start)** to get
> it running, then **[§4 — The command center](#4-the-command-center)** for the
> live walkthrough, and
> **[§5 — What the pipeline measures about itself](#5-what-the-pipeline-measures-about-itself)**
> for the integrity story.

> **What is not in this repository.** The supplied dataset, the trained model and
> the generated alert records are all deliberately git-ignored. The alerts in
> particular name 105 real accounts as suspected mules and are joinable back to
> the bank's file by row index, so they are not published. Everything here
> regenerates them from your own copy of the data.

---

## Read this first

**The challenge injected red-herrings on purpose, and finding them is a graded
criterion. Stage 0 exists to find them.**

The National Fraud Prevention Challenge brief is explicit on both points:

> *"Labels may contain noise/red-herrings. Not all labels are guaranteed to be
> correct."*
>
> **15% weightage for avoidance of red-herrings in data** — *"Rewarded for
> successfully avoiding several red-herrings injected in the training data."*

That is the same weight the brief gives to additional insights, and more than it
gives to report quality. So the integrity audit in this project is not a
complaint about somebody's data. It is a deliverable, and it runs before any
model is fitted rather than after the metrics look good.

**What it found, without being told anything.** Every negative in the supplied
account-level file comes from the **October** extract; every positive comes from
the **September, November, or December** extracts. No month contains both
classes. So any difference between monthly extraction runs correlates perfectly
with the label while describing nothing about any customer. This is precisely
the shape of a deliberately planted artefact, and `schema.partition_columns()`
identified it from structure alone — low cardinality, class-pure values — with
no knowledge of the schema and no hint that anything had been planted.

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
to month — cannot be separated within this file by any method. Separating it
needs negatives and positives sampled from the **same months**; that is a
sampling change, and it applies to every team working from this file.

**Why this matters more than the headline score.** A team that does not run this
check reports a number inflated by the planted artefact and cannot say by how
much. We can: §5 quantifies it at +0.068 AUPRC for the raw columns over a model
holding no values at all. Reporting a smaller, defensible number is the point of
the exercise the brief set.

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

**Given a dataset live?** Start the UI and drag the file in at **02 UPLOAD
DATASET** (about 100 seconds for a 5,000-row file), or from the terminal:

```powershell
.\run.ps1 -Dataset D:\theirs.csv     # writes under runs\theirs\, never touches your own results
```

```bash
./run.sh dataset /path/theirs.csv
```

It works out the target, the identifiers, the leak columns and the partition
columns for itself. See [SETUP.md](./SETUP.md) for what demo mode trades away.

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
| `DataSet*.csv` | 9,082 × 3,925. Target `F3924` (`FRAUD_TGT`). Not in this repo. |
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
| Where is `TOT_TXNAMT_CR_L7D`? | resolved by **role**, not name — see below | exact string equality |

### Resolution by role

`src/roles.py` stops matching names and starts matching meaning. Every retail
banking column name encodes the same handful of ideas — a **statistic**, a
**measure**, a **direction**, a **window**, a **channel** — so the request and
every available column are decomposed into that tuple and matched tuple to tuple:

| Column name in the wild | Resolved role |
|---|---|
| `TOT_TXNAMT_CR_L7D` | total · amount · credit · 7d |
| `InwardAmount7Day` | amount · credit · 7d |
| `sum.amt.in.7d` | total · amount · credit · 7d |
| `credit_value_week` | amount · credit · 7d |
| `AVG_BAL_MNTH` | average · balance *(no window: "month" is blocklisted — in extracts like these it usually means which file a row came from)* |
| `account_number` | *unclassified — the identifier guard runs before the count vocabulary, or "number" would parse as a count* |

When no column carries a requested role the lookup returns nothing and the stage
**records the miss in its report**. It never substitutes an approximate column.

### The partition test is the interesting one

`MNTH` was removed from this dataset because a human read the file and spotted
it. That does not generalise. `schema.partition_columns()` finds the same column
by its **shape** — low cardinality, values split between the classes rather than
shared by them — so the defence fires on a file nobody has inspected. On the
supplied data it re-derives `MNTH` (purity 1.000) unaided, and publishes the
crosstab as evidence rather than asserting a verdict. A genuine behavioural
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
four traps, builds features by role resolution, and — correctly — reports that
file as *not* contaminated, because it isn't.

Covered by `tests/test_schema.py`, `tests/test_roles.py` and `tests/test_rings.py` (**143 tests**).

### Environment overrides

| Variable | Effect |
|---|---|
| `MULEGUARD_DATA` | dataset path |
| `MULEGUARD_DICT` | data dictionary (`.xlsx` or `.csv`), optional |
| `MULEGUARD_TARGET` | name the target explicitly, skipping detection |
| `MULEGUARD_WORKDIR` | write `data/ models/ reports/` under here, so several datasets coexist |
| `MULEGUARD_REPEATS` | CV repeats (default 3) |
| `MULEGUARD_FAST` | demo mode — fewer trees and 1 repeat, for a live handover |
| `MULEGUARD_MAX_ROWS` | row cap in demo mode (default 60,000) |

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

When the dictionary is absent, `src/roles.py` (§2) does the same job from the
column names themselves.

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
| 00 | **The Problem** | What a mule account actually is, why rarity/camouflage/cost-of-error make it hard, and what was built. No dataset numbers — the method has to earn them. |
| 01 | **How It Works** | The whole method before any results: container vs conduit, the seven stages, every component and why it beat the obvious alternative, how an unseen schema is read, and what the system refuses to do. |
| 02 | **Upload Dataset** | Drag in a CSV, TSV, Excel or parquet file and run the full pipeline against it. |
| 03 | **Judge Mode** | The whole project in 12 steps, ~90 seconds, arrow-key driven, ending in a live account analysis. |
| 04–05 | Overview / Dataset | Scale, cleaning ledger, encoding, missing-value treatment, and how the schema was inferred. |
| 06 | **Integrity audit** | The partition column, the three falsification tests, the verdict and the grounds it rests on. |
| 07 | Leakage defence | All four layers with their evidence, plus the correlation backstop. |
| 08 | Mule features | All 29, grouped by family, each with why it is diagnostic. |
| 09 | Models | Per-model scores, stacking coefficients, both operating points, reproducibility. |
| 10 | Explainability | Global SHAP ranking with real banking variable names. |
| 11 | **Rules & Ablation** | The twelve-rule AML baseline measured against the base rate, and how much of our own score is the artefact. |
| 12 | Risk triage | Band edges and their provenance, band precision and recall. |
| 13 | **Account analysis** | Score, calibrated probability, band, SHAP reasons, evidence against the population, next steps, and the investigator decision panel. |
| 14 | **Operating cost** | Precision@K, false alarms per thousand, review budget, latency, extract drift. |
| 15 | **Audit trail** | Every investigator decision, timestamped and append-only. The retraining set. |
| 16 | Pipeline | Stage flow and artefact health. |

### Diagrams

Six hand-drawn inline SVG figures carry the argument visually — no libraries, no
image files, no external requests:

| Figure | Where | What it shows |
|---|---|---|
| 1 | 01 | Balance over time for a customer versus a conduit. The whole detection premise in one picture. |
| 2 | 01 | Dataflow: what enters each stage, what artefact it writes, and the conditional branch where the graph stage disables itself. |
| 3 | 01 | The leak funnel — four gates and what each one removes. |
| 4 | 01 | Which components are fitted inside the training fold, with the decision threshold highlighted. |
| 5 | 01 | Three schema-specific column names collapsing to one role tuple. |
| 6 | 12 | One score, three actions, two operating points. |

The same four method schematics exist as print figures for the paper, drawn in
its serif/300 dpi style by `src/paper_fig_method.py`.

### Three rules the UI holds to

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

### API

28 endpoints. Interactive docs at `/api/docs`, OpenAPI 3 at `/api/openapi.json`.

| Group | Endpoints |
|---|---|
| Status | `GET /api/health`, `/api/schema` |
| Reports | `GET /api/overview`, `/api/integrity`, `/api/clean`, `/api/leakage`, `/api/features`, `/api/models`, `/api/metrics`, `/api/shap`, `/api/bands`, `/api/rules`, `/api/ablation`, `/api/operating` |
| Accounts | `GET /api/accounts`, `/api/account/{idx}`, `POST /api/score` |
| Feedback | `POST /api/decision`, `GET /api/decisions` |
| Upload | `POST /api/jobs/upload`, `GET /api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/results`, `POST /api/jobs/{id}/cancel` |
| Export | `GET /api/export/alerts`, `/api/export/casepack/{idx}`, `/api/export/contract` |

The server binds to `127.0.0.1` by default. It loads a pickled model, so it is a
local analyst tool and is not hardened for exposure on a shared network. No
request supplies a path, filename, or module name; uploads are extension-checked,
size-capped, renamed server-side and run in a subprocess.

---

## 5. What the pipeline measures about itself

Two stages exist purely to test our own claims, and both produced uncomfortable
answers that we publish rather than bury.

### Feature ablation (`src/08_feature_ablation.py`)

How much of our score is the extract artefact rather than detection? Three runs,
identical folds, identical seed:

| Condition | Features | AUPRC | vs random |
|---|---|---|---|
| Everything | 1,506 | 0.883 | 99× |
| **Raw columns only** | 1,477 | **0.892** | 100× |
| *Blank patterns only, no values at all* † | *3,772 bits* | *0.824* | *92×* |
| **Behavioural features only** | 29 | **0.368** | 41× |

† from the integrity audit's test A, run on the same folds.

Read rows 2 and 3 together. The raw columns, with every value present, beat a
model that has **no values at all** by only 0.068. Almost everything they
contribute on this dataset is provenance.

The 29 behavioural features score 0.368 alone, 41× better than random. They are
ratios, so they survive a change in which fields an export populated. That makes
them the part least explained by the artefact, and **the number we would actually
defend**.

### The AML rule layer (`src/09_rules.py`)

Twelve rules from published money-mule typology, thresholds **not tuned** against
this dataset. Tuning them here would fit the confound and fail elsewhere.

- **7 of 12 perform at or below the base rate.**
- Two catch zero mules.
- All twelve combined flag 7,862 accounts, 87% of the book, at 103.4 alerts per mule.

The typology is inverted here: ordinary customers have a median 7-day
pass-through of **0.776**, mules **0.622**. Mules pass through *less* money than
everybody else.

Two rules survived: **structuring by ticket size** (4.7× lift, finds 45 of 81
mules) and **single payment rail** (1.4×). We would keep those two and delete the
other ten.

### Headline results

Every figure below is a mean over nested repeated CV, with its own standard
deviation. Read them against the integrity finding above, not on their own.

| Operating point | Precision | Recall | AUPRC | For |
|---|---|---|---|---|
| **Precision-first** | **0.989 ± 0.030** | 0.625 ± 0.147 | 0.882 ± 0.062 | Automated action: freeze, escalate, prepare an STR |
| High-recall | 0.377 ± 0.063 | **0.917 ± 0.056** | 0.882 ± 0.062 | Human review queue: accept more false alarms to miss fewer mules |

Bands, from fitted operating points (edges **2.81** and **912.09**, derived — not
round numbers somebody picked). The high cutoff targets **0.99** precision, not
0.90, because that band triggers an automated freeze with no human in the loop:

| Band | Accounts | Real mules | Precision | Action |
|---|---|---|---|---|
| **HIGH** | 49 | **49** | **1.000** | Freeze outward transfers, escalate to AML, prepare STR |
| MEDIUM | 190 | 25 | 0.132 | Enhanced monitoring, OTP step-up on transfers |
| LOW | 8,843 | 7 | 0.001 | Routine monitoring |

Review the top 50 accounts and you find 50 mules with **zero** false alarms.
Review 239 (2.6% of the book) across both bands and you catch 74 of 81 — **91% of
the fraud** — with no genuine customer frozen. The remaining 7 mules score below
most ordinary customers; they are not empty accounts, they simply behave like
customers, and the fact that would separate them is who sent them the money.

**Where the 0.99 target costs you.** Demanding near-certainty for an automated
freeze cuts single-point recall from 0.712 to 0.625, and the review queue more
than doubles from 105 accounts to 239. What it buys is a freeze band that is
100% mules: at a 0.90 target the same model puts 60 accounts in that band, of
which **2 are innocent people whose money would be stopped**. We took the
smaller, stricter band. The target was reachable on inner data in **93.3% of
folds** rather than all of them, and the one fold where it was not fell back to
best-F1; that is recorded in `03_metrics.json` rather than smoothed over.

---

## 6. EFRMS and AML platform integration

The pipeline emits alerts, case packs and an audit trail in a **documented,
vendor-neutral schema** whose fields map onto the concepts every AML case
management system uses.

```
reports/exports/alerts.json                 full alert payload
reports/exports/alerts.csv                  batch exchange format
reports/exports/case_pack_<id>.json         one investigator case pack
reports/exports/integration_contract.json   the field mapping
```

Only the contract is committed to this repository; the alerts and case packs name
real accounts and are git-ignored.

Or over HTTP: `/api/export/alerts` (add `?format=csv`),
`/api/export/casepack/{idx}`, `/api/export/contract`.

| Field | Maps to |
|---|---|
| `alert_id` | Case reference. Deterministic, so a re-run updates rather than duplicates |
| `entity_type` / `entity_ref` | Always ACCOUNT; this scores accounts, not transactions |
| `risk_score` / `risk_probability` | 0–1000 and the calibrated probability behind it |
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

### It finds rings, not just neighbours of known mules

The first version of this stage propagated suspicion outward from accounts a
bank had **already confirmed**. That is genuinely useful and it is not ring
detection: its recall is bounded by how good the existing alert book was, and a
ring where nobody has been caught yet is invisible to it.

`src/rings.py` closes that gap. It finds the network from structure alone, with
**no labels and no seeds**, because a laundering ring has a shape ordinary
payment traffic does not:

| Evidence | What it measures |
|---|---|
| **Density** | members transact with each other far more than chance allows |
| **Isolation** | many internal edges, few to the outside — formally, low `conductance` |
| **Cycles** | money can return towards its origin in a few hops |
| **Pass-through** | value entering the group leaves again; little is retained |
| **Roles** | a collector fans in, relays chain onward, a terminal cashes out |

Any one alone is weak — families and small businesses form dense clusters, and
mutual payments make innocent cycles. Requiring several at once is what
separates a ring from a neighbourhood. Every candidate is returned **with the
evidence that produced it**, so an investigator sees why a group was grouped
rather than being handed an opaque cluster id.

### Proving the code works anyway

"It would work if the data allowed it" is a claim, so we made it demonstrable:

```bash
python src/graph_demo.py     # -> reports/demo_graph/
```

This builds a synthetic ledger (2,000 accounts, 4,320 transfers, 8 planted rings)
and runs the **real** `propagate()` from `04_graph.py` against it, imported
rather than copied. Two members of each ring are treated as already confirmed;
the rest have to be found.

| Stop after | Queue size | Members found | Recall | Precision | Alerts per find |
|---|---|---|---|---|---|
| **hop 1** | **93** | **33** | **92%** | **35%** | **2.8** |
| hop 2 | 442 | 36 | 100% | 8% | 12.3 |
| hop 3 | 1,222 | 36 | 100% | 3% | 33.9 |

**One hop does almost all the work.** It recovers 92% of the unknown ring members
at under three alerts per find. Hops two and three add 3 more members and 1,129
innocent accounts. That collapse is the honest argument against untuned
propagation, and it is why the pipeline consumes this as a **0.15 blend weight**
on the risk score rather than as an alert.

The same ledger, run through `rings.py` with **nothing given to it at all** — no
seeds, no labels, no count of how many rings exist:

| Top-K candidates | Accounts surfaced | Planted members found | Precision | Recall | Alerts per find |
|---|---|---|---|---|---|
| **3** | 18 | **18** | **100%** | 34.6% | **1.0** |
| 5 | 29 | 25 | 86.2% | 48.1% | 1.2 |
| **8** | 49 | 31 | 63.3% | **59.6%** | 1.6 |
| 20 | 111 | 31 | 27.9% | 59.6% | 3.6 |

The two are complementary rather than competing. **Propagation reaches further
(92%) but has to be told where to start.** Structural detection starts from
nothing and its top three candidates are exactly right, then plateaus near 60%:
the members it cannot reach are the ones whose only link to the ring is a single
edge, which no amount of community detection recovers.

Recall stops improving after eight candidates while false positives keep
accruing, so the honest operating point is "review the top handful", exactly as
with Precision@K elsewhere in this project.

> **This is a capability demonstration, not a result.** The rings were planted by
> the script and then found by the script. It says nothing about mule detection
> on the supplied data. Every output lives under `reports/demo_graph/` and the
> figure is watermarked so it can never be mistaken for a real finding.

---

## 8. Finding rings, and what a third-party benchmark said about it

The first version of Stage 6 propagated suspicion outward from accounts a bank
had **already confirmed**. Its recall is bounded by how good the existing alert
book was, and a ring where nobody has been caught yet is invisible to it. Two
modules close that gap, and they fail in different places — which is the useful
part.

### `rings.py` — structural community detection

Finds groups that are densely connected internally and sparsely connected
outward, using density, conductance, internal cycles, group pass-through and
role composition. **No labels, no seeds.** On a synthetic ledger with planted
rings its top three candidates are exact hits (18 accounts, 18 planted, 1.0
alerts per find), reaching ~60% recall by the eighth candidate.

### Then we benchmarked it against SAML-D, and it failed

[SAML-D](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
(Oztas et al., IEEE ICEBE 2023) is third-party ground truth: 855,460 accounts,
9.5M transactions, 28 named typologies. `rings.py` does not flag its rings, and
the measurement says exactly why:

| Typology | Conductance of the true ring |
|---|---|
| Layered_Fan_Out | 0.990 |
| Layered_Fan_In | 0.987 |
| Cycle | 0.988 |
| Gather-Scatter | 0.991 |
| Stacked Bipartite | 0.979 |

A conductance near 1.0 means **each ring member carries ~200 transactions of
which only one or two are ring edges** — the laundering is about 1% of that
account's activity. The rings are real and ring-sized (38–52 components of 10–16
accounts each); they are simply not *separable*. No community method finds them,
and our gates reject them rather than manufacturing confidence.

That is a property of real muling, not a quirk of one dataset. A recruited
account keeps paying its bills.

### `motifs.py` — find the shape, not the community

So look for the laundering shape locally and in time instead, which needs no
global separability. Measured on the same SAML-D ground truth, no labels used:

| Motif | Accounts | Precision | Lift | Recall |
|---|---|---|---|---|
| **FAN_OUT** | 18,798 | 11.2% | **28.3×** | **62.1%** |
| FAN_IN | 13,766 | 7.5% | 18.9× | 30.4% |
| GATHER_SCATTER | 8,610 | 4.1% | 10.3× | 10.4% |
| CHAIN | 33,080 | 3.2% | 8.1× | 31.4% |

*(base rate 0.397%; whole file scored in 62 seconds)*

**62% of network-laundering accounts recovered inside 2.2% of the book, at 28×
the base rate, from an unlabelled transaction log.**

### Two of our own hypotheses did not survive

- **Blending the motifs made it worse.** One combined score gives 12.4× lift.
  FAN_OUT alone gives 28.3×. Averaging a strong detector with weak ones destroys
  it, so per-motif account sets are returned alongside the blend and a deployer
  can decline it.
- **Amount uniformity did not discriminate.** We expected a laundering split into
  near-equal parts to separate from a payroll run. It does not: filtering to the
  most uniform fan-outs *lowers* precision from 11.2% to 9.0% and cuts recall
  from 62% to 11%. Value conservation behaved the same way. Both are still
  reported as evidence on an alert; neither is used to gate.

Thresholds were fixed before the benchmark and left alone after it. Full result
in [`reports/bench_saml/saml_ring_benchmark.json`](./reports/bench_saml/saml_ring_benchmark.json).

---

## 9. When was it a mule? Temporal localisation

The challenge submission format asks for four columns, not two:

```csv
account_id,is_mule,suspicious_start,suspicious_end
ACCT_000003,0.87,2023-11-15T09:30:00,2024-02-20T16:45:00
```

and scores the window separately with **temporal IoU** against the ground-truth
activity period. A probability alone leaves half the submission blank.

This is a different question from "is this account a mule". A mule account is
usually a real person's real account that behaved normally for years, ran hot for
weeks or months, and then went quiet. Flagging the account is stage one. Saying
*which* period was the laundering decides which transactions go into the STR and
which are an innocent salary history.

`src/temporal.py` scores every day of an account's history against five typology
signals — pass-through symmetry, volume burst, velocity, structuring under a
reporting threshold, and round amounts — then takes the contiguous interval
carrying the most excess suspicion, via Kadane's maximum subarray.

### The one decision that made it work

The baseline you subtract before running Kadane is the whole ballgame:

| Baseline | Mean IoU | Median predicted window |
|---|---|---|
| Account's own median | **0.063** | **898 days** |
| Fixed high quantile (0.90) | 0.994 | 57 days |
| **Otsu's method** | **0.998** | **57 days** |

*(planted episodes had a median true length of 57 days inside a 900-day history)*

Kadane maximises a **sum**, so if the typical day carries even slightly positive
excess the window never stops growing — with a median baseline it returned the
entire history and scored 0.06.

A fixed quantile fixes that, but only by assuming what share of the history is
laundering, and tuning that share against our own generator would prove nothing
except that we can fit our own generator. **Otsu assumes no such share.** It asks
whether an account's daily scores form one blob or two and cuts where they are
most distinct, so a genuine episode and a flat history go through the same rule.
Its between-class variance is returned as `window_confidence`, and it separates
the classes on its own: median **0.031** for accounts with an episode against
**0.007** for those without.

Reproduce with `python src/temporal_demo.py`. Like the graph demonstration, the
episodes were planted by the script and then found by the script — it proves the
code path works and says nothing about the supplied dataset, which contains no
transactions to localise.

---

## 10. One system, and what it scores end to end

Everything above measures a component. This measures the whole thing.

`src/score_unified.py` takes a transaction ledger in one end and produces the
required submission out the other, running four independent signal families:

| Family | What it reads | Module |
|---|---|---|
| **Behavioural** | one account's own money: pass-through, retention, burstiness, threshold-hugging | `ledger_features.py` |
| **Motif** | the shape of a laundering event in a time window | `motifs.py` |
| **Structural** | closed cells, dense inside and sparse outward | `rings.py` |
| **Temporal** | when the episode happened | `temporal.py` |

They fail in *different places*, which is the point of having four. Motifs need
no global structure and work when a mule keeps trading normally. Rings need the
cell to be separable and fail when it is not. Behavioural features need no graph
at all. Blending them is coverage, not padding.

**The behavioural features are built from the raw ledger**, not from
pre-aggregated columns. That is what lets the same typology signals work on any
bank's transaction table rather than only on a file that already contains
`TOT_TXNAMT_CR_L7D`.

### Combined by a fitted model, not a hand-weighted sum

Motif and ring scores enter as ordinary columns and the ensemble learns what
each is worth. A hand-weighted blend would be us asserting the weights; this
makes the data assert them — including the possibility that a signal is worth
nothing, which is exactly what happened to the isolation forest and got
published rather than hidden. Without labels it falls back to a documented
rank-average and says so.

### Measured end to end on SAML-D

493,833 accounts, 2.7M transactions, third-party ground truth, model fitted on
the training split only:

| Held out: 148,150 accounts, base rate 0.461% | |
|---|---|
| **AUPRC** | **0.4207 — 91× the base rate** |
| AUROC | 0.9853 |
| **Precision @ top 50** | **70.0% — 152× lift** |
| Recall @ top 1,000 | 61.4% at 41.9% precision |

### Does the network layer actually earn its place?

Same ledger, same split, same seed, same model. The only difference is whether
the 10 motif/ring/role columns are present:

| | Behavioural only | **+ network** | Δ |
|---|---|---|---|
| AUPRC | 0.325 | **0.421** | **+0.096 (+29%)** |
| Lift over base rate | 70.5× | **91.2×** | +20.7 |
| Precision @ 50 | 58.0% | **70.0%** | **+12 pts** |
| Recall @ 1,000 | 48.9% | **61.4%** | **+12.5 pts** |

**Ten of thirty-eight columns buy a 29% relative gain in AUPRC.** On this data
the graph evidence is not decoration, and that is measured rather than claimed.

Two caveats, both recorded in
[`reports/bench_unified/network_ablation.json`](./reports/bench_unified/network_ablation.json):
account labels are **derived** from transaction labels, because SAML-D labels
transactions while a bank labels accounts; and validation is a single stratified
split rather than the nested repeated CV used for the headline dataset. Both
affect how the number should be read.

```bash
python src/bench_unified.py --months 3               # the full system
python src/bench_unified.py --months 3 --no-network  # the ablation
```

---

## 11. Pipeline stages

Run by `src/pipeline.py`, which executes each stage in order and prints the final
summary. `src/config.py` is the single source of truth for every path, threshold,
and hyper-parameter.

| Stage | Script | What it does |
|---|---|---|
| 0 | `06_integrity.py` | **Dataset integrity audit.** Three falsification tests; writes `reports/00_INTEGRITY.md`. Run and read first. |
| 1 | `01_clean.py` | Semantic leak removal, categorical encoding, activity-aware imputation, extract hardening, separation audit |
| 2/3 | `02_features.py` | 29 named mule-typology features + row-profile aggregates + honest graph-feasibility check |
| 4/5 | `03_train.py` | Nested repeated CV: in-fold selection, IsolationForest + XGBoost + LightGBM, stacking, isotonic calibration, threshold |
| 6 | `04_graph.py` | Seeded label propagation **and** unseeded structural ring detection (`rings.py`) — **only if** counterparty data exists |
| 7/8 | `05_score_explain.py` | 0–1000 risk score, LOW/MEDIUM/HIGH bands, SHAP reasons, per-account risk reports |
| 9 | `09_rules.py` | Deterministic AML rule layer, measured against the base rate, thresholds untuned |
| 10 | `08_feature_ablation.py` | How much of the score survives removing the behavioural features (research-only; skipped in fast mode) |
| 11 | `10_operating_metrics.py` | Precision@K, investigator load, review budget, scoring latency, extract drift |

Supporting modules, run on demand rather than by the pipeline:

| Script | Purpose |
|---|---|
| `integration.py` | Vendor-neutral alert / case-pack export for EFRMS and AML platforms |
| `graph_demo.py` | Synthetic-ledger demonstration of the graph stage (§7) |
| `temporal.py` | Suspicious-window localisation and temporal IoU (§8) |
| `temporal_demo.py` | Synthetic demonstration of window recovery |
| `plots.py` | PR / ROC / calibration curves |
| `paper_figures.py`, `paper_fig_method.py`, `paper_fig_ablation.py` | Publication figures |
| `make_paper.py`, `make_paper_tex.py` | Build the `.docx` and LaTeX manuscripts |
| `make_synthetic.py` | Generate a correctly-shaped fake dataset, including alien schemas |

---

## 12. The features encode a mule's actual behaviour

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

## 13. Leak defences

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

Row identifiers are removed before any of this, by name and by near-uniqueness —
with floating-point columns exempt, because every continuous measurement is
near-unique across thousands of rows and applying the rule to floats would
classify the entire feature matrix as identifiers.

---

## 14. Why the metrics are trustworthy *as metrics*

Everything that touches the label is fitted **inside** the training fold and
applied frozen to validation rows: feature selection, base models, stacking
weights, isotonic calibration, **and the operating threshold**. Median imputation
is fitted there too — it does not touch the label, but fitting it across all
9,082 rows would still let validation rows shape how training rows are filled.
No validation row influences how it is later scored.

Scheme: **nested 5-fold stratified CV, 3 repeats, 3 inner folds.**

This corrects three biases in the previous version — calibration fitted on the
predictions it was scored on, a threshold chosen by scanning the curve it was
reported from, and a meta-learner trained on training-set predictions where
XGBoost is near-perfect.

The whole procedure repeats across several shuffles, because 81 positives across
5 folds means ~16 mules per fold and the metric moves several points on the seed
alone. Results are reported **mean ± std**, and `reports/03_metrics.json` carries
per-fold detail plus lift over the base rate — the number an AML desk actually
uses.

The stacking coefficients are published too, including the negative one: the
isolation forest scores below random on this data (AUROC 0.314) and the
meta-learner assigns it **−0.44**. A component that did not work is reported as
not working rather than quietly dropped from the diagram.

---

## 15. Staying correct after deployment

Everything above measures the model on the day it was fitted. Two modules exist
because that is not the day it will be used.

### Label noise (`label_noise.py`)

Every other defence protects against bad *features*. None protects against a bad
*label*, and the operating threshold is chosen against those same labels, so a
mislabelled subset shifts the cutoff for everyone.

Confident learning on **out-of-fold** scores: for each class, take the average
confidence the model assigns to accounts carrying that label; an account whose
label disagrees *and* which clears the other class's threshold is a candidate.
Two conditions, not one, because "the model disagrees" alone flags every
borderline case.

**Labelled legitimate, scored high** is either a mule nobody caught or a false
positive. **Labelled mule, scored low** is either a planted red-herring label or
a real mule the data cannot show.

It does **not** claim any label is wrong. That ambiguity cannot be resolved by
arithmetic. The output is a ranked review queue with evidence attached, plus a
calibration-free rank check, because probabilities depend on the calibrator and
ranks do not.

### Drift and the re-selection policy (`drift.py`)

Detecting drift is the easy half. The hard half is deciding in advance what you
do about it. The module is built around one distinction:

| Signal | Available | Licenses |
|---|---|---|
| **Unsupervised** — feature PSI, score PSI, band populations | immediately, no labels | alarm, recalibration, retraining |
| **Supervised** — realised precision from investigator decisions | late, reviewed accounts only | **moving a threshold** |

**Unsupervised drift must never move a precision-targeted cutoff.** Re-fitting a
threshold to make band populations look normal is fitting to noise, and it would
conceal exactly the degradation the monitoring exists to find. Enforced by a test.

| Action | Trigger |
|---|---|
| `MONITOR` | inside tolerance |
| `RECALIBRATE` | scores shifted, features stable — refit the calibrator only |
| `RETRAIN` | weighted feature PSI ≥ 0.25 |
| `REFIT_THRESHOLDS` | realised precision >10% below target, on enough reviews |
| `HALT_AUTOMATION` | weighted PSI ≥ 0.50 — **automated freezing stops until a human signs off** |

That last rung exists because this system freezes people's money. There has to be
a written condition under which it stops doing that by itself.

**Hysteresis:** every rung above `MONITOR` needs the condition to hold for two
consecutive windows, so one noisy batch cannot trigger a retrain, and a clean
window resets the streak:

```
wk2 one odd batch          MONITOR            (absorbed)
wk3 it recovers            MONITOR
wk5 scores still sliding   RECALIBRATE
wk7 alerts still bad       REFIT_THRESHOLDS
wk9 population has moved   HALT_AUTOMATION    signoff required
```

PSI bands are the long-standing credit-risk conventions rather than numbers we
chose, because a bank's model-risk function already knows what 0.25 means. Drift
is weighted by SHAP importance: movement in a column the model ignores is not a
problem, and reporting it as one trains everybody to ignore the alarm.

---

## 16. Layout

```
muleguard/
├── run.ps1 / run.sh          # one-command launchers (Windows / Unix)
├── README.md, SETUP.md, STATUS.md
├── requirements.txt, verify_env.py
├── data/                     # dataset + intermediates (git-ignored)
├── models/                   # trained ensemble (git-ignored)
├── src/
│   ├── config.py             # single source of truth
│   ├── schema.py             # dataset adaptation — target, IDs, leaks, partitions
│   ├── roles.py              # resolves columns by MEANING, not by name
│   ├── dictionary.py         # the data dictionary as executable knowledge
│   ├── ensemble.py           # base models, stacking, isotonic calibration
│   ├── score_unified.py      # THE system: ledger in, submission out
│   ├── ledger_features.py    # behavioural features from raw transactions
│   ├── rings.py              # structural ring detection — no labels, no seeds
│   ├── temporal.py           # when the account was being used — window + IoU
│   ├── motifs.py             # AML typology shapes: fan-in/out, gather-scatter, chains
│   ├── label_noise.py        # which labels a human should re-check
│   ├── drift.py              # drift detection + operating-point re-selection policy
│   ├── utils.py
│   ├── 06_integrity.py       # Stage 0 — read this output first
│   ├── 01_clean.py … 05_score_explain.py
│   ├── 08_feature_ablation.py, 09_rules.py, 10_operating_metrics.py
│   ├── integration.py, graph_demo.py
│   ├── plots.py, pipeline.py, make_synthetic.py
│   └── paper_figures.py, paper_fig_method.py, make_paper.py
├── app/                      # web layer
│   ├── server.py             # FastAPI routes + error handling
│   ├── service.py            # artefact loading, account analysis, live scoring
│   ├── jobs.py               # upload → subprocess pipeline runs
│   └── static/               # index.html, brutal.css, app.js — no build step
├── tests/                    # test_schema.py, test_roles.py, test_rings.py, test_temporal.py, test_motifs.py, test_label_noise.py, test_drift.py, test_unified.py (143 tests)
├── paper/                    # LaTeX manuscript + vector figures
└── reports/
    ├── 00_INTEGRITY.md       # the finding that frames every number
    ├── MuleGuard_IEEE_Paper_v2.docx
    ├── figures/              # publication figures (PNG + PDF)
    ├── demo_graph/           # synthetic graph demonstration
    ├── exports/              # integration contract (alerts are git-ignored)
    └── 01_clean_report.json … 10_operating_metrics.json   (git-ignored)
```

---

## 17. Honest limitations

- **The benchmark is contaminated** (see [Read this first](#read-this-first)).
  The reported AUPRC is an upper bound on what this dataset can show, not a
  mule-detection result.
- **No transaction graph.** All 3,924 variables aggregate an account's own
  activity; not one names a counterparty. PageRank and label propagation need a
  who-paid-whom ledger, so Stage 6 self-skips and says so rather than inventing
  an edge list.
- **Nine mules are not caught at any usable threshold.** They rank below most
  ordinary customers and are fully populated, not empty. No account-level model
  separates them; that needs counterparty data.
- **Hyperparameters are deliberately untuned.** Tuning against a confounded target
  optimises the artefact.
- **The 29 typology features need the right columns to exist.** On a dataset with
  no transaction aggregates the pipeline still runs, but degrades to row-profile
  features and says which bases were missing rather than inventing zeros.
- **The isolation forest does not help here** and is reported with its negative
  stacking weight rather than removed from the story.
- **81 positives is small.** Every metric carries a wide error bar; that is why
  they are reported with one.

---

## 18. Reproducing

```powershell
.\run.ps1 -Verify                    # environment check
.\run.ps1 -Stage 06_integrity.py     # just the integrity audit
$env:MULEGUARD_REPEATS = "1"         # faster training (default 3)
```

```bash
./run.sh verify
./run.sh stage 06_integrity.py
MULEGUARD_REPEATS=1 ./run.sh
```

Run the tests:

```bash
python -m pytest tests/ -q           # 143 tests
```

Results are deterministic: identical data and `random_state` give identical
numbers, independent of `PYTHONHASHSEED`. That was **not** true before — set
iteration order fixed the one-hot column order and XGBoost's `colsample_bytree`
samples columns by index, so runs drifted. Verified by running Stages 1–2 under
two different hash seeds and comparing the feature matrices byte for byte.

No dataset? `python src/make_synthetic.py` writes a correctly-shaped fake file
with a planted `F3912` leak, to exercise the plumbing only. Add
`--schema alien` for a file that shares no column name with the original.
