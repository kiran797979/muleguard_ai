# Running it in a bank

The command centre, alert export, and what keeps the system correct after deployment.


---

## The command center

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

## EFRMS and AML platform integration

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

## Staying correct after deployment

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

[← Back to the README](../README.md) · [Docs index](./README.md)
