# How it works

The mechanism, end to end: how the pipeline reads a schema it has never seen, what it builds from a ledger, and why each component was chosen over the obvious alternative.


---

## It runs on any dataset

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

## The dictionary is the differentiator

Without `Description.xlsx`, this data is 3,924 opaque numbers. With it, every
leak decision, feature, and SHAP reason is a named banking variable.
`src/dictionary.py` turns the spreadsheet into structured knowledge:

- `F3891` → `CUST_OCCP` → *"Occupation code of customer"*
- Name decomposition: **stat** (`R`/`RA`/`D`/`DA`/`AVG`/`MAX`/`MIN`/`TOT`) ×
  **channel** (CASH/UPI/ATM/APB/…) × **direction** (CR/DB) × **window**
  (7D/14D/31D/7-14/7-31/14-31)
- The bank's own **18 shortlisted variables** (`Bank_Finalized_Variables`) — a
  free domain-expert feature selection, used as a prior
- **Leak classification by meaning**, which is what makes the semantic layer of
  [Leak defences](#leak-defences) possible

When the dictionary is absent, `src/roles.py` ([resolution by role](#it-runs-on-any-dataset))
does the same job from the
column names themselves.

---

## The features encode a mule's actual behaviour

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

## Leak defences

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

## Why the metrics are trustworthy *as metrics*

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

[← Back to the README](../README.md) · [Docs index](./README.md)
