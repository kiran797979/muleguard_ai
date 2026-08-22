# Data integrity and self-measurement

The checks that run before any model is fitted, what they found, and how much of a reported score they account for.


---

## What the benchmark can and cannot prove


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
**[`reports/00_INTEGRITY.md`](../reports/00_INTEGRITY.md)**.

The pipeline removes this artefact wherever it can be identified (see
[Leak defences](./method.md#leak-defences)). What
remains unidentifiable — a genuine behavioural feature that *also* drifts month
to month — cannot be separated within this file by any method. Separating it
needs negatives and positives sampled from the **same months**; that is a
sampling change, and it applies to every team working from this file.

**Why this matters more than the headline score.** A team that does not run this
check reports a number inflated by the planted artefact and cannot say by how
much. We can: the feature ablation below quantifies it at +0.068 AUPRC for the
raw columns over a model
holding no values at all. Reporting a smaller, defensible number is the point of
the exercise the brief set.

---

---

## What the pipeline measures about itself

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

[← Back to the README](../README.md) · [Docs index](./README.md)
