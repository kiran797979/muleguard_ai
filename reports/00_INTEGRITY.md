# MuleGuard AI — Dataset Integrity Report

**Read this before quoting any metric from this project.**

Random-guess AUPRC on this data is **0.0089** (0.89% of accounts are mules).

## The problem

`F2230 (MNTH)` shows how the sample was assembled:

| Value | Negative | Positive |
|---|---|---|
| Dec25 | 0 | 10 |
| Nov25 | 0 | 23 |
| Oct25 | 9,001 | 0 |
| Sep25 | 0 | 48 |

**0 value(s) contain both classes.**
The two classes fall into disjoint groups, so every difference between those
groups — which fields were populated, how the feed behaved — lines up with the
label while describing no customer behaviour at all. This column was found by
its shape, not by its name: no prior knowledge of the schema was used.

## Falsification tests

Each test feeds a model information that *cannot* identify a mule. All three
should score near 0.0089 if the dataset is sound.

| Test | What it uses | AUPRC | AUROC | vs random |
|---|---|---|---|---|
| A. Missingness only | blank/not-blank pattern, **no values** (3772 indicators) | **0.8236** | 0.9925 | 92x |
| B. Individually-useless | 250 columns each with \|corr\| < 0.05 | **0.7361** | 0.9734 | 83x |
| C. Shuffled labels | same columns, labels randomised | 0.0094 | 0.5284 | 1x |

Test C collapsing to the baseline confirms the evaluation harness is sound — so
the scores in A and B are real properties of the data, not a bug.

Test A is the decisive one: **knowing only which cells were blank, with every
number thrown away, identifies mules almost perfectly.** No model can distinguish
that from genuine behaviour, because within this file the two are the same thing.

## Verdict

CONTAMINATED on 3 ground(s): a model given only blank/not-blank patterns separates the classes far above the base rate; columns that are individually useless still separate the classes when combined; the classes fall into disjoint value sets of F2230 (MNTH), so that column alone reproduces the label. Metrics from this dataset measure how the sample was assembled as well as customer behaviour, and the two cannot be cleanly separated within this file.

## What the pipeline does about it

1. **Partition-column removal** — the pipeline drops `F2230 (MNTH)` outright — it alone separates the classes.
2. **Drops post-outcome fields** — resolution status flags and resolve-days are
   written after an investigation closes and do not exist at scoring time.
   `FRAUD_SUSPECTED` correlates 0.97; `FALSE_POSITIVE` correlates 0.05 and is
   just as unusable, which is why a correlation threshold is not a leak defence.
3. **Extract hardening** — drops every column whose blank rate differs between
   the classes by more than 10%, on the grounds that whether a cell was
   populated is a property of the extraction job, not of a customer.
4. **Nested cross-validation** — feature selection, stacking, calibration and the
   operating threshold are all fitted inside the training fold, so none of the
   remaining optimism is the evaluation's fault.

Steps 1-3 remove the artefact where it can be identified. They cannot remove what
is unidentifiable: a behavioural feature that also drifts month to month is
confounded, and no amount of modelling separates the two within this file.

## What would fix it

Negatives and positives sampled from the **same** groups. With both classes
present in every extract, the partition can be controlled for and the reported
numbers would measure behaviour alone. This is a data-collection change, not a
modelling one — worth raising with whoever supplied the data, because it applies
to everyone working from this file, not just this submission.
