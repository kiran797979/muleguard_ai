# The whole system

Temporal localisation, the unified scorer, and the end-to-end result on an external dataset.


---

## When was it a mule? Temporal localisation

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

## One system, and what it scores end to end

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
[`reports/bench_unified/network_ablation.json`](../reports/bench_unified/network_ablation.json):
account labels are **derived** from transaction labels, because SAML-D labels
transactions while a bank labels accounts; and validation is a single stratified
split rather than the nested repeated CV used for the headline dataset. Both
affect how the number should be read.

```bash
python src/bench_unified.py --months 3               # the full system
python src/bench_unified.py --months 3 --no-network  # the ablation
```

---

[← Back to the README](../README.md) · [Docs index](./README.md)
