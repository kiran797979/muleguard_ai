# Networks: rings and motifs

Finding structures rather than accounts, where each method breaks, and what an external benchmark said about both.


---

## The graph stage, and why it is switched off

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

## Finding rings, and what a third-party benchmark said about it

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
in [`reports/bench_saml/saml_ring_benchmark.json`](../reports/bench_saml/saml_ring_benchmark.json).

---

[← Back to the README](../README.md) · [Docs index](./README.md)
