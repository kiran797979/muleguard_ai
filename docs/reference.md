# Reference

Pipeline stages, repository layout, limitations, and how to reproduce every number.


---

## Pipeline stages

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
| `graph_demo.py` | Synthetic-ledger demonstration of the graph stage ([Networks](./network-detection.md)) |
| `temporal.py` | Suspicious-window localisation and temporal IoU ([The whole system](./end-to-end.md)) |
| `temporal_demo.py` | Synthetic demonstration of window recovery |
| `plots.py` | PR / ROC / calibration curves |
| `paper_figures.py`, `paper_fig_method.py`, `paper_fig_ablation.py` | Publication figures |
| `make_paper.py`, `make_paper_tex.py` | Build the `.docx` and LaTeX manuscripts |
| `make_synthetic.py` | Generate a correctly-shaped fake dataset, including alien schemas |

---

## Layout

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
    ├── MuleGuard_IEEE_Paper_v3_current.docx
    ├── figures/              # publication figures (PNG + PDF)
    ├── demo_graph/           # synthetic graph demonstration
    ├── exports/              # integration contract (alerts are git-ignored)
    └── 01_clean_report.json … 10_operating_metrics.json   (git-ignored)
```

---

## Honest limitations

- **The benchmark is contaminated** (see [Data integrity](./data-integrity.md)).
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

## Reproducing

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

---

[← Back to the README](../README.md) · [Docs index](./README.md)
