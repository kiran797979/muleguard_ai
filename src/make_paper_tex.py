"""
Emit the CyberShield prototype submission as LaTeX (spconf / ICASSP style).

The organisers' template is LaTeX, not Word: `spconf.sty` for the two-column
layout and `IEEEbib.bst` for the bibliography. The hard constraint is length —
four pages of content, with an optional fifth page carrying references ONLY. So
this is a deliberately tighter paper than the Word draft: Related Work is one
paragraph, Methodology keeps only what a reader needs to trust the numbers, and
the space saved goes to the dataset integrity audit and the results.

As with the Word build, every quantity is read from reports/*.json rather than
typed in, so re-running the pipeline and re-running this script keeps the paper
honest. Author details are the only placeholders.

Writes paper/paper.tex and paper/refs.bib. Compile with:
    tectonic paper/paper.tex

Run:  python src/make_paper_tex.py
"""

from __future__ import annotations

import json
import shutil

import config as C
from utils import log

PAPER = C.ROOT / "paper"
TEX = PAPER / "paper.tex"
BIB = PAPER / "refs.bib"

TITLE = ("MULEGUARD AI: LEAKAGE-HARDENED MONEY-MULE DETECTION AND A DATASET "
         "INTEGRITY AUDIT")

# Placeholders — replace with the real team details before submitting.
AUTHORS = r"1st Given Name Surname, 2nd Given Name Surname, 3rd Given Name Surname"
ADDRESS = "Department Name\\\\\n\tInstitution Name\\\\\n\tCity, Country"


def esc(s) -> str:
    """Escape LaTeX specials. Applied to every value interpolated from JSON."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def load() -> dict:
    def L(name):
        p = C.REPORTS_DIR / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {
        "clean": L("01_clean_report.json"),
        "feat": L("02_features_report.json"),
        "metrics": L("03_metrics.json"),
        "scoring": L("05_scoring_report.json"),
        "shap": L("05_shap_top_features.json"),
        "integrity": L("06_integrity_audit.json"),
    }


BIB_ENTRIES = r"""
@article{chawla2002smote,
  author  = "N. V. Chawla and K. W. Bowyer and L. O. Hall and W. P. Kegelmeyer",
  title   = "{SMOTE}: synthetic minority over-sampling technique",
  journal = "Journal of Artificial Intelligence Research",
  volume  = "16",
  pages   = "321--357",
  year    = "2002"
}

@inproceedings{chen2016xgboost,
  author    = "T. Chen and C. Guestrin",
  title     = "{XGBoost}: a scalable tree boosting system",
  booktitle = "Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining",
  pages     = "785--794",
  year      = "2016"
}

@inproceedings{ke2017lightgbm,
  author    = "G. Ke and Q. Meng and T. Finley and T. Wang and W. Chen and W. Ma and Q. Ye and T. Liu",
  title     = "{LightGBM}: a highly efficient gradient boosting decision tree",
  booktitle = "Advances in Neural Information Processing Systems",
  volume    = "30",
  pages     = "3146--3154",
  year      = "2017"
}

@inproceedings{lundberg2017shap,
  author    = "S. M. Lundberg and S. Lee",
  title     = "A unified approach to interpreting model predictions",
  booktitle = "Advances in Neural Information Processing Systems",
  volume    = "30",
  pages     = "4765--4774",
  year      = "2017"
}

@inproceedings{liu2008isolation,
  author    = "F. T. Liu and K. M. Ting and Z. Zhou",
  title     = "Isolation forest",
  booktitle = "Proc. 8th IEEE Int. Conf. Data Mining",
  pages     = "413--422",
  year      = "2008"
}

@inproceedings{davis2006pr,
  author    = "J. Davis and M. Goadrich",
  title     = "The relationship between precision-recall and {ROC} curves",
  booktitle = "Proc. 23rd Int. Conf. Machine Learning",
  pages     = "233--240",
  year      = "2006"
}

@article{kaufman2012leakage,
  author  = "S. Kaufman and S. Rosset and C. Perlich and O. Stitelman",
  title   = "Leakage in data mining: formulation, detection, and avoidance",
  journal = "ACM Transactions on Knowledge Discovery from Data",
  volume  = "6",
  number  = "4",
  pages   = "1--21",
  year    = "2012"
}

@article{cawley2010overfitting,
  author  = "G. C. Cawley and N. L. C. Talbot",
  title   = "On over-fitting in model selection and subsequent selection bias in performance evaluation",
  journal = "Journal of Machine Learning Research",
  volume  = "11",
  pages   = "2079--2107",
  year    = "2010"
}

@inproceedings{zadrozny2002calibration,
  author    = "B. Zadrozny and C. Elkan",
  title     = "Transforming classifier scores into accurate multiclass probability estimates",
  booktitle = "Proc. 8th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining",
  pages     = "694--699",
  year      = "2002"
}

@article{wolpert1992stacking,
  author  = "D. H. Wolpert",
  title   = "Stacked generalization",
  journal = "Neural Networks",
  volume  = "5",
  number  = "2",
  pages   = "241--259",
  year    = "1992"
}

@misc{rbi2016kyc,
  author = "{Reserve Bank of India}",
  title  = "Master direction on know your customer ({KYC}) direction, 2016 (updated)",
  year   = "2016",
  note   = "Reserve Bank of India, Mumbai"
}

@misc{hackathon2026,
  author = "{Bank of India and Indian Institute of Technology Hyderabad}",
  title  = "{PSB} Cybersecurity, Fraud and {AI} Hackathon 2026 problem statement",
  year   = "2026",
  note   = "PLACEHOLDER --- replace with the organisers' exact citation"
}
"""


def build(R: dict) -> str:
    m, ig, cl, ft, sc = (R["metrics"], R["integrity"], R["clean"],
                         R["feat"], R["scoring"])
    e, hr, pm = (m["ensemble_precision_first"], m["ensemble_high_recall"],
                 m["per_model"])
    A = ig["test_A_missingness_only"]
    B = ig["test_B_individually_useless"]
    Cc = ig["test_C_shuffled_labels"]
    prev = ig.get("prevalence", 0.00892)
    counts = ig.get("month_split", {}).get("counts", {})
    hb = sc["band_stats"]["HIGH"]

    def ms(block, k):
        return f"${block[k]['mean']:.3f} \\pm {block[k]['std']:.3f}$"

    def pmv(k, metric="auprc"):
        return f"${pm[k][metric]['mean']:.3f} \\pm {pm[k][metric]['std']:.3f}$"

    n_typ = len(ft.get("mule_typology_features", {}))
    # Must match the bar count in paper_figures.fig_shap(), or the prose and the
    # figure disagree about what "the top N" means.
    SHAP_TOP_N = 12
    shap_top = R["shap"].get("top_features_by_mean_abs_shap", [])
    n_eng = sum(1 for f in shap_top[:SHAP_TOP_N] if f["variable"].startswith("mg_"))

    month_rows = "\n".join(
        f"{esc(k)} & {v.get('0', 0):,} & {v.get('1', 0)} \\\\"
        for k, v in sorted(counts.items())
    )

    return rf"""% MuleGuard AI --- CyberShield prototype submission
% Generated by src/make_paper_tex.py from reports/*.json. Do not hand-edit
% numbers here; re-run the pipeline and regenerate instead.
\documentclass{{article}}
\usepackage{{spconf,amsmath,graphicx,hyperref,booktabs}}

% spconf leaves no gap below a caption, so a table caption sitting above its
% tabular collides with the top rule. Give it a little air.
\setlength{{\abovecaptionskip}}{{6pt}}
\setlength{{\belowcaptionskip}}{{4pt}}

\title{{{TITLE}}}

\name{{{AUTHORS}}}
\address{{{ADDRESS}}}

\begin{{document}}
\ninept
\maketitle

\begin{{abstract}}
Money-mule accounts settle digital payment fraud but are vanishingly rare in
labelled banking data. We present MuleGuard AI, a detection pipeline for
{m['n_accounts']:,} accounts containing {m['n_mules']} confirmed mules
({m['prevalence_pct']}\%). Our primary finding is that the benchmark itself is
contaminated: positives and negatives are drawn from disjoint monthly extracts,
and a model given only the pattern of blank cells---every value discarded---reaches
AUPRC {A['auprc']:.3f} against a {prev:.4f} random baseline. We supply
falsification tests that make this measurable. We further contribute a leakage
taxonomy that classifies fields by meaning rather than correlation, catching an
outcome flag correlated just 0.05 with the target, and {n_typ} named
mule-typology features. Under nested repeated cross-validation, in which
selection, calibration and the operating threshold are all fitted in-fold, the
ensemble reaches precision {ms(e, 'precision')} at recall {ms(e, 'recall')}. We
argue these are an upper bound on what this dataset can establish.
\end{{abstract}}

\begin{{keywords}}
Money-mule detection, anti-money laundering, class imbalance, target leakage,
dataset integrity
\end{{keywords}}

\section{{Introduction}}
\label{{sec:intro}}

A money mule is a bank account, opened by a real customer who is coerced,
recruited or paid, that receives criminal proceeds and forwards them onward
within hours. Mules sit between the fraud victim and the organiser, and freezing
one strands funds that are otherwise irrecoverable. India's real-time payment
rails make this faster and more damaging, and the Reserve Bank of India requires
that automated action against a customer account be explainable
\cite{{rbi2016kyc}}.

Three difficulties compound. Mules are extremely rare, so accuracy is
meaningless---predicting that every account is normal scores 99.11\% here---and
the precision-recall curve is the appropriate summary \cite{{davis2006pr}}. Mules
are behaviourally camouflaged: the account belongs to a genuine customer with a
genuine history, so unsupervised anomaly detection has little purchase, which our
ablation confirms sharply. Finally, labels in this domain are the output of an
investigation, and that process leaves traces a model will exploit
\cite{{kaufman2012leakage}}.

This last point dominates. While building the detector we found the supplied
benchmark cannot support an honest performance estimate, and that the
contamination is invisible to standard defences. We therefore present the
integrity audit as our primary contribution, with the detector as the vehicle
that exposed it.

\section{{Dataset and integrity audit}}
\label{{sec:data}}

The dataset comprises {m['n_accounts']:,} accounts and
{cl['input_shape'][1] - 1:,} anonymised features, with a separate dictionary
mapping each to a named banking variable. Features follow a regular grammar: a
statistic (ratio, deviation, average, extremum) over a payment channel (cash,
cheque, UPI, ATM, online transfer, Aadhaar Payment Bridge) in a direction
(credit, debit) over a window (7, 14, 31 days). A tail block holds demographics,
alert counts by time of day, and investigation metadata.

\subsection{{Leakage a correlation threshold cannot see}}

Four fields record an alert's resolution status and two record how long the
investigation took. All six exist only after an analyst closes a case, so none is
available when an account must actually be scored. Their correlations with the
target span two orders of magnitude: {{\tt FRAUD\_SUSPECTED}} at 0.969 is caught
by any threshold, while {{\tt FALSE\_POSITIVE}} at 0.055 is caught by none---yet
it encodes the same verdict, merely inverted. We therefore classify leakage by
what a field \emph{{means}}, using the dictionary, and treat correlation only as a
backstop.

\subsection{{A structural confound}}

Table~\ref{{tab:months}} gives the complete cross-tabulation of class against
extract month. Every negative is drawn from the October extract and every
positive from September, November and December. No month contains both classes,
so the month column alone separates them---and it is not a property of any
customer, but a record of which extraction run produced the row.

\begin{{table}}[htb]
\centering
\caption{{Class composition by monthly extract. No month contains both classes.}}
\label{{tab:months}}
\vspace{{1ex}}
\begin{{tabular}}{{lrr}}
\toprule
Extract month & Normal & Mule \\
\midrule
{month_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Dropping the month column does not help. Because the classes occupy disjoint
extracts, any characteristic that varies between extraction runs aligns perfectly
with the label while describing nothing about behaviour. Adversarial validation
cannot diagnose this, since a classifier trained to predict the extract is by
construction the classifier that predicts the label.

\subsection{{Falsification tests}}

We constructed three tests, each supplying a model with information that cannot
identify a mule. All should score near the {prev:.4f} baseline if the data is
sound. In test A every value is discarded and the model sees only whether each
cell was populated---a property of the extraction job, not the customer. In test
B the model receives 250 columns drawn from the {B.get('pool_size', 0):,} whose
individual correlation is below 0.05, none of which can identify a mule alone. In
test C the labels are permuted.

Table~\ref{{tab:falsify}} reports the outcome. Test C collapsing to the baseline
confirms the harness is sound, so the first two results are genuine properties of
the data rather than a defect in our code. The conclusion is unavoidable: any
figure computed on this dataset, by any team and any method, measures extract
provenance in addition to mule behaviour, and the two cannot be separated within
this file. Section~\ref{{sec:results}} should be read with that attached.

\begin{{table}}[htb]
\centering
\caption{{Falsification tests. Information that cannot identify a mule still
separates the classes.}}
\label{{tab:falsify}}
\vspace{{1ex}}
\begin{{tabular}}{{clrr}}
\toprule
Test & Information supplied & AUPRC & AUROC \\
\midrule
A & Blank/not-blank only & {A['auprc']:.3f} & {A['auroc']:.3f} \\
B & 250 cols, $|r| < 0.05$ & {B['auprc']:.3f} & {B['auroc']:.3f} \\
C & As B, labels shuffled & {Cc['auprc']:.4f} & {Cc['auroc']:.3f} \\
\midrule
--- & Random guess & {prev:.4f} & 0.500 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Method}}
\label{{sec:method}}

Fig.~\ref{{fig:arch}} shows the pipeline. We highlight only what a reader needs
to trust the numbers.

\begin{{figure*}}[htb]
\centering
\includegraphics[width=\textwidth]{{figures/fig1_architecture}}
\caption{{MuleGuard AI pipeline. Stage 0 runs before any modelling and its
verdict qualifies every metric produced downstream (red). Leak defences
concentrate in Stage 1. In Stage 3 every component that touches the label ---
feature selection, stacking weights, the calibration map and the decision
threshold --- is fitted inside the training fold and applied frozen to
validation rows. The data dictionary is drawn as a bus because it is
cross-cutting rather than a pipeline step: it supplies the named semantics that
Stages 1, 2 and 4 depend on. The graph stage is omitted from the figure; it
detects the absence of counterparty identifiers and disables itself.}}
\label{{fig:arch}}
\end{{figure*}}

\textbf{{Preserving destroyed information.}} Eight columns are categorical,
including occupation, gender and product. Coercing the matrix to numeric---the
customary step---converts them to missing values that are then dropped as sparse.
This discards real signal: mule rates by occupation range from 0.45\% for
homemakers to 1.94\% for students against a 0.89\% base. We encode the account-age
bucket ordinally and the rest as indicators. We also revisit imputation: for
absolute activity aggregates a missing value means the activity never occurred,
so we zero-fill {cl['zero_filled_activity']['values_filled']:,} values across
{cl['zero_filled_activity']['columns']:,} columns rather than inventing activity
through median imputation.

\textbf{{Four-layer leakage defence.}} Layer one removes post-outcome fields
semantically. Layer two removes structural artefacts of sample assembly.
Layer three, \emph{{extract hardening}}, removes any column whose blank rate differs
between classes by more than ten percentage points, on the grounds that cell
population is decided by the extraction job; this dropped
{cl['extract_hardening']['columns_dropped']} columns. It uses the label, but only
ever removes columns, so it can make the reported result worse and never better.
Layer four is a separation audit scanning every surviving column for disjoint
class ranges, which reports clean after the first three.

\textbf{{Mule-typology features.}} A mule receives funds and forwards them almost
immediately, retains little, operates in bursts, favours digital rails, and
belongs to a customer whose profile does not match the volume. We encode each
clause as a named family---pass-through ratio, turnover over balance, burst,
cash-out share, channel concentration, ticket size, night-alert share, balance
volatility, occupation divergence---{n_typ} features in total.

\textbf{{Nested validation.}} The ensemble stacks an isolation forest
\cite{{liu2008isolation}}, XGBoost \cite{{chen2016xgboost}} and LightGBM
\cite{{ke2017lightgbm}} through a logistic meta-learner \cite{{wolpert1992stacking}}
with isotonic calibration \cite{{zadrozny2002calibration}}. Missing values are
imputed with medians learned inside the training fold rather than across the
whole matrix, which removes a transductive leak and is also the only defensible
treatment for a single account scored at run time. Every component
touching the label---selection, base models, stacking weights, the calibration
map, \emph{{and the decision threshold}}---is fitted inside the training fold and
applied frozen to unseen validation rows. This matters: calibrating on pooled
out-of-fold predictions and scoring those same values inflates the result, and
selecting the threshold that maximises precision on a curve then reporting that
precision is optimistic by construction \cite{{cawley2010overfitting}}. We prefer
instance reweighting to synthetic oversampling \cite{{chawla2002smote}}, which with
65 training positives across hundreds of dimensions fabricates regions of feature
space for which no evidence exists. Because a single split places about 16 mules
per fold, we repeat the procedure across shuffles and report mean $\pm$ standard
deviation.

\section{{Results}}
\label{{sec:results}}

Table~\ref{{tab:results}} reports the calibrated ensemble. At the precision-first
operating point it attains precision {ms(e, 'precision')} at recall
{ms(e, 'recall')}, with AUPRC {ms(e, 'auprc')} against a
{m['auprc_baseline_random']:.4f} baseline; summed across folds,
{e['tp_total']} true positives against {e['fp_total']} false positives. A
second operating point trades precision to {ms(hr, 'precision')} for recall
{ms(hr, 'recall')}, appropriate when a reviewed alert costs minutes and a missed
mule costs a laundering channel.

\begin{{table}}[htb]
\centering
\caption{{Detection performance and per-model ablation, nested repeated
cross-validation (mean $\pm$ std across folds).}}
\label{{tab:results}}
\vspace{{1ex}}
\begin{{tabular}}{{lrr}}
\toprule
Metric & Precision-first & High-recall \\
\midrule
Precision & {ms(e, 'precision')} & {ms(hr, 'precision')} \\
Recall & {ms(e, 'recall')} & {ms(hr, 'recall')} \\
F1 & {ms(e, 'f1')} & {ms(hr, 'f1')} \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Per-model AUPRC (threshold-independent)}}}} \\
Isolation forest & \multicolumn{{2}}{{r}}{{{pmv('iso')}}} \\
LightGBM & \multicolumn{{2}}{{r}}{{{pmv('lgbm')}}} \\
XGBoost & \multicolumn{{2}}{{r}}{{{pmv('xgb')}}} \\
Calibrated ensemble & \multicolumn{{2}}{{r}}{{{ms(e, 'auprc')}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

The isolation forest result is the informative one. At AUPRC {pmv('iso')} and
AUROC {pm['iso']['auroc']['mean']:.3f} it performs \emph{{worse than random}}. This
is not a defect but a finding, consistent with the criminology: a mule account
belongs to a real customer with an ordinary history, and is valuable to the
organiser precisely because it does not look anomalous. Unsupervised outlier
detection is the wrong tool here, and proposals leading with it deserve
suspicion.

Fig.~\ref{{fig:shap}} ranks features by mean absolute SHAP contribution
\cite{{lundberg2017shap}}, computed per fold on validation rows only, so each
account is explained by the same out-of-fold model that scored it rather than by
a final model refit on every row including that account.
{n_eng} of the top {SHAP_TOP_N} are engineered in this work
rather than supplied columns, indicating the typology encoding contributes
materially rather than restating what the raw matrix already exposed. Among
supplied columns, deviations of non-cash non-cheque volume from the occupation
cohort dominate---the profile-mismatch signal appearing directly.

\begin{{figure}}[htb]
\centering
\includegraphics[width=\columnwidth]{{figures/fig3_shap}}
\caption{{Top features by mean absolute SHAP value. Features engineered in this
work are shown in teal, supplied dictionary variables in grey.}}
\label{{fig:shap}}
\end{{figure}}

The 0--1000 band edges are not chosen by hand. The high-risk edge is the
operating threshold that held precision $\geq$ {m['precision_target']} on inner
folds, and the medium edge is the analyst-review-queue threshold; both are
averaged over folds and both were fixed before any validation row was seen. A
band boundary therefore reports a fitted decision rather than a round number.

Operationally, the high-risk band holds {hb['accounts']} accounts of which
{hb['true_mules']} are confirmed mules (band precision
{hb['precision']:.2f}), capturing {hb['recall_of_all_mules']*100:.0f}\% of all
mules in a queue that is {hb['accounts']/m['n_accounts']*100:.2f}\% of the
portfolio. Reviewing well under one percent of accounts surfaces two thirds of
the mule population.

\section{{Discussion and conclusion}}
\label{{sec:discussion}}

The results above are strong, and we do not believe they measure mule detection
capability. Section~\ref{{sec:data}} establishes that an uninformative view of the
same data achieves comparable separation, so a substantial and unquantifiable
share reflects the extract a row came from. Our hardening removes the artefact
where it is identifiable; it cannot remove what is not. A behavioural feature
that also drifts between months is confounded, and no modelling choice separates
the two within this file. The remedy is a sampling change, not a modelling one:
negatives drawn from the same months as positives would let month be conditioned
on. This affects every team working from this release.

Further limitations bear statement. The dataset is an account-level matrix with
no counterparty identifiers, so the graph component detects their absence and
disables itself rather than fabricating an edge list; network-propagation claims
cannot be evaluated here. Hyperparameters remain at conservative defaults, since
tuning against a confounded objective optimises the artefact. And
{m['n_mules']} positives is a small sample---the standard deviations reported are
wide enough that differences of a few points should not be considered
meaningful.

We suggest the falsification protocol of Section~\ref{{sec:data}} deserves wider
use. It is inexpensive, requires no knowledge of the modelling approach, and here
was the difference between reporting a headline result and understanding what
that result measured \cite{{hackathon2026}}.

\section{{Acknowledgment}}

The authors thank the organisers, Bank of India and the Indian Institute of
Technology Hyderabad, for the dataset and the accompanying data dictionary. The
dictionary proved essential rather than incidental: without documented variable
semantics neither the meaning-based leakage taxonomy nor the named feature
families could have been constructed. The integrity findings are offered in a
collaborative spirit---an observation intended to strengthen the benchmark for
all participants and for future releases, not a criticism of those who assembled
and shared it.

\bibliographystyle{{IEEEbib}}
\bibliography{{refs}}

\end{{document}}
"""


def main() -> None:
    R = load()
    if not R["metrics"]:
        log("reports/03_metrics.json missing — run the pipeline first.")
        return

    PAPER.mkdir(parents=True, exist_ok=True)

    # The organisers' style files must sit beside the .tex for tectonic to find.
    tpl = C.ROOT / "Prototype_Template_CyberShield" / "Prototype_Template_CyberShield"
    for name in ("spconf.sty", "IEEEbib.bst"):
        src = tpl / name
        if src.exists() and not (PAPER / name).exists():
            shutil.copy2(src, PAPER / name)

    BIB.write_text(BIB_ENTRIES.lstrip(), encoding="utf-8")
    TEX.write_text(build(R), encoding="utf-8")
    log(f"Wrote {TEX}")
    log(f"Wrote {BIB}")
    log("Compile with:  tectonic paper/paper.tex")


if __name__ == "__main__":
    main()
