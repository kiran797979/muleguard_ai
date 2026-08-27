"""
Stage 13 — Disparate-impact audit.

A bank cannot deploy a model that freezes customer accounts without showing it
does not fall disproportionately on one kind of customer. That is a supervisory
expectation, not a nice-to-have, and it is the question this stage answers.

What this audit CANNOT do, stated first
---------------------------------------
The supplied extract contains no protected attribute. There is no gender, no
age, no occupation code, no branch, no district, no income band. Every one of
the 3,924 variables is either a behavioural aggregate over the account's own
activity or a deviation from an occupation cohort whose identity is not
included. The 296 `*_OCC` columns are the deviation, not the cohort.

So a fairness audit by protected class is impossible on this data, and any
number claiming otherwise would be invented. What is possible, and what a
supervisor would actually ask about first in Indian retail banking, is
disparate impact on two proxies that ARE present:

  BALANCE BAND      a wealth proxy. If the model flags low-balance customers at
                    several times the rate of high-balance ones, it is a
                    poverty detector wearing a fraud detector's clothes.
  CASH DEPENDENCE   share of activity conducted in cash. Cash-heavy customers
                    skew rural, older, informal-sector and unbanked-adjacent.
                    A model that treats cash as inherently suspicious will
                    systematically over-flag exactly the population a public
                    sector bank exists to serve.

Both are proxies and are reported as proxies. The test a bank must run before
deployment, on data it holds and we do not, is named in the output.

How to read the numbers
-----------------------
Two measures per slice, because they answer different questions:

  SELECTION RATE   share of the slice placed in an actionable band. This is
                   the burden the model imposes on that group.
  PRECISION        share of those flags that were real mules. This is whether
                   the burden was justified.

A high selection rate is only a fairness problem if precision falls with it.
Flagging a group more often AND being right less often is the signature of a
model that has learned a proxy rather than a behaviour.

The reference point is the four-fifths rule, the long-standing US EEOC
convention and the most widely recognised disparate-impact screen: a selection
rate below 80% of the most-selected group's rate is treated as evidence of
adverse impact. It is a screen, not a verdict, and it is applied here to
proxies rather than protected classes, so it is reported as a flag for review.

Run:  python src/11_fairness.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config as C
import dictionary as D
import roles as R
from utils import load_frame, log, save_json

ACTIONABLE = ("MEDIUM", "HIGH")

# The four-fifths rule. A screen for review, not a legal finding.
IMPACT_RATIO_FLOOR = 0.80


def _first_by_role(df: pd.DataFrame, want: R.Role) -> str | None:
    """A column carrying this meaning, whatever this dataset calls it."""
    idx = R.RoleIndex(df.columns, label_of=D.real_name)
    return idx.find(want)


def _cash_amount_column(df: pd.DataFrame) -> str | None:
    """A real cash-amount column. `NON_CASH` is the negation and must not match."""
    best = None
    for c in df.columns:
        n = str(D.real_name(c)).upper()
        if "NON_CASH" in n or "CASH" not in n or "AMT" not in n:
            continue
        if re.search(r"^(TOT|SUM)_CASH_AMT", n):
            return c
        if best is None and not n.startswith(("MIN_", "MAX_", "R_", "D_")):
            best = c
    return best


def _quantile_bands(s: pd.Series, labels: list[str]) -> pd.Series | None:
    """Split into bands of comparable size, without letting ties invent order.

    The first version ranked with method="first" and cut into quartiles. On a
    column where most rows share one value that is silently catastrophic: 76% of
    accounts have zero cash share, so the tie-break fell back to ROW ORDER, and
    this file is ordered by extract month. The "most cash" quartile came out
    holding all 81 mules and none of the negatives — a perfect separation that
    measured the extract, not the customer, and would have been reported as a
    finding about cash usage.

    So a dominant value is given its own band and the rest are cut inside the
    remainder. If what is left is too small or too flat to cut, the slice is
    refused rather than fabricated.
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < len(labels) * 20:
        return None

    mode_share = s.value_counts(normalize=True, dropna=True)
    if len(mode_share) == 0:
        return None
    top_val, top_frac = mode_share.index[0], float(mode_share.iloc[0])

    if top_frac < 0.15:
        if s.nunique(dropna=True) < len(labels):
            return None
        try:
            return pd.qcut(s.rank(method="first"), q=len(labels), labels=labels)
        except ValueError:
            return None

    # A dominant value: keep it whole, cut only what is genuinely spread out.
    rest = s[(s != top_val) & s.notna()]
    inner = [l for l in labels[1:]]
    if len(rest) < 20 * len(inner) or rest.nunique() < len(inner):
        inner = inner[:2] if len(rest) >= 40 and rest.nunique() >= 2 else []
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out[s == top_val] = labels[0]
    if inner:
        try:
            cuts = pd.qcut(rest.rank(method="first"), q=len(inner), labels=inner)
            out.loc[rest.index] = cuts.astype(object)
        except ValueError:
            out.loc[rest.index] = inner[-1]
    else:
        out.loc[rest.index] = "any activity"
    return out.astype("category")


def _slice_report(groups: pd.Series, band: pd.Series, y: pd.Series) -> dict:
    """Selection rate and precision for every value of `groups`.

    Direction matters here. The four-fifths rule was written for selection into
    a benefit, where being chosen less often is the harm. Being flagged by this
    model is a burden — a review queue or a frozen account — so the group that
    needs scrutiny is the one selected MOST, measured against the group selected
    least. A group carrying more than 1/0.8 = 1.25x the least-burdened group's
    rate is flagged for review.

    A higher rate alone is not evidence of unfairness: if a group genuinely
    contains more mules, flagging it more is correct. The signature of a model
    that has learned a proxy is a higher selection rate together with LOWER
    precision — flagged more often, right less often.
    """
    rows = []
    flagged = band.isin(ACTIONABLE)
    for name, idx in groups.groupby(groups, observed=True).groups.items():
        g_flag, g_y = flagged.loc[idx], y.loc[idx]
        n_flag = int(g_flag.sum())
        rows.append({
            "group": str(name),
            "accounts": int(len(idx)),
            "true_mules": int(g_y.sum()),
            "base_rate": round(float(g_y.mean()), 5),
            "flagged": n_flag,
            "selection_rate": round(n_flag / max(len(idx), 1), 4),
            "precision": round(float(g_y.loc[g_flag.loc[g_flag].index].mean()), 4)
            if n_flag else None,
        })

    sized = [r for r in rows if r["accounts"] >= 20]
    rates = [r["selection_rate"] for r in sized]
    floor = min(rates) if rates else 0.0
    ceiling = max(rates) if rates else 0.0
    burden_cap = 1 / IMPACT_RATIO_FLOOR

    for r in rows:
        r["burden_ratio"] = (round(r["selection_rate"] / floor, 3)
                             if floor > 0 and r["accounts"] >= 20 else None)
        r["flagged_for_review"] = bool(
            r["burden_ratio"] is not None and r["burden_ratio"] > burden_cap)

    heaviest = max(sized, key=lambda r: r["selection_rate"], default=None)
    precisions = [r["precision"] for r in sized if r["precision"] is not None]
    mean_prec = float(np.mean(precisions)) if precisions else None

    # The finding that would actually matter: most-burdened AND least accurate.
    proxy_signature = bool(
        heaviest and mean_prec is not None and heaviest["precision"] is not None
        and heaviest["burden_ratio"] and heaviest["burden_ratio"] > burden_cap
        and heaviest["precision"] < mean_prec)

    return {
        "groups": rows,
        "least_burdened_rate": round(floor, 4),
        "most_burdened_rate": round(ceiling, 4),
        "selection_rate_spread": (round(ceiling / floor, 2) if floor > 0 else None),
        "spread_note": "How many times more often the most-flagged band is "
                       "flagged than the least-flagged one.",
        "burden_ratio_cap": round(burden_cap, 3),
        "most_burdened_group": heaviest["group"] if heaviest else None,
        "any_group_over_cap": any(r["flagged_for_review"] for r in rows),
        "mean_precision_across_groups": (round(mean_prec, 4)
                                         if mean_prec is not None else None),
        "proxy_signature_present": proxy_signature,
        "proxy_signature_note": "True only if the most-burdened group is ALSO "
                                "flagged less accurately than average, which is "
                                "what a model that learned a proxy looks like.",
    }


def main() -> None:
    df = load_frame(C.FEATURES_PARQUET)
    scores = pd.read_csv(C.DATA_DIR / "risk_scores.csv")
    if len(scores) != len(df):
        raise SystemExit(f"risk_scores has {len(scores)} rows, features has {len(df)}")

    band = scores["band"].astype(str)
    y = scores["y_true"].astype(int)
    report: dict = {
        "what_this_can_and_cannot_test": {
            "protected_attributes_in_this_extract": 0,
            "checked_for": ["gender", "age", "occupation code", "branch",
                            "district", "income band"],
            "why": "Every supplied variable is a behavioural aggregate over the "
                   "account's own activity, or a deviation from an occupation "
                   "cohort whose identity is not included. A fairness audit by "
                   "protected class is therefore impossible on this data, and "
                   "any such number would be invented.",
            "what_is_tested_instead": "Disparate impact on two proxies that are "
                                      "present and that a supervisor would ask "
                                      "about first: wealth (balance) and cash "
                                      "dependence.",
            "what_the_bank_must_run_before_deployment":
                "The same selection-rate and precision slices, computed on "
                "gender, age band, occupation code and branch district, which "
                "the bank holds in its CRM and we do not. The code path is "
                "identical; only the grouping column changes.",
        },
        "method": {
            "actionable_bands": list(ACTIONABLE),
            "selection_rate": "share of the group placed in an actionable band",
            "precision": "share of those flags that were confirmed mules",
            "impact_ratio_floor": IMPACT_RATIO_FLOOR,
            "floor_note": "Four-fifths rule, the standard disparate-impact "
                          "screen. Applied here to proxies, so it flags for "
                          "review rather than establishing a finding.",
            "reading": "A high selection rate is only a fairness problem if "
                       "precision falls with it. Flagging a group more often "
                       "AND being right less often is the signature of a model "
                       "that learned a proxy rather than a behaviour.",
        },
    }

    # --- wealth proxy: average balance ------------------------------------
    bal_col = _first_by_role(df, R.Role(measure="balance", stat="AVG"))
    if bal_col is None:
        bal_col = _first_by_role(df, R.Role(measure="balance"))
    if bal_col is not None:
        bands = _quantile_bands(df[bal_col],
                                ["lowest balance", "Q2", "Q3", "highest balance"])
        if bands is not None:
            report["by_balance_quartile"] = {
                "column_used": f"{bal_col} ({D.real_name(bal_col)})",
                "proxy_for": "customer wealth",
                **_slice_report(bands, band, y),
            }
            log(f"Balance slice built from {D.real_name(bal_col)}")
    if "by_balance_quartile" not in report:
        report["by_balance_quartile"] = {"available": False,
                                         "reason": "no balance column resolved"}

    # --- cash dependence ---------------------------------------------------
    # R.Role(channel="CASH") matches R_NON_CASH_CHQ_AMT_L7_14D, because the CASH
    # vocabulary hits the substring inside NON_CASH and would have inverted the
    # meaning of this whole slice. Resolve it by name, excluding the negation.
    cash = _cash_amount_column(df)
    total = _first_by_role(df, R.Role(measure="amount", stat="TOT"))
    if cash is not None and total is not None:
        share = (pd.to_numeric(df[cash], errors="coerce").abs()
                 / (pd.to_numeric(df[total], errors="coerce").abs() + 1.0))
        bands = _quantile_bands(share, ["no cash activity", "some cash", "more cash", "most cash"])
        if bands is not None:
            report["by_cash_dependence"] = {
                "columns_used": [f"{cash} ({D.real_name(cash)})",
                                 f"{total} ({D.real_name(total)})"],
                "proxy_for": "rural, older and informal-sector customers, who "
                             "transact in cash more than the branch average",
                **_slice_report(bands, band, y),
            }
            log(f"Cash-dependence slice built from {D.real_name(cash)}")
    if "by_cash_dependence" not in report:
        report["by_cash_dependence"] = {"available": False,
                                        "reason": "no cash/total amount columns resolved"}

    flagged_any = [k for k in ("by_balance_quartile", "by_cash_dependence")
                   if report[k].get("any_group_over_cap")]
    proxy_hits = [k for k in ("by_balance_quartile", "by_cash_dependence")
                  if report[k].get("proxy_signature_present")]
    report["verdict"] = {
        "slices_run": [k for k in ("by_balance_quartile", "by_cash_dependence")
                       if report[k].get("groups")],
        "slices_where_a_group_exceeds_the_burden_cap": flagged_any,
        "slices_showing_the_proxy_signature": proxy_hits,
        "summary": (
            "No group is flagged disproportionately relative to the least-"
            "flagged group on either proxy."
            if not flagged_any else
            "A group carries more than 1.25x the least-flagged group's rate in: "
            + ", ".join(flagged_any) + ". "
            + ("Precision does not fall with it in any slice, so this reflects a "
               "genuine difference in mule prevalence rather than a proxy the "
               "model has learned."
               if not proxy_hits else
               "Precision ALSO falls in: " + ", ".join(proxy_hits) + ". That is "
               "the proxy signature and must be investigated before deployment.")),
        "not_a_clearance": "This tests proxies, not protected classes. It is "
                           "evidence the model is not obviously a wealth or "
                           "cash detector. It is not a fairness clearance, and "
                           "we would not deploy without the protected-class "
                           "slices named above.",
    }

    save_json(report, C.REPORTS_DIR / "11_fairness_audit.json")
    log(f"Fairness audit written. Over-cap: {flagged_any or 'none'} | "
        f"proxy signature: {proxy_hits or 'none'}")


if __name__ == "__main__":
    main()
