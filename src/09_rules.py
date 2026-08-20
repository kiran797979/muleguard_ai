"""
Stage 9 — The rules layer, and an honest measurement of it.

Every bank runs deterministic rules before it runs a model. They are fast,
interpretable, and a compliance officer can sign them off without understanding
gradient boosting. Any serious mule-detection proposal has to include one, so
here it is: twelve rules drawn from published AML money-mule typology.

WHAT MAKES THIS DIFFERENT FROM A RULES LAYER THAT JUST GETS SHIPPED
-------------------------------------------------------------------
The thresholds below are written the way the typology says they should be
written, and are NOT tuned against this dataset. Tuning them here would be the
same mistake as tuning hyperparameters against a confounded target: it would
produce rules that fit this file's artefact and fail everywhere else.

So each rule is stated, run, and reported with whatever precision it actually
achieves. Where a rule underperforms, that is published rather than quietly
dropped or re-tuned until it looks good. A rule that fires on 3,000 accounts to
catch 4 mules is worse than useless in an AML desk, and the only way to know
which rules those are is to measure all of them.

Outputs:
  reports/09_rules_report.json   per-rule precision, recall, lift, and the
                                 combined rule layer against the ML ensemble

Run:  python src/09_rules.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import dictionary as D
import schema as S
from utils import load_frame, log, save_json


# --------------------------------------------------------------------------
# The rule book. (id, plain English, why an AML analyst believes it, predicate)
# --------------------------------------------------------------------------
def build_rules(df: pd.DataFrame):
    """Each predicate returns a boolean mask, or None if its inputs are absent."""

    def col(name: str):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        code = D.resolve(name)
        if code and code in df.columns:
            return pd.to_numeric(df[code], errors="coerce")
        return None

    def rule(fn):
        try:
            m = fn()
            return None if m is None else m.fillna(False).to_numpy(dtype=bool)
        except Exception:  # noqa: BLE001 — a missing input is not a crash
            return None

    R = []

    def add(rid, title, why, fn):
        R.append({"id": rid, "title": title, "why": why, "mask": rule(fn)})

    def _pair(a, b, op):
        x, y = col(a), col(b)
        return None if x is None or y is None else op(x, y)

    add("R01", "Pass-through account",
        "Credits and debits match almost exactly over a week, so the account is "
        "a conduit rather than a wallet.",
        lambda: (lambda v: None if v is None else v >= 0.90)(col("mg_passthrough_7d")))

    add("R02", "Turnover far exceeds balance",
        "Moves many multiples of what it ever holds. A salaried customer does not.",
        lambda: (lambda v: None if v is None else v >= 50)(col("mg_turnover_over_balance_7d")))

    add("R03", "Sudden activation",
        "This week's rupee velocity is at least triple the monthly average. "
        "A dormant account waking up is a classic recruitment signal.",
        lambda: (lambda v: None if v is None else v >= 3.0)(col("mg_amount_burst_7v31")))

    add("R04", "Cash-out dominance",
        "More than half of last week's debits left as physical cash, which breaks "
        "the audit trail.",
        lambda: (lambda v: None if v is None else v >= 0.50)(col("mg_cash_out_share_7d")))

    add("R05", "Single payment rail",
        "Uses exactly one channel and nothing else. Real customers spread across "
        "several; a single-purpose account does not.",
        lambda: (lambda v: None if v is None else v <= 1)(col("mg_channel_active_7d")))

    add("R06", "Structuring by ticket size",
        "Average transaction under five thousand rupees, consistent with splitting "
        "one large sum to stay under reporting thresholds.",
        lambda: (lambda v: None if v is None else (v > 0) & (v <= 5000))(col("mg_avg_ticket_7d")))

    add("R07", "Nocturnal alerting",
        "At least half this account's alerts were raised at night.",
        lambda: (lambda v: None if v is None else v >= 0.50)(col("mg_alert_share_night")))

    add("R08", "Spike and drain balance",
        "Peak-to-trough balance swing is more than twice the average balance.",
        lambda: (lambda v: None if v is None else v >= 2.0)(col("mg_balance_volatility_7d")))

    add("R09", "Profile mismatch",
        "Behaviour sits more than fifty units from the norm for this customer's "
        "occupation cohort.",
        lambda: (lambda v: None if v is None else v >= 50)(col("mg_occ_deviation_max")))

    add("R10", "New account moving money fast",
        "Opened within ninety days and already turning over ten times its balance.",
        lambda: _pair("ACCT_OPN_DAYS", "mg_turnover_over_balance_7d",
                      lambda a, t: (a <= 90) & (t >= 10)))

    add("R11", "Pure conduit",
        "Net flow near zero while throughput is high: everything that came in "
        "went straight back out.",
        lambda: _pair("mg_net_flow_7d", "mg_turnover_over_balance_7d",
                      lambda nf, t: (nf.abs() <= 0.05) & (t >= 5)))

    add("R12", "Conduit on a single rail",
        "Pass-through behaviour AND only one channel in use. The two strongest "
        "typology signals firing together.",
        lambda: _pair("mg_passthrough_7d", "mg_channel_active_7d",
                      lambda p, c: (p >= 0.85) & (c <= 1)))

    return R


# --------------------------------------------------------------------------
def score_mask(mask: np.ndarray, y: np.ndarray) -> dict:
    flagged = int(mask.sum())
    tp = int((mask & (y == 1)).sum())
    prevalence = float(y.mean())
    precision = tp / flagged if flagged else 0.0
    return {
        "accounts_flagged": flagged,
        "mules_caught": tp,
        "false_alarms": flagged - tp,
        "precision": round(precision, 4),
        "recall": round(tp / max(int(y.sum()), 1), 4),
        "lift_over_prevalence": round(precision / prevalence, 2) if prevalence and flagged else 0.0,
        "alerts_per_mule": round(flagged / tp, 1) if tp else None,
    }


def main() -> None:
    df = load_frame(C.FEATURES_PARQUET)
    target = S.bind_target(df, C)
    y = df[target].astype(int).to_numpy()
    prevalence = float(y.mean())
    log(f"Rule layer on {len(y):,} accounts, {int(y.sum())} mules "
        f"(base rate {prevalence:.4%})")

    rules = build_rules(df)
    rows, usable = [], []
    for r in rules:
        if r["mask"] is None:
            rows.append({"id": r["id"], "title": r["title"], "why": r["why"],
                         "status": "NOT EVALUATED — required columns absent"})
            log(f"  {r['id']} {r['title']:<34} inputs unavailable")
            continue
        res = score_mask(r["mask"], y)
        rows.append({"id": r["id"], "title": r["title"], "why": r["why"],
                     "status": "evaluated", **res})
        usable.append(r)
        log(f"  {r['id']} {r['title']:<34} flags {res['accounts_flagged']:>5} "
            f"catches {res['mules_caught']:>3}  precision {res['precision']:.3f}  "
            f"lift {res['lift_over_prevalence']:>6.1f}x")

    out = {
        "base_rate": round(prevalence, 6),
        "n_accounts": int(len(y)),
        "n_mules": int(y.sum()),
        "philosophy": (
            "Thresholds come from published AML money-mule typology and are NOT "
            "tuned against this dataset. Tuning them here would fit this file's "
            "confound and fail elsewhere, which is the same mistake as tuning "
            "hyperparameters against a contaminated target. Every rule is "
            "reported with the precision it actually achieved."
        ),
        "rules": rows,
    }

    # Persist WHICH rules fired for WHICH account. Without this the export layer
    # can only ever say "the model flagged it", which is useless to an AML case
    # manager: a scenario code is what lets an analyst route, prioritise and
    # explain an alert, and what lets a bank report on scenario performance.
    if usable:
        fired = pd.DataFrame({"account_idx": np.arange(len(y))})
        codes = []
        for i in range(len(y)):
            codes.append("|".join(r["id"] for r in usable if r["mask"][i]))
        fired["rules_fired"] = codes
        fired["n_rules_fired"] = [0 if not c else c.count("|") + 1 for c in codes]
        path = C.DATA_DIR / "rule_hits.csv"
        fired.to_csv(path, index=False, encoding="utf-8")
        log(f"Wrote {path}  (per-account scenario codes for the export layer)")

    if usable:
        any_mask = np.zeros(len(y), dtype=bool)
        hits = np.zeros(len(y), dtype=int)
        for r in usable:
            any_mask |= r["mask"]
            hits += r["mask"].astype(int)
        out["combined_any_rule"] = {"description": "at least one rule fires",
                                    **score_mask(any_mask, y)}
        out["combined_two_or_more"] = {"description": "at least two rules fire",
                                       **score_mask(hits >= 2, y)}
        out["combined_three_or_more"] = {"description": "at least three rules fire",
                                         **score_mask(hits >= 3, y)}
        log(f"  COMBINED any rule       flags {out['combined_any_rule']['accounts_flagged']:>5} "
            f"catches {out['combined_any_rule']['mules_caught']:>3}  "
            f"precision {out['combined_any_rule']['precision']:.3f}")
        log(f"  COMBINED 2+ rules       flags {out['combined_two_or_more']['accounts_flagged']:>5} "
            f"catches {out['combined_two_or_more']['mules_caught']:>3}  "
            f"precision {out['combined_two_or_more']['precision']:.3f}")

    # Head-to-head against the learned model, so the comparison is explicit.
    mpath = C.REPORTS_DIR / "03_metrics.json"
    if mpath.exists():
        import json
        m = json.loads(mpath.read_text(encoding="utf-8"))
        e = m["ensemble_precision_first"]
        out["ml_ensemble_for_comparison"] = {
            "precision": e["precision"]["mean"], "recall": e["recall"]["mean"],
            "lift_over_prevalence": e["lift_over_prevalence"]["mean"],
            "note": "Nested out-of-fold. Not directly comparable to the rules, "
                    "which are evaluated on every row because they have no "
                    "fitted parameters to overfit with.",
        }

    save_json(out, C.REPORTS_DIR / "09_rules_report.json")


if __name__ == "__main__":
    main()
