"""
MuleGuard AI — integration layer for EFRMS and AML case management systems.

WHAT THIS IS, STATED PRECISELY
------------------------------
This module emits alerts, case packs and an audit trail in a documented,
vendor-neutral schema whose fields map onto the concepts every AML case
management system uses: an alert identifier, the entity under review, a
normalised score, a priority, the scenario that fired, human-readable reasons,
the evidence behind them, and a disposition.

WHAT THIS IS NOT
----------------
It is **not certified compatibility with any named product**. We have not tested
against Oracle FCCM, SAS AML, NICE Actimize, Clari5, Amlock or any other
platform, and we do not have their integration specifications. Anyone claiming
certified compatibility without that testing is guessing.

What we can say honestly: the output is a documented schema, the field mapping
is published below, the API carries an OpenAPI specification, and both JSON and
delimited-file exports are provided because batch file exchange is how most bank
systems actually integrate. Wiring this into a specific EFRMS is a mapping
exercise measured in days, not a rebuild.

STR / SAR NOTE
--------------
The case pack assembles the material an analyst needs to *prepare* a Suspicious
Transaction Report. It does not generate a filing. FIU-IND submission has its own
schema and its own legal responsibilities, and a model must never file on a
human's behalf.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import dictionary as D
from utils import log, save_json

MODEL_ID = "muleguard-ensemble"
SCHEMA_VERSION = "1.0"

# Our rule identifiers mapped to plain typology language. These are OUR
# taxonomy, not an official FATF or regulator code list, and are labelled as
# such in the export so nobody downstream mistakes them for a standard.
TYPOLOGY = {
    "R01": "Pass-through conduit",
    "R02": "Turnover disproportionate to balance",
    "R03": "Sudden account activation",
    "R04": "Cash-out dominance",
    "R05": "Single payment rail concentration",
    "R06": "Structuring by ticket size",
    "R07": "Nocturnal activity pattern",
    "R08": "Spike-and-drain balance profile",
    "R09": "Customer profile mismatch",
    "R10": "New account rapid turnover",
    "R11": "Pure conduit, net flow near zero",
    "R12": "Conduit on a single rail",
    "ML": "Model-derived behavioural risk (no single rule fired)",
}

PRIORITY = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}

# How each field in our export maps onto the concept an AML platform expects.
# Published so an integrator can do the mapping without reading our code.
FIELD_MAP = {
    "alert_id": "Unique alert reference. Deterministic, so re-running does not duplicate cases.",
    "alert_datetime_utc": "Alert generation time, ISO 8601 UTC.",
    "entity_type": "Always ACCOUNT. This model scores accounts, not customers or transactions.",
    "entity_ref": "Account reference as supplied. Row index where the source is anonymised.",
    "risk_score": "Integer 0-1000. Calibrated probability x 1000.",
    "risk_probability": "Calibrated probability that the account is a mule, 0-1.",
    "priority": "1 = highest. Derived from the band, for queue ordering.",
    "risk_band": "LOW / MEDIUM / HIGH. Edges are fitted operating points, not fixed constants.",
    "scenario_codes": "Typology codes that fired. OUR taxonomy, not a regulator code list.",
    "scenario_names": "Plain-language typology names for the codes above.",
    "scenario_detail": "Per-code MEASURED lift and precision on this dataset, so a "
                       "receiving system can weight scenarios rather than treating "
                       "them as equally meaningful. Seven of ours score at or below "
                       "the base rate and say so.",
    "recommended_action": "Suggested disposition. Advisory only; a human decides.",
    "reasons": "Ranked feature attributions with direction and magnitude.",
    "model_id / model_version": "Provenance, for model risk management and audit.",
    "score_provenance": "Whether the score is out-of-fold or from the deployed model.",
    "data_integrity_warning": "Present when the source dataset failed its integrity audit.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _alert_id(entity_ref: str, score: int) -> str:
    """Deterministic id, so a re-run updates a case rather than duplicating it."""
    h = hashlib.sha256(f"{MODEL_ID}|{entity_ref}|{score}".encode()).hexdigest()[:12]
    return f"MG-{h.upper()}"


def _codes_for(idx: int, hits: dict[int, list[str]]) -> list[str]:
    """Scenario codes for one account, falling back to the model-derived code.

    An account can score highly without any deterministic rule firing. That is
    not a gap to paper over: it means the model found something the rule book
    does not describe, and the alert should say so rather than borrow a code it
    did not earn.
    """
    codes = hits.get(int(idx), [])
    return codes if codes else ["ML"]


def _integrity_warning() -> dict | None:
    """Carry the dataset caveat into every downstream system.

    If the benchmark this model was fitted on failed its integrity audit, that
    fact has to travel with the alert. An AML platform receiving a score with no
    provenance will treat it as trustworthy, and on this dataset it is not.
    """
    path = C.REPORTS_DIR / "06_integrity_audit.json"
    if not path.exists():
        return None
    try:
        a = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not a.get("verdict", {}).get("contaminated"):
        return None
    return {
        "severity": "ADVISORY",
        "finding": "The dataset this model was fitted on failed its integrity "
                   "audit: positives and negatives come from disjoint extracts.",
        "implication": "Scores are an upper bound on demonstrable performance, "
                       "not a validated real-world detection rate. Treat as "
                       "triage support pending a same-period sample.",
        "reference": "reports/00_INTEGRITY.md",
    }


def _rule_hits() -> dict[int, list[str]]:
    """Per-account scenario codes, written by Stage 9.

    An alert carrying only "the model flagged it" is close to useless in a case
    manager: the scenario code is what an analyst routes on, what a team reports
    on, and what a reviewer argues with. Absent this file the export degrades to
    a model-derived code rather than failing.
    """
    path = C.DATA_DIR / "rule_hits.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[int, list[str]] = {}
    for idx, fired in zip(df["account_idx"], df["rules_fired"].fillna("")):
        if fired:
            out[int(idx)] = [c for c in str(fired).split("|") if c]
    return out


def _scenario_reliability() -> dict:
    """Measured lift per scenario, shipped so the receiver can weight them.

    Stage 9 found that seven of our twelve rules score at or below the base rate
    on this dataset. Exporting a scenario code without that context invites a
    downstream system to treat every code as equally meaningful, which they
    demonstrably are not.
    """
    path = C.REPORTS_DIR / "09_rules_report.json"
    if not path.exists():
        return {}
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out = {}
    for r in rep.get("rules", []):
        if r.get("status") != "evaluated":
            continue
        lift = r.get("lift_over_prevalence", 0.0)
        out[r["id"]] = {
            "name": TYPOLOGY.get(r["id"], r.get("title", "")),
            "measured_lift_over_base_rate": lift,
            "precision": r.get("precision"),
            "guidance": ("useful signal" if lift > 2
                         else "weak signal, corroborate before acting" if lift > 1
                         else "performs at or below the base rate on this data; "
                              "informational only"),
        }
    return out


def build_alerts(min_band: str = "MEDIUM", limit: int = 500) -> dict:
    """Alerts at or above a band, in the documented export schema."""
    from utils import load_frame  # local import keeps module import cheap

    scores_path = C.DATA_DIR / "risk_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError("risk_scores.csv — run Stage 7/8 first.")
    rs = pd.read_csv(scores_path)

    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    floor = order.get(min_band.upper(), 2)
    sel = rs[rs["band"].map(order).fillna(0) >= floor]
    sel = sel.sort_values("risk_score", ascending=False).head(limit)

    # Out-of-fold SHAP gives the reasons; absent it, alerts still export.
    sv = names = None
    npz = C.DATA_DIR / "oof_shap.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=True) as z:
            sv, names = z["shap"], [str(x) for x in z["feature_names"]]

    warning = _integrity_warning()
    hits = _rule_hits()
    reliability = _scenario_reliability()
    alerts = []
    for _, row in sel.iterrows():
        idx = int(row["account_idx"])
        entity = str(idx)
        score = int(row["risk_score"])
        reasons = []
        if sv is not None and idx < len(sv):
            contrib = sv[idx]
            for j in np.argsort(np.abs(contrib))[::-1][:6]:
                if contrib[j] == 0:
                    continue
                code = names[j]
                reasons.append({
                    "variable": D.real_name(code),
                    "meaning": D.explain(code),
                    "direction": "INCREASES_RISK" if contrib[j] > 0 else "DECREASES_RISK",
                    "contribution": round(float(contrib[j]), 6),
                })

        alerts.append({
            "alert_id": _alert_id(entity, score),
            "alert_datetime_utc": _now(),
            "schema_version": SCHEMA_VERSION,
            "entity_type": "ACCOUNT",
            "entity_ref": entity,
            "risk_score": score,
            "risk_probability": round(float(row.get("calibrated_probability", score / 1000)), 6),
            "risk_band": str(row["band"]),
            "priority": PRIORITY.get(str(row["band"]), 3),
            "scenario_codes": _codes_for(idx, hits),
            "scenario_names": [TYPOLOGY.get(c, c) for c in _codes_for(idx, hits)],
            "scenario_detail": [
                {"code": c, "name": TYPOLOGY.get(c, c), **reliability.get(c, {})}
                for c in _codes_for(idx, hits) if c != "ML"
            ],
            "recommended_action": str(row["recommended_action"]),
            "reasons": reasons,
            "model_id": MODEL_ID,
            "model_version": SCHEMA_VERSION,
            "score_provenance": "out_of_fold",
            "disposition": "OPEN",
            "data_integrity_warning": warning,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now(),
        "model_id": MODEL_ID,
        "alert_count": len(alerts),
        "minimum_band": min_band.upper(),
        "taxonomy_note": "scenario_codes use MuleGuard's own typology taxonomy. "
                         "They are NOT FATF or regulator code lists and must be "
                         "mapped to the receiving system's scenario catalogue.",
        "field_mapping": FIELD_MAP,
        "scenario_reliability": reliability,
        "compatibility_statement": (
            "Vendor-neutral export. NOT certified against any named EFRMS or AML "
            "product; no such testing has been performed. Integration is a field "
            "mapping exercise against the schema documented in field_mapping."
        ),
        "alerts": alerts,
    }


def alerts_to_dataframe(payload: dict) -> pd.DataFrame:
    """Flatten to a delimited file, which is how most bank systems ingest."""
    rows = []
    for a in payload["alerts"]:
        rows.append({
            "alert_id": a["alert_id"],
            "alert_datetime_utc": a["alert_datetime_utc"],
            "entity_type": a["entity_type"],
            "entity_ref": a["entity_ref"],
            "risk_score": a["risk_score"],
            "risk_probability": a["risk_probability"],
            "risk_band": a["risk_band"],
            "priority": a["priority"],
            "scenario_codes": "|".join(a["scenario_codes"]),
            "scenario_names": " ; ".join(a["scenario_names"]),
            "n_scenarios": len(a["scenario_codes"]),
            "recommended_action": a["recommended_action"],
            "top_reasons": " ; ".join(
                f"{r['variable']} ({r['direction']})" for r in a["reasons"][:3]),
            "model_id": a["model_id"],
            "model_version": a["model_version"],
            "score_provenance": a["score_provenance"],
            "disposition": a["disposition"],
            "data_integrity_flag": bool(a["data_integrity_warning"]),
        })
    return pd.DataFrame(rows)


def case_pack(idx: int) -> dict:
    """Everything an investigator needs to work one alert, in one object."""
    import sys
    from pathlib import Path

    app_dir = Path(__file__).resolve().parent.parent / "app"
    if str(app_dir.parent) not in sys.path:
        sys.path.insert(0, str(app_dir.parent))
    from app import service

    a = service.analyse_account(int(idx))
    entity = str(idx)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now(),
        "case_reference": _alert_id(entity, int(a["risk_score"])),
        "entity": {"type": "ACCOUNT", "ref": entity,
                   "note": "Source data is anonymised; no personal data is "
                           "held or exported by this system."},
        "assessment": {
            "risk_score": a["risk_score"],
            "risk_probability": a["calibrated_probability"],
            "risk_band": a["band"],
            "priority": PRIORITY.get(a["band"], 3),
            "recommended_action": a["recommended_action"],
            "score_provenance": a["score_provenance"],
        },
        "reasons": a["top_reasons"],
        "evidence": a["evidence"],
        "investigator_steps": a["investigator_next_steps"],
        "model_provenance": {
            "model_id": MODEL_ID, "model_version": SCHEMA_VERSION,
            "explanation_provenance": a["explanation_provenance"],
        },
        "data_integrity_warning": _integrity_warning(),
        "str_preparation": {
            "status": "MATERIAL ASSEMBLED, NOT FILED",
            "note": "This pack assembles what an analyst needs to prepare a "
                    "Suspicious Transaction Report. It is not a filing and does "
                    "not constitute one. FIU-IND submission has its own schema "
                    "and its own legal responsibilities, and that decision "
                    "belongs to a human.",
        },
    }


def main() -> None:
    """Write the export bundle so it can be inspected without the API running."""
    out_dir = C.REPORTS_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_alerts(min_band="MEDIUM", limit=500)
    save_json(payload, out_dir / "alerts.json")

    df = alerts_to_dataframe(payload)
    df.to_csv(out_dir / "alerts.csv", index=False, encoding="utf-8")
    log(f"Wrote {out_dir / 'alerts.csv'}  ({len(df)} alerts)")

    if len(df):
        top = int(payload["alerts"][0]["entity_ref"])
        save_json(case_pack(top), out_dir / f"case_pack_{top}.json")

    save_json({
        "schema_version": SCHEMA_VERSION,
        "field_mapping": FIELD_MAP,
        "typology_taxonomy": TYPOLOGY,
        "compatibility_statement": payload["compatibility_statement"],
        "formats_provided": ["JSON (API and file)", "CSV (batch exchange)",
                             "OpenAPI 3 specification at /api/openapi.json"],
        "what_would_be_needed_for_a_named_platform": [
            "That platform's integration specification.",
            "A scenario catalogue to map scenario_codes onto.",
            "An agreed entity key, since this dataset is anonymised.",
            "A test environment to validate ingestion against.",
        ],
    }, out_dir / "integration_contract.json")
    log(f"Export bundle written to {out_dir}")


if __name__ == "__main__":
    main()
