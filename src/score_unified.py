"""
The whole system behind one entry point.

The problem this solves
-----------------------
Until now this project was a set of excellent parts that never met: a
behavioural ensemble, a motif detector, a ring detector, a temporal localiser.
Each is measured; the whole was not, and "show me your model" got you one of
them. Against a brief that explicitly rewards several models working in series
or parallel, that is the gap.

This takes a transaction ledger in one end and produces the required submission
out the other:

    account_id, is_mule, suspicious_start, suspicious_end

Four signal families, deliberately independent
----------------------------------------------
    BEHAVIOURAL   what one account's own money does — pass-through, retention,
                  burstiness, threshold hugging.        (ledger_features.py)
    MOTIF         the shape of a laundering event in a time window — fan-in,
                  fan-out, gather-scatter, chains.      (motifs.py)
    STRUCTURAL    closed cells: dense inside, sparse outward.   (rings.py)
    TEMPORAL      when the episode happened.            (temporal.py)

They fail in different places, which is the point of having four. Motifs need no
global structure and work when a mule keeps trading normally; rings need the
cell to be separable and fail when it is not; behavioural features need no graph
at all. On SAML-D, rings score nothing and motifs reach 28x lift — on a closed
synthetic cell, rings find it and motifs are noisy. Blending them is not
padding, it is coverage.

How they are combined
---------------------
By a **supervised** model, not a hand-weighted sum, whenever labels exist. The
motif and ring scores enter as ordinary columns alongside the behavioural ones
and the ensemble learns what each is worth — including learning that one is
worth nothing, which is what happened to the isolation forest and got published
rather than hidden. A hand-weighted blend would be us asserting the weights;
this makes the data assert them.

Without labels it falls back to a documented rank-average and says so, because
an unsupervised blend is a guess and should be labelled as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ledger_features as LF
import motifs as MO
import rings as RG
import temporal as TE

MOTIF_COLUMNS = ("motif_score", "motif_fan_out", "motif_fan_in",
                 "motif_gather_scatter", "motif_chain")
RING_COLUMNS = ("ring_score", "role_relay", "role_collector",
                "role_disperser", "role_terminal")


# ==========================================================================
# Feature assembly
# ==========================================================================
def network_features(txns: pd.DataFrame, accounts: pd.Index) -> pd.DataFrame:
    """Motif and structural evidence, as columns the model can learn from."""
    m = txns.rename(columns={"timestamp": "date"})
    motif = MO.detect_motifs(m[["src", "dst", "amount", "date"]])

    f = pd.DataFrame(index=accounts)
    f.index.name = "account_id"
    motif_s = pd.Series(motif["scores"], dtype=float)
    f["motif_score"] = motif_s.reindex(accounts).fillna(0.0).to_numpy()
    for name, col in (("FAN_OUT", "motif_fan_out"), ("FAN_IN", "motif_fan_in"),
                      ("GATHER_SCATTER", "motif_gather_scatter"),
                      ("CHAIN", "motif_chain")):
        members = set(motif.get("accounts_by_motif", {}).get(name, []))
        f[col] = pd.Series(accounts.isin(members).astype(float), index=accounts)

    edges = list(zip(txns["src"].to_numpy(), txns["dst"].to_numpy(),
                     txns["amount"].to_numpy(dtype=float)))
    ring = RG.detect_rings(edges)
    ring_s = pd.Series(RG.account_ring_scores(ring), dtype=float)
    f["ring_score"] = ring_s.reindex(accounts).fillna(0.0).to_numpy()

    roles = RG.structural_roles(RG.build_graph(edges))
    role_s = pd.Series(roles).reindex(accounts)
    for role, col in (("RELAY", "role_relay"), ("COLLECTOR", "role_collector"),
                      ("DISPERSER", "role_disperser"), ("TERMINAL", "role_terminal")):
        f[col] = (role_s == role).astype(float).to_numpy()

    return f, {"motif_counts": motif["counts"],
               "rings_flagged": ring.get("n_rings_flagged", 0)}


def build_features(txns: pd.DataFrame, with_network: bool = True):
    """Behavioural features, optionally enriched with network evidence."""
    behav = LF.build(txns)
    meta = {"n_transactions": int(len(txns)), "n_accounts": int(len(behav)),
            "network_used": bool(with_network)}
    if not with_network:
        return behav, meta
    net, nmeta = network_features(txns, behav.index)
    meta.update(nmeta)
    return behav.join(net, how="left").fillna(0.0), meta


# ==========================================================================
# Scoring
# ==========================================================================
class UnifiedScorer:
    """Behavioural + motif + structural evidence, combined by a fitted model."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.columns: list[str] = []
        self.mode = "unfitted"

    def fit(self, features: pd.DataFrame, y: np.ndarray) -> "UnifiedScorer":
        from ensemble import MuleEnsemble
        self.columns = list(features.columns)
        X = features.to_numpy(dtype=np.float32)
        self.model = MuleEnsemble(seed=self.seed).fit(X, np.asarray(y).astype(int))
        self.mode = "supervised"
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return self._unsupervised(features)
        X = features[self.columns].to_numpy(dtype=np.float32)
        return self.model.predict_proba(X)

    @staticmethod
    def _unsupervised(features: pd.DataFrame) -> np.ndarray:
        """Rank-average of the signals that point one way without a label.

        A fallback, and labelled as one. Only features whose direction is known
        a priori are used: a high pass-through is suspicious, a high turnover is
        not necessarily. Everything is converted to a rank first so that no
        single heavy-tailed column dominates the average.
        """
        directional = ["passthrough", "burstiness", "threshold_share",
                       "round_share", "night_share", "txn_per_counterparty",
                       "motif_score", "ring_score"]
        use = [c for c in directional if c in features.columns]
        if not use:
            return np.zeros(len(features))
        r = features[use].rank(pct=True)
        # Retention is inverted: keeping nothing is the suspicious direction.
        if "retention" in features.columns:
            r["retention_inv"] = (1.0 - features["retention"]).rank(pct=True)
        return r.mean(axis=1).to_numpy()


def to_account_view(txns: pd.DataFrame) -> pd.DataFrame:
    """Ledger -> one row per account per transaction, with a direction.

    A transfer touches two accounts: it is a debit for the sender and a credit
    for the receiver. The temporal localiser reasons about one account at a
    time, so the ledger has to be stacked into that view before it can run.
    """
    out = pd.concat([
        pd.DataFrame({"account_id": txns["src"].to_numpy(),
                      "timestamp": txns["timestamp"].to_numpy(),
                      "amount": txns["amount"].to_numpy(),
                      "direction": "D"}),
        pd.DataFrame({"account_id": txns["dst"].to_numpy(),
                      "timestamp": txns["timestamp"].to_numpy(),
                      "amount": txns["amount"].to_numpy(),
                      "direction": "C"}),
    ], ignore_index=True)
    return out


# ==========================================================================
# End to end
# ==========================================================================
def run(txns: pd.DataFrame, labels: pd.Series | None = None,
        threshold: float = 0.5, with_network: bool = True) -> dict:
    """Ledger in, submission out.

    `labels` is a Series indexed by account_id. When present the blend is
    learned; when absent it is a documented rank-average and the report says so.
    """
    features, meta = build_features(txns, with_network=with_network)
    scorer = UnifiedScorer()

    if labels is not None:
        y = labels.reindex(features.index).fillna(0).to_numpy().astype(int)
        if y.sum() >= 10:
            scorer.fit(features, y)
        else:
            meta["note"] = (f"only {int(y.sum())} positives supplied; too few to fit, "
                            f"fell back to the unsupervised blend")
    proba = scorer.predict_proba(features)

    risk = pd.DataFrame({"account_id": features.index, "is_mule": proba})
    windows = TE.detect_windows(
        to_account_view(txns),
        risk=dict(zip(risk.account_id, risk.is_mule)), min_risk=threshold)

    submission = TE.to_submission(risk, windows, threshold=threshold)
    meta["mode"] = scorer.mode
    meta["n_features"] = len(features.columns)
    meta["flagged"] = int((risk.is_mule >= threshold).sum())
    return {"submission": submission, "risk": risk, "features": features,
            "windows": windows, "meta": meta, "scorer": scorer}
