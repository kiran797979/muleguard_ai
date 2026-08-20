"""
Stage 6 — Graph intelligence: label propagation from confirmed-mule seeds.

This stage runs ONLY if Stage 2 detected counterparty/edge columns (a real
transaction ledger of who-paid-whom). Otherwise it exits cleanly, writing a
skip note — we do not fabricate an edge list to manufacture a graph score.

When edges exist:
  * Build a directed transfer graph in NetworkX (node = account).
  * Seed the confirmed mules (target == 1) with propagation score 1.0.
  * Propagate suspicion outward with hop decay (0.85 / 0.70 / 0.55), stopping
    after MAX_HOPS to avoid over-flagging.
  * Write per-account graph proximity scores to data/graph_scores.csv.

Run:  python src/04_graph.py
"""

from __future__ import annotations

import json

import pandas as pd

import config as C
import schema as S
from utils import load_frame, log, save_json


def load_edge_info() -> dict:
    path = C.REPORTS_DIR / "02_features_report.json"
    if not path.exists():
        return {"graph_possible": False, "edge_columns": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("graph_detection", {"graph_possible": False, "edge_columns": []})


def propagate(edges: list[tuple[int, int]], seeds: set[int], n: int) -> dict[int, float]:
    """BFS-style hop-decayed propagation from seed nodes over a directed graph."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    scores = {i: 0.0 for i in range(n)}
    for s in seeds:
        scores[s] = 1.0

    frontier = set(seeds)
    visited = set(seeds)
    for hop in range(1, C.MAX_HOPS + 1):
        decay = C.HOP_DECAY.get(hop, 0.0)
        nxt = set()
        for node in frontier:
            # Follow both directions: money in and money out of a known mule.
            for nb in list(G.successors(node)) + list(G.predecessors(node)):
                if nb not in visited:
                    scores[nb] = max(scores[nb], decay)
                    nxt.add(nb)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return scores


def main() -> None:
    info = load_edge_info()
    if not info.get("graph_possible"):
        note = {
            "status": "SKIPPED",
            "reason": "No counterparty/edge columns detected in Stage 2. "
                      "Data is an account-level feature matrix, not a transaction "
                      "ledger, so a money-flow graph cannot be built honestly.",
        }
        save_json(note, C.REPORTS_DIR / "04_graph_report.json")
        log("Graph stage SKIPPED (no edge data). See reports/04_graph_report.json")
        return

    df = load_frame(C.FEATURES_PARQUET)
    n = len(df)
    edge_cols = info["edge_columns"]

    # Build edges: each edge column maps row -> referenced account index.
    edges: list[tuple[int, int]] = []
    for col in edge_cols:
        for src, dst in enumerate(df[col].fillna(-1).astype(int).tolist()):
            if 0 <= dst < n and dst != src:
                edges.append((src, dst))

    target = S.bind_target(df, C)
    seeds = set(df.index[df[target] == 1].tolist())
    log(f"Graph: {n} nodes, {len(edges):,} edges, {len(seeds)} seeds")

    scores = propagate(edges, seeds, n)
    out = pd.DataFrame({"account_idx": list(scores.keys()),
                        "graph_proximity": list(scores.values())})
    out.to_csv(C.GRAPH_SCORES_CSV, index=False)

    save_json({
        "status": "OK",
        "n_nodes": n,
        "n_edges": len(edges),
        "n_seeds": len(seeds),
        "n_reached": int((out["graph_proximity"] > 0).sum()),
    }, C.REPORTS_DIR / "04_graph_report.json")
    log(f"Wrote graph proximity scores -> {C.GRAPH_SCORES_CSV}")


if __name__ == "__main__":
    main()
