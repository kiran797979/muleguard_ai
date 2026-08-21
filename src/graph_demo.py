"""
Graph capability demonstration, on SYNTHETIC data. Read this header first.

WHAT THIS IS
------------
The hackathon dataset has no counterparty column anywhere in its 3,924
variables, so Stage 6 detects the absence and disables itself rather than
inventing an edge list. That is the correct behaviour and it stays.

But "the code would work if the data allowed it" is a claim, and claims should
be demonstrable. So this script manufactures a transaction ledger that DOES have
counterparties, plants mule rings inside it, and runs the real `propagate()`
from `04_graph.py` against it. Same function, same hop decay, same constants.

WHAT THIS IS NOT
----------------
It is **not a result**. Not a benchmark, not a metric about mule detection, and
not evidence that the system finds rings in the supplied data. The rings here
were placed by us and then found by us. The only thing it establishes is that
the graph stage is implemented and functions when given edges.

Every output file is written under reports/demo_graph/ and every figure is
watermarked, so a number from here can never be mistaken for a number from the
real pipeline.

THE QUESTION IT ACTUALLY ANSWERS
--------------------------------
Given a ring where a bank has already confirmed only a couple of members, does
propagation surface the rest? And what does it cost in innocent accounts pulled
in? That trade is the real argument for and against graph propagation, and it is
what the figure shows.

Run:  python src/graph_demo.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import log, save_json

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The real Stage 6 propagation. Importing it rather than copying is the whole
# point: if that function changes, this demonstration changes with it.
_graph = importlib.import_module("04_graph")
propagate = _graph.propagate

import rings as R          # structural detection, no labels involved

OUT_DIR = C.REPORTS_DIR / "demo_graph"

N_ACCOUNTS = 2000
N_RINGS = 8
RING_SIZE = (5, 9)
CONFIRMED_PER_RING = 2        # what a bank already knows when the alert lands
BACKGROUND_EDGES = 4200       # ordinary customer-to-customer transfers
SEED = C.RANDOM_STATE


def build_ledger(rng: np.random.Generator) -> dict:
    """A synthetic transfer ledger with mule rings planted inside it."""
    edges: list[tuple[int, int]] = []

    # Ordinary traffic. Most accounts pay a handful of unrelated counterparties.
    for _ in range(BACKGROUND_EDGES):
        a, b = rng.integers(0, N_ACCOUNTS, size=2)
        if a != b:
            edges.append((int(a), int(b)))

    rings: list[list[int]] = []
    taken: set[int] = set()
    for _ in range(N_RINGS):
        size = int(rng.integers(*RING_SIZE))
        pool = [i for i in range(N_ACCOUNTS) if i not in taken]
        members = [int(x) for x in rng.choice(pool, size=size, replace=False)]
        taken.update(members)
        rings.append(members)

        # Layering shape: a collector fans out to intermediaries, who chain
        # onward to a cash-out account. This is what the propagation has to walk.
        collector, cashout = members[0], members[-1]
        middles = members[1:-1]
        for m in middles:
            edges.append((collector, m))
            edges.append((m, cashout))
        # A little intra-ring cross traffic, as real layering has.
        for _ in range(size):
            a, b = rng.choice(members, size=2, replace=False)
            if a != b:
                edges.append((int(a), int(b)))

    return {"edges": edges, "rings": rings}


def evaluate(ledger: dict, rng: np.random.Generator) -> dict:
    """Seed a fraction of each ring, propagate, and measure what it recovers."""
    rings = ledger["rings"]
    all_members = {m for r in rings for m in r}

    confirmed: set[int] = set()
    unknown: set[int] = set()
    for r in rings:
        picked = [int(x) for x in rng.choice(r, size=min(CONFIRMED_PER_RING, len(r)),
                                             replace=False)]
        confirmed.update(picked)
        unknown.update(set(r) - set(picked))

    scores = propagate(ledger["edges"], confirmed, N_ACCOUNTS)

    # Break the result down BY HOP rather than treating any non-zero score as a
    # flag. That distinction matters: the pipeline consumes this as a 0.15 blend
    # weight on the risk score, never as a hard alert, and lumping every hop
    # together hides the fact that reach and noise diverge sharply with distance.
    by_hop = []
    for hop in sorted(C.HOP_DECAY):
        decay = C.HOP_DECAY[hop]
        at = {i for i, s in scores.items()
              if abs(s - decay) < 1e-9 and i not in confirmed}
        rec = at & unknown
        inn = at - all_members
        by_hop.append({
            "hop": hop,
            "proximity_score": decay,
            "accounts_at_this_hop": len(at),
            "ring_members_recovered": len(rec),
            "innocent_accounts": len(inn),
            "precision": round(len(rec) / max(len(at), 1), 4),
        })

    # Cumulative view: what you get if you stop after N hops.
    cumulative = []
    running: set[int] = set()
    for row in by_hop:
        decay = row["proximity_score"]
        running |= {i for i, s in scores.items()
                    if s >= decay - 1e-9 and i not in confirmed}
        rec = running & unknown
        cumulative.append({
            "stop_after_hop": row["hop"],
            "queue_size": len(running),
            "ring_members_recovered": len(rec),
            "recall_of_unknown": round(len(rec) / max(len(unknown), 1), 4),
            "precision": round(len(rec) / max(len(running), 1), 4),
            "alerts_per_member_found": round(len(running) / max(len(rec), 1), 1),
        })

    surfaced = {i for i, s in scores.items() if s > 0 and i not in confirmed}
    recovered = surfaced & unknown
    return {
        "confirmed_seeds": sorted(confirmed),
        "unknown_ring_members": len(unknown),
        "by_hop": by_hop,
        "cumulative": cumulative,
        "all_hops_queue_size": len(surfaced),
        "all_hops_recall": round(len(recovered) / max(len(unknown), 1), 4),
        "all_hops_precision": round(len(recovered) / max(len(surfaced), 1), 4),
        "scores": scores,
        "all_members": all_members,
        "recovered": recovered,
    }


def evaluate_rings(ledger: dict) -> dict:
    """Find rings from structure alone, then score against what was planted.

    Nothing about the planted rings is passed in. `detect_rings` sees an edge
    list and nothing else, so this measures whether a laundering network is
    recoverable from its shape rather than from prior suspicion.

    Reported as a curve over the ranked candidate list. A single number would
    hide the part that matters operationally: the top few candidates are almost
    pure, and quality degrades predictably after that.
    """
    truth = {m for r in ledger["rings"] for m in r}
    res = R.detect_rings(ledger["edges"], n_nodes=N_ACCOUNTS)
    ranked = res["rings"]

    curve = []
    for k in (3, 5, 8, 10, 15, 20, 30, 50):
        if k > len(ranked):
            break
        acc = {m for r in ranked[:k] for m in r["members"]}
        tp = len(acc & truth)
        curve.append({
            "top_k_rings": k,
            "accounts_surfaced": len(acc),
            "planted_members_found": tp,
            "precision": round(tp / max(len(acc), 1), 4),
            "recall_of_all_members": round(tp / max(len(truth), 1), 4),
            "alerts_per_member_found": round(len(acc) / max(tp, 1), 1),
        })

    pure = sum(1 for r in ranked[:8] if r["members"] and
               set(r["members"]).issubset(truth))
    return {
        "uses_labels": False,
        "uses_seeds": False,
        "planted_members": len(truth),
        "communities_examined": res["n_communities_examined"],
        "candidate_rings": len(ranked),
        "top8_entirely_planted": pure,
        "curve": curve,
        "top_candidates": [
            {k: v for k, v in r.items() if k not in ("members", "roles")}
            | {"size": r["size"],
               "planted_members_inside": len(set(r["members"]) & truth)}
            for r in ranked[:8]
        ],
    }


def draw(ledger: dict, res: dict) -> None:
    """One ring, drawn: seeds, recovered members, and the edges walked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as exc:  # noqa: BLE001
        log(f"Plotting unavailable ({exc}); skipping figure.")
        return

    ring = max(ledger["rings"], key=len)
    confirmed = set(res["confirmed_seeds"])
    recovered = res["recovered"]

    # The ring plus one hop of its real neighbourhood, so the picture shows the
    # surrounding traffic propagation had to distinguish from the ring.
    keep = set(ring)
    for a, b in ledger["edges"]:
        if a in ring and len(keep) < len(ring) + 14:
            keep.add(b)
        elif b in ring and len(keep) < len(ring) + 14:
            keep.add(a)

    G = nx.DiGraph()
    G.add_nodes_from(keep)
    G.add_edges_from([(a, b) for a, b in ledger["edges"] if a in keep and b in keep])

    colours, sizes, edgecols = [], [], []
    for n in G.nodes():
        if n in confirmed:
            colours.append("#b3261e"); sizes.append(520); edgecols.append("#000")
        elif n in recovered:
            colours.append("#f0a202"); sizes.append(430); edgecols.append("#000")
        elif n in ring:
            colours.append("#8a8a8a"); sizes.append(300); edgecols.append("#000")
        else:
            colours.append("#d9dcd6"); sizes.append(150); edgecols.append("#aaa")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    pos = nx.spring_layout(G, seed=SEED, k=.65)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c2c6c0", width=.9,
                           arrowsize=8, alpha=.85)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colours, node_size=sizes,
                           edgecolors=edgecols, linewidths=.7)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor="#b3261e",
               markeredgecolor="#000", label="confirmed mule (seed)"),
        Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor="#f0a202",
               markeredgecolor="#000", label="recovered by propagation"),
        Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor="#d9dcd6",
               markeredgecolor="#aaa", label="ordinary account"),
    ], loc="upper left", fontsize=8.5, frameon=False)

    ax.set_title("Graph propagation on a SYNTHETIC ledger\n"
                 "Seeded with 2 confirmed members; the rest of the ring is recovered by hop decay",
                 fontsize=10.5, loc="left")
    ax.axis("off")

    # Unmissable watermark. This figure must never be mistaken for a result.
    ax.text(.5, .5, "SYNTHETIC\nDEMONSTRATION", transform=ax.transAxes,
            fontsize=30, color="#b3261e", alpha=.13, ha="center", va="center",
            fontweight="bold", rotation=22, zorder=10)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"ring_propagation.{ext}", dpi=220)
    plt.close(fig)
    log(f"wrote {OUT_DIR / 'ring_propagation.png'}")


def _interpret(res: dict) -> str:
    """State the trade-off in the terms an AML desk would argue about."""
    h1 = res["cumulative"][0]
    last = res["cumulative"][-1]
    return (
        f"Stopping after one hop puts {h1['queue_size']} accounts in the queue and "
        f"recovers {h1['ring_members_recovered']} of {res['unknown_ring_members']} "
        f"unknown ring members, at {h1['precision']:.0%} precision. Letting it run "
        f"to hop {last['stop_after_hop']} reaches {last['recall_of_unknown']:.0%} "
        f"recall but inflates the queue to {last['queue_size']} accounts at "
        f"{last['precision']:.0%} precision, roughly "
        f"{last['alerts_per_member_found']} alerts for every member found. "
        f"That collapse is the honest argument against untuned propagation: on a "
        f"graph with ordinary background traffic, three hops reaches most of the "
        f"bank. It is also why the pipeline consumes this as a 0.15 blend weight "
        f"rather than as an alert, and why a deployment would cap it at one or two "
        f"hops and tune the decay against its own transaction density."
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    log("=" * 64)
    log("GRAPH CAPABILITY DEMONSTRATION — SYNTHETIC DATA, NOT A RESULT")
    log("The supplied dataset has no counterparty column, so Stage 6 disables")
    log("itself. This shows the same code working when edges do exist.")
    log("=" * 64)

    ledger = build_ledger(rng)
    log(f"Synthetic ledger: {N_ACCOUNTS:,} accounts, {len(ledger['edges']):,} transfers, "
        f"{len(ledger['rings'])} planted rings "
        f"({sum(len(r) for r in ledger['rings'])} members)")

    res = evaluate(ledger, rng)
    log(f"Seeded {len(res['confirmed_seeds'])} confirmed members "
        f"({CONFIRMED_PER_RING} per ring); {res['unknown_ring_members']} ring "
        f"members left to find")
    log("")
    log("  STOP AFTER   QUEUE   FOUND   RECALL   PRECISION   ALERTS PER FIND")
    for c in res["cumulative"]:
        log(f"  hop {c['stop_after_hop']}        {c['queue_size']:>5}   "
            f"{c['ring_members_recovered']:>5}   {c['recall_of_unknown']:>6.1%}   "
            f"{c['precision']:>9.1%}   {c['alerts_per_member_found']:>15}")

    ring_res = evaluate_rings(ledger)
    log("")
    log("-" * 64)
    log("NOW THE HARDER QUESTION: find the ring with NO labels and NO seeds")
    log("-" * 64)
    log(f"  {ring_res['communities_examined']} communities examined, "
        f"{ring_res['candidate_rings']} candidate rings ranked by structure alone")
    log("")
    log("  TOP-K RINGS   ACCOUNTS   FOUND   PRECISION   RECALL   ALERTS PER FIND")
    for c in ring_res["curve"]:
        log(f"  {c['top_k_rings']:>11}   {c['accounts_surfaced']:>8}   "
            f"{c['planted_members_found']:>5}   {c['precision']:>9.1%}   "
            f"{c['recall_of_all_members']:>6.1%}   "
            f"{c['alerts_per_member_found']:>15}")
    log("")
    log(f"  {ring_res['top8_entirely_planted']} of the top 8 candidates contain "
        f"planted members and nothing else.")

    draw(ledger, res)

    save_json({
        "ring_detection_unseeded": ring_res,
        "WARNING": "SYNTHETIC CAPABILITY DEMONSTRATION. Not a result, not a "
                   "benchmark, and not evidence about the hackathon dataset. "
                   "The rings below were planted by this script and then found "
                   "by this script. Its only purpose is to show that the Stage 6 "
                   "code path executes when a counterparty ledger exists.",
        "why_this_exists": "reports/04_graph_report.json records that the real "
                           "dataset has no counterparty column, so Stage 6 "
                           "disables itself. This demonstrates the disabled code.",
        "uses_production_code": "src/04_graph.py:propagate — imported, not copied",
        "hop_decay": C.HOP_DECAY,
        "max_hops": C.MAX_HOPS,
        "synthetic_ledger": {
            "accounts": N_ACCOUNTS,
            "transfers": len(ledger["edges"]),
            "planted_rings": len(ledger["rings"]),
            "ring_members": sum(len(r) for r in ledger["rings"]),
            "confirmed_per_ring": CONFIRMED_PER_RING,
        },
        "result": {k: v for k, v in res.items()
                   if k not in ("scores", "all_members", "recovered", "confirmed_seeds")},
        "how_this_score_is_consumed": (
            "As a 0.15 blend weight on the calibrated probability in "
            "05_score_explain.py, never as a standalone alert. The per-hop "
            "breakdown matters because reach and noise diverge sharply with "
            "distance, and a single 'flagged / not flagged' number hides that."
        ),
        "interpretation": _interpret(res),
    }, OUT_DIR / "graph_demo_report.json")


if __name__ == "__main__":
    main()
