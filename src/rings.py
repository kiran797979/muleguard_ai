"""
Structural detection of mule rings — networks, not just account-level roles.

Why this exists
---------------
Stage 6 (`04_graph.py`) propagates suspicion outward from accounts a bank has
*already confirmed*. That is useful and it is not ring detection: it cannot find
a network that nobody has flagged yet, and its recall is bounded by how good the
existing alerts were. The honest criticism of the original design is that it
detects account-level roles and proximity to known bad actors, not rings.

This module finds the ring itself, from structure alone, with **no labels and no
seeds**. A laundering network has a shape that ordinary payment traffic does not:

  DENSITY      members transact with each other far more than chance allows
  ISOLATION    the group has many internal edges and few to the outside world,
               which is `conductance` — the standard community-quality measure
  PASS-THROUGH value entering the group leaves it again; little is retained
  CYCLES       money can return towards its origin in a few hops, which almost
               never happens in genuine retail payment traffic
  ROLES        a collector fans in, relays chain onward, a cash-out terminates

Any one of those is weak on its own. Ordinary customers form dense clusters
(families, small businesses), and cycles occur innocently (mutual payments).
Requiring several at once is what separates a ring from a neighbourhood.

Known limit, measured rather than assumed
-----------------------------------------
This method requires the ring to be *separable* from the rest of the graph.
Benchmarked against SAML-D (a third-party AML dataset with named typologies) it
does not flag the true rings, and the reason is worth stating precisely: those
rings have a **conductance of 0.98-0.99**. Each member carries roughly 200
transactions of which only one or two are ring edges, so the laundering is about
1% of that account's activity and the group is not distinguishable from the
graph by any community method.

The gates below reject those groups, which is the correct behaviour — a detector
that flagged them would be flagging essentially every dense neighbourhood. Where
the recruited accounts keep transacting normally, use `motifs.py`, which looks
for the laundering *shape* inside a time window and does not require global
separability. The two are complementary: this module finds closed cells, that
one finds patterns hidden in ordinary traffic.

What this deliberately does NOT do
----------------------------------
It does not run on the supplied hackathon dataset, because that dataset has no
counterparty column and therefore no edges. Every function here takes an edge
list as an argument; nothing infers edges from correlation, shared attribute
values, or feature similarity. An "inferred" graph built from column similarity
would produce confident, meaningless rings, which is precisely the failure mode
the rest of this project exists to avoid.

Run standalone against a ledger CSV of `src,dst[,amount]`:

    python src/rings.py --edges path/to/ledger.csv
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import networkx as nx

# --------------------------------------------------------------------------
# Tunables. These are deliberately conservative and NOT fitted to any dataset:
# fitting them against planted rings in our own synthetic ledger would prove
# nothing except that we can fit our own generator.
# --------------------------------------------------------------------------
MIN_RING_SIZE = 3
MAX_RING_SIZE = 25
MAX_CYCLE_LEN = 5
CONDUCTANCE_CEILING = 0.60     # above this the group is not meaningfully separate
MIN_INTERNAL_DENSITY = 0.15    # relative to a complete directed subgraph

# Weights for the composite ring score. They sum to 1.0 and are round numbers on
# purpose: a precise-looking weight vector would imply a tuning exercise that
# this problem does not yet have the labelled data to support.
W_DENSITY, W_ISOLATION, W_CYCLES, W_PASSTHROUGH, W_ROLES = 0.25, 0.25, 0.20, 0.20, 0.10


# ==========================================================================
# Graph construction
# ==========================================================================
def build_graph(edges, n_nodes: int | None = None) -> nx.DiGraph:
    """Build a directed multigraph-collapsed transfer graph.

    `edges` is an iterable of (src, dst) or (src, dst, amount). Parallel
    transfers between the same pair collapse into one edge carrying `weight`
    (count) and `amount` (total value), because for structure it is the presence
    of a relationship that matters, while value is kept for the flow tests.
    """
    G = nx.DiGraph()
    if n_nodes:
        # Only meaningful when nodes are row indices. Real ledgers carry string
        # account ids ("ACCT_000003"), and coercing those to int used to raise
        # ValueError the moment this met production-shaped data.
        G.add_nodes_from(range(n_nodes))
    for e in edges:
        if len(e) == 3:
            u, v, amt = e
        else:
            (u, v), amt = e, 1.0
        if u == v:
            continue                      # self-transfers are not a relationship
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
            G[u][v]["amount"] += float(amt)
        else:
            G.add_edge(u, v, weight=1, amount=float(amt))
    return G


# ==========================================================================
# Account-level structural roles
# ==========================================================================
def structural_roles(G: nx.DiGraph) -> dict:
    """Classify each account by the shape of its own traffic.

    These are the roles a laundering network needs somebody to play. They are
    descriptive, not accusatory: plenty of legitimate accounts are collectors
    (a merchant) or dispersers (a payroll account). The role matters only in
    combination with the group-level evidence below.
    """
    roles: dict[int, str] = {}
    for node in G.nodes():
        ind, outd = G.in_degree(node), G.out_degree(node)
        in_amt = sum(d["amount"] for _, _, d in G.in_edges(node, data=True))
        out_amt = sum(d["amount"] for _, _, d in G.out_edges(node, data=True))
        total = in_amt + out_amt

        if ind == 0 and outd == 0:
            roles[node] = "ISOLATED"
        elif ind == 0:
            roles[node] = "SOURCE"
        elif outd == 0:
            roles[node] = "TERMINAL"          # where value stops: a cash-out point
        else:
            # Retention: how much of what arrived stayed. A relay keeps ~nothing.
            retention = abs(in_amt - out_amt) / total if total else 1.0
            if retention < 0.15 and ind >= 1 and outd >= 1:
                roles[node] = "RELAY"          # pass-through conduit
            elif ind >= 3 * max(outd, 1):
                roles[node] = "COLLECTOR"      # fans in
            elif outd >= 3 * max(ind, 1):
                roles[node] = "DISPERSER"      # fans out
            else:
                roles[node] = "ORDINARY"
    return roles


# ==========================================================================
# Group-level evidence
# ==========================================================================
def _internal_density(G: nx.DiGraph, members: set) -> float:
    """Edges present inside the group over edges possible inside it.

    Via `subgraph`, which is O(|members|). Scanning every edge in the graph and
    testing membership is O(|E|) *per candidate group*, which is invisible on a
    2,000-node demo and fatal on a 9.5M-edge ledger.
    """
    k = len(members)
    if k < 2:
        return 0.0
    return G.subgraph(members).number_of_edges() / (k * (k - 1))


def _conductance(U: nx.Graph, members: set, total_volume: float) -> float:
    """Share of the group's edge endpoints that leave the group.

    0.0 = perfectly sealed, 1.0 = no internal structure at all. A laundering
    ring is unusually sealed because its members mostly transact with each
    other; ordinary customers spray payments across the whole economy.

    Computed directly rather than through `nx.conductance`, which needs the
    complement set materialised. Building `set(U.nodes()) - members` is O(N) per
    group, and the previous version also rebuilt the entire undirected
    projection on every call. Both are fine at demo scale and neither survives
    855,000 accounts.

    Same definition as NetworkX: cut / min(vol(S), vol(complement of S)),
    where the cut is
    the group's total degree less twice its internal edges.
    """
    if not members or total_volume <= 0:
        return 1.0
    vol_s = float(sum(d for _, d in U.degree(members)))
    internal = U.subgraph(members).number_of_edges()
    cut = vol_s - 2.0 * internal
    vol_rest = total_volume - vol_s
    denom = min(vol_s, vol_rest)
    if denom <= 0:
        return 1.0
    return float(max(cut, 0.0) / denom)


def _cycle_count(G: nx.DiGraph, members: set, max_len: int = MAX_CYCLE_LEN) -> int:
    """Short cycles wholly inside the group: money that comes back around.

    Bounded by `max_len` and by group size, because enumerating simple cycles is
    exponential and we only ever ask it about small candidate groups.
    """
    sub = G.subgraph(members)
    if sub.number_of_edges() == 0:
        return 0
    try:
        return sum(1 for c in nx.simple_cycles(sub, length_bound=max_len))
    except TypeError:                      # networkx < 3.1 has no length_bound
        n = 0
        for c in nx.simple_cycles(sub):
            if len(c) <= max_len:
                n += 1
            if n > 500:
                break
        return n
    except Exception:
        return 0


def _passthrough(G: nx.DiGraph, members: set) -> float:
    """How completely value entering the group leaves it again.

    1.0 means every rupee that came in went back out — the defining property of
    a conduit network. A group that accumulates (a business and its suppliers)
    scores low.
    """
    # Only edges incident to the group are examined, not the whole ledger.
    inflow = sum(d["amount"] for u, _, d in G.in_edges(members, data=True)
                 if u not in members)
    outflow = sum(d["amount"] for _, v, d in G.out_edges(members, data=True)
                  if v not in members)
    if inflow <= 0:
        return 0.0
    return min(outflow / inflow, 1.0)


def _role_evidence(roles: dict, members: set) -> float:
    """Does the group contain the division of labour a ring needs?

    A collector or disperser to gather, relays to layer, a terminal to cash out.
    Scored as the share of those three functions present.
    """
    present = {roles.get(m, "ORDINARY") for m in members}
    score = 0.0
    if present & {"COLLECTOR", "DISPERSER"}:
        score += 1 / 3
    if "RELAY" in present:
        score += 1 / 3
    if "TERMINAL" in present:
        score += 1 / 3
    return score


# ==========================================================================
# Ring detection
# ==========================================================================
MAX_REFINE_DEPTH = 6


def _louvain(G: nx.DiGraph, resolution: float, seed: int):
    U = G.to_undirected()
    if U.number_of_edges() == 0:
        return []
    try:
        comms = nx.community.louvain_communities(U, resolution=resolution, seed=seed)
    except Exception:
        comms = nx.community.label_propagation_communities(U)
    return [set(c) for c in comms]


def candidate_groups(G: nx.DiGraph, resolution: float = 1.0, seed: int = 42,
                     max_size: int = MAX_RING_SIZE, _depth: int = 0):
    """Partition the graph into ring-sized communities, refining hierarchically.

    Louvain is used because it is deterministic given a seed, near-linear, and
    needs no guess at the number of communities — a bank does not know how many
    rings it has.

    One pass is not enough. At default resolution Louvain keeps a tight ring
    intact but merges it with the loosely-attached background traffic around it,
    so a 7-account ring arrives inside a 150-account community and no size gate
    can see it. Communities larger than a plausible ring are therefore split
    again at higher resolution, which favours smaller, denser groups, until they
    are ring-sized or refinement stops making progress.

    Both the coarse group and its refined children are returned. A ring that is
    genuinely large should still be scoreable as one object, and the scoring
    gates decide between them rather than this function guessing.
    """
    comms = _louvain(G, resolution, seed)
    out: list[set] = []
    for c in comms:
        if len(c) <= max_size or _depth >= MAX_REFINE_DEPTH:
            out.append(c)
            continue
        sub = G.subgraph(c)
        children = candidate_groups(sub, resolution * 2.0, seed, max_size, _depth + 1)
        # No progress means the group is genuinely one dense blob; keep it whole
        # rather than recursing until the depth cap.
        if not children or (len(children) == 1 and len(children[0]) == len(c)):
            out.append(c)
        else:
            out.append(c)
            out.extend(children)
    # De-duplicate: refinement can surface the same set by two routes.
    seen, uniq = set(), []
    for c in out:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def score_group(G: nx.DiGraph, members: set, roles: dict,
                U: nx.Graph | None = None, total_volume: float | None = None) -> dict:
    """Assemble every piece of evidence about one candidate group.

    `U` and `total_volume` are the undirected projection and its total degree.
    They are passed in so the caller builds them once for the whole graph rather
    than once per candidate.
    """
    if U is None:
        U = G.to_undirected()
    if total_volume is None:
        total_volume = 2.0 * U.number_of_edges()
    k = len(members)
    density = _internal_density(G, members)
    conductance = _conductance(U, members, total_volume)
    cycles = _cycle_count(G, members)
    passthrough = _passthrough(G, members)
    role_score = _role_evidence(roles, members)

    # Normalise each to 0-1 in the direction where "more" means "more ring-like".
    isolation = max(0.0, 1.0 - conductance)
    cycle_score = min(cycles / max(k, 1), 1.0)

    ring_score = (W_DENSITY * min(density / 0.5, 1.0)
                  + W_ISOLATION * isolation
                  + W_CYCLES * cycle_score
                  + W_PASSTHROUGH * passthrough
                  + W_ROLES * role_score)

    return {
        "members": sorted(members, key=str),
        "size": k,
        "internal_density": round(density, 4),
        "conductance": round(conductance, 4),
        "isolation": round(isolation, 4),
        "cycles_within": cycles,
        "passthrough": round(passthrough, 4),
        "role_evidence": round(role_score, 4),
        "roles": {str(m): roles.get(m, "ORDINARY") for m in sorted(members)},
        "ring_score": round(ring_score, 4),
    }


def detect_rings(edges, n_nodes: int | None = None, *,
                 min_size: int = MIN_RING_SIZE, max_size: int = MAX_RING_SIZE,
                 min_score: float = 0.0, seed: int = 42) -> dict:
    """Find candidate mule rings from structure alone. No labels are used.

    Returns candidates ranked by `ring_score`, each with the evidence that
    produced it, so an investigator can see why a group was surfaced rather than
    being handed an opaque cluster id.
    """
    G = build_graph(edges, n_nodes)
    if G.number_of_edges() == 0:
        return {"graph_possible": False, "reason": "no edges supplied",
                "rings": [], "n_nodes": G.number_of_nodes()}

    roles = structural_roles(G)
    groups = candidate_groups(G, seed=seed)

    # Built once, not once per candidate.
    U = G.to_undirected()
    total_volume = 2.0 * U.number_of_edges()

    scored = []
    for members in groups:
        if not (min_size <= len(members) <= max_size):
            continue
        rec = score_group(G, members, roles, U, total_volume)
        # Two hard gates before scoring is even consulted. A group that is not
        # separable from the rest of the graph, or barely connected internally,
        # is a slice of ordinary traffic no matter how its other numbers land.
        if rec["conductance"] > CONDUCTANCE_CEILING:
            continue
        if rec["internal_density"] < MIN_INTERNAL_DENSITY:
            continue
        if rec["ring_score"] < min_score:
            continue
        scored.append(rec)

    scored.sort(key=lambda r: -r["ring_score"])
    role_counts: dict[str, int] = defaultdict(int)
    for r in roles.values():
        role_counts[r] += 1

    return {
        "graph_possible": True,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities_examined": len(groups),
        "n_rings_flagged": len(scored),
        "accounts_in_flagged_rings": sum(r["size"] for r in scored),
        "role_distribution": dict(role_counts),
        "gates": {
            "min_size": min_size, "max_size": max_size,
            "conductance_ceiling": CONDUCTANCE_CEILING,
            "min_internal_density": MIN_INTERNAL_DENSITY,
        },
        "weights": {"density": W_DENSITY, "isolation": W_ISOLATION,
                    "cycles": W_CYCLES, "passthrough": W_PASSTHROUGH,
                    "roles": W_ROLES},
        "rings": scored,
    }


def account_ring_scores(result: dict) -> dict:
    """Flatten ring findings to a per-account score, for blending into risk."""
    out: dict = {}
    for ring in result.get("rings", []):
        for m in ring["members"]:
            out[m] = max(out.get(m, 0.0), ring["ring_score"])
    return out


# ==========================================================================
# CLI
# ==========================================================================
def _load_edges_csv(path: pathlib.Path):
    import csv
    edges = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_amount = header is not None and len(header) >= 3
        for row in reader:
            if len(row) < 2:
                continue
            if has_amount and len(row) >= 3:
                edges.append((int(row[0]), int(row[1]), float(row[2])))
            else:
                edges.append((int(row[0]), int(row[1])))
    return edges


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural mule-ring detection")
    ap.add_argument("--edges", required=True, help="CSV of src,dst[,amount]")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    edges = _load_edges_csv(pathlib.Path(args.edges))
    res = detect_rings(edges, min_score=args.min_score)
    print(f"nodes {res.get('n_nodes')}  edges {res.get('n_edges')}  "
          f"communities {res.get('n_communities_examined')}  "
          f"rings flagged {res.get('n_rings_flagged')}")
    for r in res["rings"][:10]:
        print(f"  score {r['ring_score']:.3f}  size {r['size']:2d}  "
              f"density {r['internal_density']:.2f}  conductance {r['conductance']:.2f}  "
              f"cycles {r['cycles_within']:3d}  passthrough {r['passthrough']:.2f}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
