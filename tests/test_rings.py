"""Tests for structural ring detection.

The properties worth pinning down are the ones that would let this module
embarrass the project: that it finds a planted ring without being told anything,
that it does NOT call ordinary traffic a ring, and above all that it refuses to
run when there are no edges rather than inventing a graph.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rings as R  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: two graphs with deliberately different shapes
# --------------------------------------------------------------------------
def _ring_edges(members, amount=1000.0):
    """A layering shape: collector fans out, relays chain to a cash-out."""
    collector, cashout = members[0], members[-1]
    middles = members[1:-1]
    edges = []
    for m in middles:
        edges.append((collector, m, amount))
        edges.append((m, cashout, amount * 0.98))
    for a, b in zip(middles, middles[1:]):
        edges.append((a, b, amount * 0.5))
    return edges


def _star_edges(centre, spokes, amount=500.0):
    """Ordinary retail traffic: a merchant paid by unrelated customers."""
    return [(s, centre, amount) for s in spokes]


# --------------------------------------------------------------------------
# The refusal that matters most
# --------------------------------------------------------------------------
def test_no_edges_means_no_rings():
    """With no edge list there is no graph, and none may be invented."""
    res = R.detect_rings([], n_nodes=100)
    assert res["graph_possible"] is False
    assert res["rings"] == []
    assert "no edges" in res["reason"]


def test_self_transfers_are_not_relationships():
    G = R.build_graph([(1, 1), (2, 2), (1, 2)])
    assert G.number_of_edges() == 1
    assert G.has_edge(1, 2)


def test_parallel_transfers_collapse_but_keep_value():
    G = R.build_graph([(1, 2, 100.0), (1, 2, 50.0), (1, 2, 25.0)])
    assert G.number_of_edges() == 1
    assert G[1][2]["weight"] == 3
    assert G[1][2]["amount"] == pytest.approx(175.0)


# --------------------------------------------------------------------------
# Structural roles
# --------------------------------------------------------------------------
def test_relay_is_detected_by_near_zero_retention():
    # 500 in, 495 out: keeps ~1%, which is a conduit.
    roles = R.structural_roles(R.build_graph([(1, 2, 500.0), (2, 3, 495.0)]))
    assert roles[2] == "RELAY"


def test_terminal_is_where_value_stops():
    roles = R.structural_roles(R.build_graph([(1, 2, 500.0), (3, 2, 400.0)]))
    assert roles[2] == "TERMINAL"


def test_collector_fans_in():
    edges = [(i, 99, 100.0) for i in range(6)] + [(99, 100, 10.0)]
    roles = R.structural_roles(R.build_graph(edges))
    assert roles[99] in {"COLLECTOR", "RELAY"}


def test_isolated_account_has_no_role():
    G = R.build_graph([(1, 2)], n_nodes=10)
    assert R.structural_roles(G)[7] == "ISOLATED"


# --------------------------------------------------------------------------
# Group-level evidence
# --------------------------------------------------------------------------
def test_passthrough_is_one_when_everything_leaves():
    # 1000 enters the group {2,3} from outside, ~1000 leaves it to outside.
    edges = [(1, 2, 1000.0), (2, 3, 1000.0), (3, 4, 1000.0)]
    G = R.build_graph(edges)
    assert R._passthrough(G, {2, 3}) == pytest.approx(1.0, abs=0.01)


def test_passthrough_is_low_when_the_group_accumulates():
    edges = [(1, 2, 1000.0), (2, 3, 20.0), (3, 4, 10.0)]
    G = R.build_graph(edges)
    assert R._passthrough(G, {2, 3}) < 0.2


def test_cycle_is_counted_when_money_returns():
    G = R.build_graph([(1, 2), (2, 3), (3, 1)])
    assert R._cycle_count(G, {1, 2, 3}) >= 1


def test_no_cycle_in_a_pure_chain():
    G = R.build_graph([(1, 2), (2, 3), (3, 4)])
    assert R._cycle_count(G, {1, 2, 3, 4}) == 0


def test_conductance_is_low_for_a_sealed_group():
    """A tight group with one link out is nearly sealed.

    The surrounding graph has to be substantial. NetworkX normalises
    conductance by min(volume(S), volume(outside)), so on a toy graph where the
    "outside" is a single node the denominator collapses to 1 and every group
    scores 1.0 regardless of how sealed it is. Ring candidates are always small
    against a large book, which is the regime this is used in.
    """
    background = [(100 + i, 200 + i) for i in range(60)]
    sealed = R.build_graph(
        [(1, 2), (2, 3), (3, 1), (1, 3), (2, 1), (3, 99)] + background)
    U = sealed.to_undirected()
    assert R._conductance(U, {1, 2, 3}, 2.0 * U.number_of_edges()) < 0.3


def test_conductance_is_high_for_a_group_that_only_faces_outward():
    """Three accounts with no links to each other are not a group."""
    background = [(100 + i, 200 + i) for i in range(60)]
    loose = R.build_graph([(1, 50), (2, 51), (3, 52)] + background)
    U = loose.to_undirected()
    assert R._conductance(U, {1, 2, 3}, 2.0 * U.number_of_edges()) > 0.9


# --------------------------------------------------------------------------
# End to end: does it find a ring, and does it leave ordinary traffic alone?
# --------------------------------------------------------------------------
def test_planted_ring_is_found_without_any_label():
    ring = [10, 11, 12, 13, 14, 15]
    edges = _ring_edges(ring)
    # Background traffic so the ring has something to be distinguished from.
    edges += [(100 + i, 200 + i, 50.0) for i in range(60)]
    edges += _star_edges(500, list(range(600, 615)))

    res = R.detect_rings(edges, n_nodes=800)
    assert res["graph_possible"] is True
    assert res["rings"], "no candidate rings surfaced at all"

    top = res["rings"][0]
    found = set(top["members"]) & set(ring)
    assert len(found) >= 4, f"top candidate recovered only {found} of {ring}"


def test_ordinary_star_traffic_is_not_called_a_ring():
    """A merchant paid by 15 unrelated customers has no internal structure."""
    edges = _star_edges(500, list(range(600, 615)))
    res = R.detect_rings(edges, n_nodes=800)
    for r in res["rings"]:
        # The hub plus its spokes must never be flagged as a group.
        assert not (500 in r["members"] and len(r["members"]) > 2), r


def test_detection_is_deterministic():
    edges = _ring_edges([10, 11, 12, 13, 14]) + [(100 + i, 200 + i) for i in range(40)]
    a = R.detect_rings(edges, n_nodes=400, seed=7)
    b = R.detect_rings(edges, n_nodes=400, seed=7)
    assert [r["members"] for r in a["rings"]] == [r["members"] for r in b["rings"]]
    assert [r["ring_score"] for r in a["rings"]] == [r["ring_score"] for r in b["rings"]]


def test_every_ring_carries_its_evidence():
    """An investigator handed a group must be able to see why it was grouped."""
    edges = _ring_edges([10, 11, 12, 13, 14, 15])
    edges += [(100 + i, 200 + i) for i in range(40)]
    res = R.detect_rings(edges, n_nodes=400)
    assert res["rings"]
    for r in res["rings"]:
        for field in ("internal_density", "conductance", "cycles_within",
                      "passthrough", "role_evidence", "ring_score", "roles"):
            assert field in r, f"{field} missing from ring evidence"
        assert 0.0 <= r["ring_score"] <= 1.0


def test_account_scores_take_the_strongest_ring():
    edges = _ring_edges([10, 11, 12, 13, 14, 15])
    edges += [(100 + i, 200 + i) for i in range(40)]
    res = R.detect_rings(edges, n_nodes=400)
    scores = R.account_ring_scores(res)
    assert scores, "no per-account ring scores produced"
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    for r in res["rings"]:
        for m in r["members"]:
            assert scores[m] >= r["ring_score"] - 1e-9
