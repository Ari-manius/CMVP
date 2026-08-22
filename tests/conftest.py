"""
Shared fixtures: small synthetic graphs covering the directed x weighted
2x2, kept tiny (~12 nodes) so the full NEMtropy-fitting matrix in
test_core_pipeline.py runs in seconds rather than minutes.
"""

import networkx as nx
import numpy as np
import pytest


def _undirected_unweighted(n=12, seed=1):
    G = nx.gnp_random_graph(n, 0.45, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    while list(nx.isolates(G)):
        G.remove_nodes_from(list(nx.isolates(G)))
    return G


def _undirected_weighted(n=12, seed=1):
    G = _undirected_unweighted(n, seed)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]['weight'] = int(rng.integers(1, 6))
    return G


def _directed_unweighted(n=12, seed=1):
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(n, 0.35, seed=seed, directed=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    # Ensure every node has both in- and out-degree >= 1 (required for dcm/decm fits).
    nodes = list(G.nodes())
    for node in nodes:
        if G.out_degree(node) == 0:
            target = nodes[(nodes.index(node) + 1) % len(nodes)]
            G.add_edge(node, target)
        if G.in_degree(node) == 0:
            source = nodes[(nodes.index(node) - 1) % len(nodes)]
            G.add_edge(source, node)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _directed_weighted(n=12, seed=1):
    G = _directed_unweighted(n, seed)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]['weight'] = int(rng.integers(1, 6))
    return G


GRAPH_BUILDERS = {
    'undirected_unweighted': _undirected_unweighted,
    'undirected_weighted': _undirected_weighted,
    'directed_unweighted': _directed_unweighted,
    'directed_weighted': _directed_weighted,
}

# Expected auto-selected NEMtropy model per graph type (core.py's model='auto' rule).
EXPECTED_AUTO_MODEL = {
    'undirected_unweighted': 'cm_exp',
    'undirected_weighted': 'ecm_exp',
    'directed_unweighted': 'dcm_exp',
    'directed_weighted': 'decm_exp',
}


@pytest.fixture(params=list(GRAPH_BUILDERS))
def graph_kind(request):
    """Parametrized graph-type label; use with GRAPH_BUILDERS[graph_kind]() to build."""
    return request.param


@pytest.fixture
def small_graph(graph_kind):
    return GRAPH_BUILDERS[graph_kind](seed=1)


@pytest.fixture
def uu_graph():
    """Undirected, unweighted — the fast default for tests that don't care about graph type."""
    return _undirected_unweighted(seed=1)


@pytest.fixture
def uw_graph():
    return _undirected_weighted(seed=1)


@pytest.fixture
def du_graph():
    return _directed_unweighted(seed=1)


@pytest.fixture
def dw_graph():
    return _directed_weighted(seed=1)
