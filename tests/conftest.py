"""Utils for tests."""
import networkx as nx
import pytest

from pygenstability.constructors import load_constructor
from pygenstability import pygenstability as pgs


@pytest.fixture()
def graph_nx():
    """Create barbell graph."""
    graph = nx.barbell_graph(10, 2)
    pos = nx.spring_layout(graph)
    for u in graph:
        graph.nodes[u]['pos'] = pos[u]

    return graph

@pytest.fixture()
def graph(graph_nx):
    """Create barbell graph."""
    return nx.to_scipy_sparse_matrix(graph_nx, dtype=float)


@pytest.fixture()
def graph_non_connected(graph_nx):
    """Create barbell graph."""
    graph_nx.add_node(len(graph_nx))
    return nx.to_scipy_sparse_matrix(graph_nx, dtype=float)


@pytest.fixture()
def graph_directed():
    """Create barbell graph."""
    return nx.to_scipy_sparse_matrix(nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)]), dtype=float)


@pytest.fixture()
def graph_signed(graph_nx):
    """Create barbell graph."""
    graph_nx[0][1]["weight"] = -1
    return nx.to_scipy_sparse_matrix(graph_nx, dtype=float)


@pytest.fixture()
def results(graph):
    constructor = load_constructor("continuous_combinatorial", graph)
    return pgs.run(graph, constructor=constructor)
