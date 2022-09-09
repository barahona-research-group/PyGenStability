"""Utils for tests."""
import networkx as nx
import pytest


@pytest.fixture()
def graph():
    """Create barbell graph."""
    return nx.to_scipy_sparse_matrix(nx.barbell_graph(10, 2), dtype=float)
    g= nx.adjacency_matrix(nx.barbell_graph(10, 2), dtype=float)
    print(type(graph), type(g))
    raise
