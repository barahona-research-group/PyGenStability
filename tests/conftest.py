"""Utils for tests."""
import networkx as nx
import pytest


@pytest.fixture()
def graph():
    """Create barbell graph."""
    return nx.adjacency_matrix(nx.barbell_graph(10, 2), dtype=float)
