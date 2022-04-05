"""Utils for tests."""
import pytest
import networkx as nx

collect_ignore = ["setup.py"]

@pytest.fixture()
def graph():
    """Create barbell graph."""
    return nx.adjacency_matrix(nx.barbell_graph(10, 2), dtype=float)


