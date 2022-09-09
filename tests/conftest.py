"""Utils for tests."""
import networkx as nx
import pytest


@pytest.fixture()
def graph():
    """Create barbell graph."""
    return nx.to_scipy_sparse_matrix(nx.barbell_graph(10, 2), dtype=float)


@pytest.fixture()
def graph_non_connected():
    """Create barbell graph."""
    graph = nx.barbell_graph(10, 2)
    graph.add_node(len(graph))
    return nx.to_scipy_sparse_matrix(graph, dtype=float)

@pytest.fixture()
def graph_directed():
    """Create barbell graph."""
    return nx.to_scipy_sparse_matrix(nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)]), dtype=float)

@pytest.fixture()
def graph_signed():
    """Create barbell graph."""
    graph = nx.barbell_graph(10, 2)
    graph[0][1]['weight'] = -1
    return nx.to_scipy_sparse_matrix(graph, dtype=float)



