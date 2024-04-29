"""Utils for tests."""
import networkx as nx
import numpy as np
import pytest

from pygenstability.constructors import load_constructor
from pygenstability import pygenstability as pgs


@pytest.fixture()
def graph_nx():
    """Create barbell graph."""
    return nx.barbell_graph(10, 2)


@pytest.fixture()
def graph(graph_nx):
    """Create barbell graph."""
    return nx.to_scipy_sparse_array(graph_nx, dtype=float)


@pytest.fixture()
def graph_non_connected():
    """Create barbell graph."""
    graph_nx = nx.barbell_graph(10, 2)
    graph_nx.add_node(len(graph_nx))
    return nx.to_scipy_sparse_array(graph_nx, dtype=float)


@pytest.fixture()
def graph_directed():
    """Create barbell graph."""
    return nx.to_scipy_sparse_array(nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)]), dtype=float)


@pytest.fixture()
def graph_signed():
    """Create barbell graph."""
    graph_nx = nx.barbell_graph(10, 2)
    graph_nx[0][1]["weight"] = -1
    return nx.to_scipy_sparse_array(graph_nx, dtype=float)


@pytest.fixture()
def results(graph):
    constructor = load_constructor("continuous_combinatorial", graph)
    return pgs.run(graph, constructor=constructor)


def generate_circles(
    n_samples_out=300,
    n_groups_out=3,
    gap_out=np.pi / 15,
    n_samples_in=300,
    n_groups_in=3,
    gap_in=np.pi / 8,
    offset_in=np.pi / 7,
    factor=0.5,
    noise=0.03,
    seed=42,
):
    """Generate two circles with multiscale structure.

    Adapted from: sklearn.datasets.make_circles
    """
    rng = np.random.default_rng(seed)

    # generate outer circle that is split into groups
    linspace_out = []
    for i in range(n_groups_out):
        linspace_out += list(
            np.linspace(
                i * 2 / n_groups_out * np.pi + gap_out / 2,
                (i + 1) * 2 / n_groups_out * np.pi - gap_out / 2,
                int(n_samples_out / n_groups_out),
                endpoint=False,
            )
        )
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)

    # generate inner circle that is split into groups
    linspace_in = []
    for j in range(n_groups_out):
        linspace_in += list(
            np.linspace(
                j * 2 / n_groups_in * np.pi + gap_in / 2,
                (j + 1) * 2 / n_groups_in * np.pi - gap_in / 2,
                int(n_samples_in / n_groups_in),
                endpoint=False,
            )
            + offset_in
        )
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    # combine circles
    X = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T

    # create group labels
    y = np.zeros(n_samples_out + n_samples_in)
    for i in range(n_groups_in):
        y[int(i * n_samples_in / n_groups_in) : int((i + 1) * n_samples_in / n_groups_in)] = i
    for i in range(n_groups_out):
        y[
            int(n_samples_in + i * n_samples_out / n_groups_out) : int(
                n_samples_in + (i + 1) * n_samples_out / n_groups_out
            )
        ] = (i + n_groups_in)

    # add noise to data
    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)

    return X, y

@pytest.fixture()
def X():
    X, y = generate_circles()
    return X
