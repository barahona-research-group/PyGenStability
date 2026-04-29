"""Test constructor module."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import yaml
from numpy.testing import assert_almost_equal

from pygenstability import constructors

CONSTRUCTORS = [
    "linearized",
    "continuous_combinatorial",
    "continuous_normalized",
    "signed_modularity",
    "directed",
    "linearized_directed",
    "signed_combinatorial",
]
DATA = Path(__file__).absolute().parent / "data"


def _list_data(data):
    data["quality"] = np.array(data["quality"].toarray(), dtype=float).tolist()
    data["null_model"] = np.array(data["null_model"], dtype=float).tolist()
    return data


def test_load_constructor(graph):
    """Test load constructors."""
    with pytest.raises(Exception):
        constructors.load_constructor(CONSTRUCTORS[0], None)

    with pytest.raises(Exception):
        constructors.load_constructor("WRONG", graph)

    with pytest.raises(Exception):
        constructors.load_constructor([1, 2, 3], graph)

    constructor = constructors.load_constructor(CONSTRUCTORS[0], graph)

    assert constructors.load_constructor(constructor, graph) == constructor

    for constr in CONSTRUCTORS:
        data = _list_data(
            constructors.load_constructor(constr, graph, exp_comp_mode="expm").get_data(1)
        )
        # yaml.dump(data, open(DATA / f"test_constructor_{constr}.yaml", "w"))
        expected_data = yaml.safe_load(open(DATA / f"test_constructor_{constr}.yaml", "r"))
        assert_almost_equal(data["quality"], expected_data["quality"])
        assert_almost_equal(data["null_model"], expected_data["null_model"])

    for constr in CONSTRUCTORS:
        data = _list_data(
            constructors.load_constructor(
                constr, graph, with_spectral_gap=True, exp_comp_mode="expm"
            ).get_data(1)
        )
        # yaml.dump(data, open(DATA / f"test_constructor_{constr}_gap.yaml", "w"))
        expected_data = yaml.safe_load(open(DATA / f"test_constructor_{constr}_gap.yaml", "r"))
        assert_almost_equal(data["quality"], expected_data["quality"])
        assert_almost_equal(data["null_model"], expected_data["null_model"])


def test_linearized_directed(graph_directed):
    data = _list_data(
        constructors.load_constructor(
            "linearized_directed", graph_directed, exp_comp_mode="expm", alpha=1.0
        ).get_data(1)
    )
    # yaml.dump(data, open(DATA / "test_constructor_linearized_directed_alpha.yaml", "w"))
    expected_data = yaml.safe_load(
        open(DATA / "test_constructor_linearized_directed_alpha.yaml", "r")
    )
    assert_almost_equal(data["quality"], expected_data["quality"])
    assert_almost_equal(data["null_model"], expected_data["null_model"])


def test_spectral_exp(graph):
    """Test spectral exp computation."""
    _skip = [
        "directed",
        "linearized_directed",
        "signed_combinatorial",  # unstable computations on github
    ]
    for constr in CONSTRUCTORS:
        if constr not in _skip:
            data = _list_data(
                constructors.load_constructor(constr, graph, exp_comp_mode="spectral").get_data(1)
            )
            # yaml.safe_dump(data, open(DATA / f"test_constructor_spectral_{constr}.yaml", "w"))
            expected_data = yaml.safe_load(
                open(DATA / f"test_constructor_spectral_{constr}.yaml", "r")
            )
            assert_almost_equal(data["quality"], expected_data["quality"])
            assert_almost_equal(data["null_model"], expected_data["null_model"])

    with pytest.raises(Exception):
        data = _list_data(
            constructors.load_constructor("directed", graph, exp_comp_mode="spectral").get_data(1)
        )


def _dense_linearized_directed_reference(graph, alpha, scale):
    """Dense reference for linearized_directed: returns (B_ref, pi_ref).

    B_ref is the full N*N modularity-style matrix, computed via the original
    O(N^2) formula. The sparse implementation must match B = quality - sum of
    rank-1 null pairs at machine epsilon.
    """
    n = graph.shape[0]
    out = np.array(graph.sum(1)).flatten()
    dinv = np.where(out > 0, 1.0 / np.maximum(out, 1), 0.0)
    A = graph.toarray()
    M_minus_I = (
        alpha * np.diag(dinv) @ A
        + ((1 - alpha) * np.eye(n) + np.diag(alpha * (dinv == 0.0))) @ (np.ones((n, n)) / n)
        - np.eye(n)
    )
    pi = np.abs(sp.linalg.eigs(M_minus_I.T, which="SM", k=1)[1][:, 0])
    pi /= pi.sum()
    F = np.diag(pi) @ M_minus_I
    B = scale * F - np.outer(pi, pi)
    return B, pi


@pytest.mark.parametrize("alpha", [0.5, 0.8, 0.99])
def test_linearized_directed_alpha_lt_one_matches_dense(graph_directed, alpha):
    """Sparse alpha<1 path must match the dense O(N^2) reference at machine eps."""
    B_ref, pi_ref = _dense_linearized_directed_reference(graph_directed, alpha, scale=1.0)
    data = constructors.load_constructor(
        "linearized_directed", graph_directed, alpha=alpha
    ).get_data(1.0)
    Q = data["quality"].toarray()
    nm = data["null_model"]
    rank1 = sum(np.outer(nm[k], nm[k + 1]) for k in range(0, len(nm), 2))
    np.testing.assert_allclose(Q - rank1, B_ref, atol=1e-10)
    np.testing.assert_allclose(np.abs(nm[0]), np.abs(pi_ref), atol=1e-8)
    assert nm.shape == (4, graph_directed.shape[0])


@pytest.mark.parametrize("alpha", [0.5, 0.85])
def test_linearized_directed_alpha_lt_one_dangling(alpha):
    """Sparse path must handle dangling sinks (out-degree zero) correctly."""
    g = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    g.add_node(3)
    graph = nx.to_scipy_sparse_array(g, dtype=float, nodelist=range(4))
    B_ref, pi_ref = _dense_linearized_directed_reference(graph, alpha, scale=1.0)
    data = constructors.load_constructor("linearized_directed", graph, alpha=alpha).get_data(1.0)
    Q = data["quality"].toarray()
    nm = data["null_model"]
    rank1 = sum(np.outer(nm[k], nm[k + 1]) for k in range(0, len(nm), 2))
    np.testing.assert_allclose(Q - rank1, B_ref, atol=1e-10)
    np.testing.assert_allclose(np.abs(nm[0]), np.abs(pi_ref), atol=1e-8)


def test_linearized_directed_alpha_lt_one_sparse_memory():
    """alpha<1 path must keep memory linear in N: no dense N*N allocation."""
    rng = np.random.default_rng(0)
    n = 10_000
    rows = np.concatenate([np.arange(n), rng.integers(0, n, size=5 * n)])
    cols = np.concatenate([(np.arange(n) + 1) % n, rng.integers(0, n, size=5 * n)])
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    data = constructors.load_constructor("linearized_directed", G, alpha=0.85).get_data(1.0)
    assert sp.issparse(data["quality"])
    # quality.nnz <= nnz(A) + N (off-diagonal from A, diagonal from -I)
    assert data["quality"].nnz <= G.nnz + n
    # null model has 4 vectors (2 pairs) for alpha<1
    assert data["null_model"].shape == (4, n)


def test__total_degree():
    """Test check total degree."""
    with pytest.raises(Exception):
        constructors._check_total_degree(np.array([1, 1, -3]))
