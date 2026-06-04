"""Test main pygenstability module."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from dictdiffer import diff
from numpy.testing import assert_almost_equal

from pygenstability import pygenstability as pgs
from pygenstability.constructors import load_constructor

from .utils import _to_list

DATA = Path(__file__).absolute().parent / "data"


def test_run(graph, graph_non_connected, graph_directed, graph_signed):
    """Test main run function."""
    np.random.seed(42)

    # test some warnings/raises
    with pytest.raises(Exception):
        results = pgs.run(graph_non_connected)

    with pytest.raises(Exception):
        results = pgs.run(graph, exp_comp_mode="UNKNOWN")

    results = pgs.run(graph_directed, min_scale=-1, max_scale=0, n_scale=5, n_tries=10)
    assert results is not None

    # test with_all_tries
    results = pgs.run(graph_directed, min_scale=-1, max_scale=0, n_scale=5, with_all_tries=True, n_tries=10)
    assert "all_tries" in results

    # test we don't use spectral for directed
    results = pgs.run(graph_directed, min_scale=-1, max_scale=0, n_scale=5, constructor="directed", n_tries=10)
    assert results is not None

    # test we don't use spectral for linearized_directed
    results = pgs.run(
        graph_directed, min_scale=-1, max_scale=0, n_scale=5, constructor="linearized_directed", n_tries=10
    )
    assert results is not None
    results = pgs.run(graph_signed, min_scale=-1, max_scale=0, n_scale=5, n_tries=10)

    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(graph_signed, min_scale=-1, max_scale=0, n_scale=5, constructor=constructor, n_tries=10)

    results = pgs.run(graph, min_scale=-1, max_scale=0, n_scale=5, with_optimal_scales=False, n_tries=10, n_workers=1, seed=42)
    results = _to_list(results)
    #yaml.dump(results, open(DATA / "test_run_default.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_default.yaml", "r"))
    assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    results = pgs.run(
        graph,
        min_scale=-2,
        max_scale=-1,
        n_scale=5,
        with_spectral_gap=True,
        with_optimal_scales=False,
        n_tries=10,
        seed=42,
    )
    results = _to_list(results)
    #yaml.dump(results, open(DATA / "test_run_gap.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_gap.yaml", "r"))
    assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    results = pgs.run(
        graph,
        min_scale=-1,
        max_scale=0,
        n_scale=5,
        with_NVI=False,
        with_postprocessing=False,
        with_ttprime=False,
        with_optimal_scales=False,
        n_tries=10,
        n_workers=1,
        seed=42,
    )
    results = _to_list(results)
    #yaml.dump(results, open(DATA / "test_run_minimal.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_minimal.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0

    results = pgs.run(graph, scales=[0.1, 0.5, 1.0], log_scale=False, with_optimal_scales=False, n_tries=10, n_workers=1, seed=42)
    results = _to_list(results)
    # regenerate fixture (run manually after deliberate numerics changes):
    #yaml.dump(results, open(DATA / "test_run_times.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_times.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0

    # test leiden method
    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(
        graph_signed, min_scale=-1, max_scale=0, n_scale=5, constructor=constructor, method="leiden", n_tries=10, n_workers=1, seed=42,
    )
    assert results is not None

    results = pgs.run(
        graph, min_scale=-1, max_scale=0, n_scale=5, with_optimal_scales=False, method="leiden", n_tries=10, n_workers=1, seed=42,
    )
    results = _to_list(results)
    #yaml.dump(results, open(DATA / "test_run_default_leiden.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_default_leiden.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0


def test__get_scales():
    """Test _get_scales."""
    assert_almost_equal(pgs._get_scales(n_scale=3, log_scale=True), [0.01, 0.17782794, 3.16227766])
    assert_almost_equal(pgs._get_scales(n_scale=3, log_scale=False), [-2.0, -0.75, 0.5])


def test_evaluate_NVI():
    """Test evaluate_NVI."""
    assert pgs.evaluate_NVI([0, 1], [[1, 1, 1, 1], [1, 1, 1, 1]]) == 0.0
    assert pgs.evaluate_NVI([0, 1], [[0, 0, 1, 1], [1, 1, 1, 1]]) == 1.0


def test__optimise(graph):
    constructor = load_constructor("continuous_combinatorial", graph)
    data = constructor.get_data(1)
    quality_indices, quality_values = pgs._to_indices(data["quality"])
    stability, community_id = pgs._optimise(
        0, 42, quality_indices, quality_values, data["null_model"], 0
    )
    assert_almost_equal(stability, 0.5590341906608186)
    assert community_id == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    stability, community_id = pgs._optimise(
        0, 42, quality_indices, quality_values, data["null_model"], 0, method="leiden"
    )
    assert_almost_equal(stability, 0.36540825919902664)
    assert community_id == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        0,
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]


def test__evaluate_quality(graph):
    constructor = load_constructor("continuous_combinatorial", graph)
    data = constructor.get_data(1)
    community_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    qualities_index = pgs._to_indices(data["quality"])
    quality = pgs._evaluate_quality(community_id, qualities_index[0], qualities_index[1], data["null_model"], 0)
    assert_almost_equal(quality, 0.5590341906608186)

    quality = pgs._evaluate_quality(
        community_id, qualities_index[0], qualities_index[1], data["null_model"], 0, method="leiden"
    )
    assert_almost_equal(quality, 0.2741359784037568)


def test__evaluate_quality_leiden_asymmetric_null_is_swap_invariant(graph):
    """Leiden must depend only on the symmetric part of the null model.

    The null contribution ``sum_c S_a(c) S_b(c)`` is invariant under swapping the two
    vectors of a pair, so Leiden quality for ``null=[a, b]`` must equal that for
    ``null=[b, a]``. The previous implementation only read ``null_model[::2]`` (the first
    vector of each pair), so it failed this for asymmetric/signed pairs.
    """
    quality_indices, quality_values = pgs._to_indices(graph)
    n = graph.shape[0]
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)
    community_id = list(rng.integers(0, 3, size=n))

    q_ab = pgs._evaluate_quality(
        community_id, quality_indices, quality_values, np.array([a, b]), 0, method="leiden"
    )
    q_ba = pgs._evaluate_quality(
        community_id, quality_indices, quality_values, np.array([b, a]), 0, method="leiden"
    )
    assert_almost_equal(q_ab, q_ba)


def test__evaluate_quality_leiden_multi_pair_matches_louvain(graph):
    """For ``n_null >= 2`` the Leiden null term must match the Louvain backend (issue #111).

    Isolate the null contribution from the (backend-dependent) edge term by comparing
    ``Q(nm1) - Q(nm2)``, where ``nm2`` duplicates the single symmetric pair ``nm1``
    (``n_null = 2``); this difference equals the per-partition null term and must agree
    between backends.
    """
    data = load_constructor("continuous_combinatorial", graph).get_data(1)
    quality_indices, quality_values = pgs._to_indices(data["quality"])
    nm1 = np.asarray(data["null_model"])  # single symmetric pair -> n_null = 1
    nm2 = np.vstack([nm1, nm1])  # two identical pairs -> n_null = 2

    rng = np.random.default_rng(0)
    for _ in range(5):
        community_id = list(rng.integers(0, 4, size=nm1.shape[1]))
        deltas = {}
        for method in ("louvain", "leiden"):
            deltas[method] = pgs._evaluate_quality(
                community_id, quality_indices, quality_values, nm1, 0, method=method
            ) - pgs._evaluate_quality(
                community_id, quality_indices, quality_values, nm2, 0, method=method
            )
        assert_almost_equal(deltas["leiden"], deltas["louvain"])


def test__evaluate_quality_leiden_null_matches_definition(graph_signed):
    """Leiden's null term equals the generalized-stability definition up to a constant.

    For a signed/asymmetric null (``signed_modularity``), the null contribution computed by
    Leiden must equal ``sum_c S_a(c) S_b(c)`` summed over pairs, up to a partition-independent
    constant (so the optimised partition / argmax matches the definition). The previous
    implementation squared a single vector per pair, dropping the sign of negative-degree
    terms and giving a partition-dependent error.
    """
    data = load_constructor("signed_modularity", graph_signed).get_data(1)
    quality_indices, quality_values = pgs._to_indices(data["quality"])
    nm = np.asarray(data["null_model"])
    n = nm.shape[1]

    def true_null(community_id):
        community_id = np.asarray(community_id)
        total = 0.0
        for k in range(nm.shape[0] // 2):
            a, b = nm[2 * k], nm[2 * k + 1]
            for community in np.unique(community_id):
                mask = community_id == community
                total += a[mask].sum() * b[mask].sum()
        return total

    rng = np.random.default_rng(0)
    offsets = []
    for _ in range(6):
        community_id = list(rng.integers(0, 4, size=n))
        leiden_q = pgs._evaluate_quality(
            community_id, quality_indices, quality_values, nm, 0, method="leiden"
        )
        # zero null model -> Leiden returns the edge term only; subtract to isolate the null
        edge_q = pgs._evaluate_quality(
            community_id, quality_indices, quality_values, np.zeros_like(nm), 0, method="leiden"
        )
        offsets.append((edge_q - leiden_q) - true_null(community_id))
    assert_almost_equal(offsets, offsets[0] * np.ones(len(offsets)))


@pytest.mark.parametrize("n_workers", [1, 2])
def test_run_is_deterministic(graph, n_workers):
    """Same seed yields identical communities and stabilities across re-runs.

    Parameterised on n_workers to cover the multiprocessing.Pool ordering path
    that motivated the seed pre-generation fix.
    """
    common = dict(min_scale=-1, max_scale=0, n_scale=4, n_tries=5,
                  with_optimal_scales=False, n_workers=n_workers, seed=42)

    r1 = pgs.run(graph, **common)
    r2 = pgs.run(graph, **common)

    assert [list(c) for c in r1["community_id"]] == [list(c) for c in r2["community_id"]]
    assert list(r1["stability"]) == list(r2["stability"])
