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

    results = pgs.run(graph, min_scale=-1, max_scale=0, n_scale=5, with_optimal_scales=False, n_tries=10, n_workers=1)
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
        n_workers=1
    )
    results = _to_list(results)
    #yaml.dump(results, open(DATA / "test_run_minimal.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_minimal.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0

    results = pgs.run(graph, scales=[0.1, 0.5, 1.0], log_scale=False, with_optimal_scales=False, n_tries=10,n_workers=1)
    results = _to_list(results)
    yaml.dump(results, open(DATA / "test_run_times.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_times.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0

    # test leiden method
    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(
        graph_signed, min_scale=-1, max_scale=0, n_scale=5, constructor=constructor, method="leiden", n_tries=10, n_workers=1,
    )
    assert results is not None

    results = pgs.run(
        graph, min_scale=-1, max_scale=0, n_scale=5, with_optimal_scales=False, method="leiden", n_tries=10, n_workers=1,
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
        0, quality_indices, quality_values, data["null_model"], 0
    )
    assert_almost_equal(stability, 0.5590341906608186)
    assert community_id == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    stability, community_id = pgs._optimise(
        0, quality_indices, quality_values, data["null_model"], 0, method="leiden"
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
