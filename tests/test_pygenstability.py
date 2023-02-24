"""Test main pygenstability module."""
from pathlib import Path
import pytest
from numpy.testing import assert_almost_equal

import numpy as np
import yaml
from dictdiffer import diff

from pygenstability.constructors import load_constructor
from pygenstability import pygenstability as pgs

DATA = Path(__file__).absolute().parent / "data"


def _to_list(data):
    """Convert dict to list with floats for yaml encoding."""
    data.pop("NVI", None)  # NVI computation is unstable, we don't test it
    data.pop("ttprime", None)  # ttprime computation is unstable, we don't test it
    for key, val in data.items():
        if isinstance(val, dict):
            data[key] = _to_list(data[key])
        if isinstance(val, (np.ndarray, list)):
            data[key] = np.array(val, dtype=float).tolist()
    return data


def test_run(graph, graph_non_connected, graph_directed, graph_signed):
    """Test main run function."""

    # test some warnings/raises
    with pytest.raises(Exception):
        results = pgs.run(graph_non_connected)

    with pytest.raises(Exception):
        results = pgs.run(graph, exp_comp_mode="UNKNOWN")

    results = pgs.run(graph_directed, min_scale=-1, max_scale=0, n_scale=5)

    # test we don't use spectral for directed
    results = pgs.run(graph_directed, min_scale=-1, max_scale=0, n_scale=5, constructor="directed")
    assert results is not None

    # test we don't use spectral for linearized_directed
    results = pgs.run(
        graph_directed, min_scale=-1, max_scale=0, n_scale=5, constructor="linearized_directed"
    )
    assert results is not None
    results = pgs.run(graph_signed, min_scale=-1, max_scale=0, n_scale=5)

    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(graph_signed, min_scale=-1, max_scale=0, n_scale=5, constructor=constructor)

    results = pgs.run(graph, min_scale=-1, max_scale=0, n_scale=5, with_optimal_scales=False)
    results = _to_list(results)
    # yaml.dump(results, open(DATA / "test_run_default.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_default.yaml", "r"))
    assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    results = pgs.run(
        graph,
        min_scale=-2,
        max_scale=-1,
        n_scale=5,
        with_spectral_gap=True,
        with_optimal_scales=False,
    )
    results = _to_list(results)
    results["community_id"].pop(2)  # unstable
    # yaml.dump(results, open(DATA / "test_run_gap.yaml", "w"))
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
    )
    results = _to_list(results)
    # yaml.dump(results, open(DATA / "test_run_minimal.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_minimal.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0

    results = pgs.run(graph, scales=[0.1, 0.5, 1.0], log_scale=False, with_optimal_scales=False)
    results = _to_list(results)
    results["community_id"].pop(1)  # unstable
    # yaml.dump(results, open(DATA / "test_run_times.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_times.yaml", "r"))
    assert len(list(diff(expected_results, results))) == 0

    # test leiden method
    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(
        graph_signed, min_scale=-1, max_scale=0, n_scale=5, constructor=constructor, method="leiden"
    )

    results = pgs.run(
        graph, min_scale=-1, max_scale=0, n_scale=5, with_optimal_scales=False, method="leiden"
    )
    results = _to_list(results)
    results["community_id"].pop(2)  # unstable
    # yaml.dump(results, open(DATA / "test_run_default_leiden.yaml", "w"))
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
    quality = pgs._evaluate_quality(community_id, qualities_index, data["null_model"], 0)
    assert_almost_equal(quality, 0.5590341906608186)
