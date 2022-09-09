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
    results = pgs.run(graph_directed)
    results = pgs.run(graph_signed)

    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(graph_signed, constructor=constructor)

    results = pgs.run(graph, with_optimal_scales=False)
    results = _to_list(results)
    # yaml.dump(results, open(DATA / "test_run_default.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_default.yaml", "r"))
    diff(expected_results, results)

    results = pgs.run(graph, with_spectral_gap=True, with_optimal_scales=False)
    results = _to_list(results)
    # yaml.dump(results, open(DATA / "test_run_gap.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_gap.yaml", "r"))
    diff(expected_results, results)

    results = pgs.run(
        graph,
        with_VI=False,
        with_postprocessing=False,
        with_ttprime=False,
        with_optimal_scales=False,
    )
    results = _to_list(results)
    # yaml.dump(results, open(DATA / "test_run_minimal.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_minimal.yaml", "r"))
    diff(expected_results, results)

    results = pgs.run(graph, times=[1, 2, 3, 4], log_time=False, with_optimal_scales=False)
    results = _to_list(results)
    # yaml.dump(results, open(DATA / "test_run_times.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_run_times.yaml", "r"))
    diff(expected_results, results)


def test__get_times():
    """Test _get_time."""
    assert_almost_equal(pgs._get_times(n_time=3, log_time=True), [0.01, 0.17782794, 3.16227766])
    assert_almost_equal(pgs._get_times(n_time=3, log_time=False), [-2.0, -0.75, 0.5])


def test__evaluate_NVI():
    """Test _evaluatae_NVI."""
    assert pgs._evaluate_NVI([0, 1], [[1, 1, 1, 1], [1, 1, 1, 1]]) == 0.0
