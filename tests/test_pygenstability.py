"""Test main pygenstability module."""
from dictdiffer import diff
import numpy as np
from pygenstability import pygenstability as pgs
import yaml
from utils import graph


def _to_list(data):
    """Convert dict to list with floats for yaml encoding."""
    for key, val in data.items():
        if isinstance(val, dict):
            data[key] = _to_list(data[key])
        if isinstance(val, (np.ndarray, list)):
            data[key] = np.array(val, dtype=float).tolist()
    return data


def test_run(graph):
    # test run with default params
    results = pgs.run(graph)
    results = _to_list(results)
    # yaml.dump(results, open("data/test_run_default.yaml", "w"))
    expected_results = yaml.safe_load(open("data/test_run_default.yaml", "r"))
    diff(expected_results, results)

    results = pgs.run(graph, with_spectral_gap=True)
    results = _to_list(results)
    # yaml.dump(results, open("data/test_run_gap.yaml", "w"))
    expected_results = yaml.safe_load(open("data/test_run_gap.yaml", "r"))
    diff(expected_results, results)

    results = pgs.run(graph, with_VI=False, with_postprocessing=False, with_ttprime=False)
    results = _to_list(results)
    # yaml.dump(results, open("data/test_run_minimal.yaml", "w"))
    expected_results = yaml.safe_load(open("data/test_run_minimal.yaml", "r"))
    diff(expected_results, results)

    results = pgs.run(graph, times=[1, 2, 3, 4], log_time=False)
    results = _to_list(results)
    # yaml.dump(results, open("data/test_run_times.yaml", "w"))
    expected_results = yaml.safe_load(open("data/test_run_times.yaml", "r"))
    diff(expected_results, results)
