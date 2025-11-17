"""Test dataclustering module."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from dictdiffer import diff

from pygenstability import DataClustering, optimal_scales

from .utils import _to_list

DATA = Path(__file__).absolute().parent / "data"

optimal_scales.THRESHOLD = -0.3

entries = ['block_nvi', 'community_id', 'number_of_communities', 'run_params', 'scales', 'selected_partitions', 'stability',]
def test_DataClustering_default(X):
    """Test the DataClustering class"""
    # fixing seed and n_workers=1 makes it reproducible
    np.random.seed(42)
    clustering = DataClustering(n_tries=10, n_workers=1)
    res = clustering.fit(X)
    for entry in entries:
        assert entry in res.results_

    # this is unstable due to not consistent rng  in C++
    #results = _to_list({"labels": res.labels_, "results": res.results_})
    #yaml.dump(results, open(DATA / "test_dataclustering_default.yaml", "w"))
    #expected_results = yaml.safe_load(open(DATA / "test_dataclustering_default.yaml", "r"))
    #assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    # scales selection
    scales = clustering.scale_selection(store_basins=True)
    assert len(scales) > 2
    assert 'basin_centers' in clustering.results_

    # test plots
    clustering.plot_scan(live=False)
    clustering.plot_robust_partitions(x_coord=X[:, 0], y_coord=X[:, 1], show=False)
    clustering.plot_robust_partitions(x_coord=X[:, 0], y_coord=X[:, 1], show=True)
    clustering.plot_sankey(live=False)

    clustering.results_ = {} 
    assert clustering.plot_scan() == None

def test_DataClustering_with_knn(X):
    """with knn"""
    # fixing seed and n_workers=1 makes it reproducible
    np.random.seed(42)
    clustering = DataClustering(n_tries=10, n_workers=1, graph_method="knn-mst")
    res = clustering.fit(X)
    for entry in entries:
        assert entry in res.results_

    #results = _to_list({"labels": res.labels_, "results": res.results_})
    #yaml.dump(results, open(DATA / "test_dataclustering_knn.yaml", "w"))
    #expected_results = yaml.safe_load(open(DATA / "test_dataclustering_knn.yaml", "r"))
    #assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

def test_DataClustering_precomputed(X):
    """precomputed"""
    clustering = DataClustering(n_tries=10, graph_method="precomputed")
    with pytest.raises(AssertionError):
        clustering.fit(X)

def test_DataClustering_precomputed_default(X):
    """precompute it as default"""
    # fixing seed and n_workers=1 makes it reproducible
    np.random.seed(42)
    clustering = DataClustering(n_tries=10, n_workers=1, graph_method="precomputed")
    clustering.method = "cknn-mst"
    X = clustering.get_graph(X)
    clustering.method = "precomputed"
    res = clustering.fit(X)
    for entry in entries:
        assert entry in res.results_

    #results = _to_list({"labels": res.labels_, "results": res.results_})
    #expected_results = yaml.safe_load(open(DATA / "test_dataclustering_default.yaml", "r"))
    #assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0
