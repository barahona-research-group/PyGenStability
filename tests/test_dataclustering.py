"""Test dataclustering module."""
import pytest
from pathlib import Path
import yaml

from pygenstability import DataClustering

from dictdiffer import diff

from .utils import _to_list

DATA = Path(__file__).absolute().parent / "data"


def test_DataClustering(X):
    """Test the DataClustering class"""
    # default
    clustering = DataClustering()
    res = clustering.fit(X)
    results = _to_list({"labels": res.labels_, "results": res.results_})
    yaml.dump(results, open(DATA / "test_dataclustering_default.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_dataclustering_default.yaml", "r"))
    assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    # with knn
    clustering = DataClustering(graph_method="knn-mst")
    res = clustering.fit(X)
    results = _to_list({"labels": res.labels_, "results": res.results_})
    yaml.dump(results, open(DATA / "test_dataclustering_knn.yaml", "w"))
    expected_results = yaml.safe_load(open(DATA / "test_dataclustering_knn.yaml", "r"))
    assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    # precomputed
    clustering = DataClustering(graph_method="precomputed")
    with pytest.raises(AssertionError):
        res = clustering.fit(X)

    # precompute it as default
    clustering.method = "cknn-mst"
    X = clustering.get_graph(X)
    clustering.method = "precomputed"

    res = clustering.fit(X)
    results = _to_list({"labels": res.labels_, "results": res.results_})
    expected_results = yaml.safe_load(open(DATA / "test_dataclustering_default.yaml", "r"))
    assert len(list(diff(expected_results, results, tolerance=1e-5))) == 0

    # scales selection
    scales = clustering.scale_selection()
    assert len(scales) > 2

    # test plots
    clustering.plot_scan(live=False)
    clustering.plot_robust_partitions(x_coord=X[:, 0], y_coord=X[:, 1], show=False)
    clustering.plot_robust_partitions(x_coord=X[:, 0], y_coord=X[:, 1], show=True)
    clustering.plot_sankey(live=False)

    clustering.results_ = None
    assert clustering.plot_scan() == None
