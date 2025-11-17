"""Test plotting module."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pygenstability import plotting

DATA = Path(__file__).absolute().parent / "data"


def test_plot_scan(results, tmp_path):
    plotting.plot_scan(results, figure_name=tmp_path / "scan.pdf")
    assert (tmp_path / "scan.pdf").exists()

    assert (
        plotting.plot_scan(
            results, use_plotly=True, live=False, plotly_filename=str(tmp_path / "scan.html")
        )
        is not None
    )
    plt.close('all')


def test_plot_clustered_adjacency(graph, results, tmp_path):
    plotting.plot_clustered_adjacency(
        graph.toarray(), results, 0, figure_name=tmp_path / "clustered_adjacency.pdf"
    )
    assert (tmp_path / "clustered_adjacency.pdf").exists()
    plt.close('all')


def test_plot_communities(graph_nx, results, tmp_path):
    np.random.seed(42)
    plotting.plot_communities(graph_nx, results, tmp_path / "communities")
    assert (tmp_path / "communities/scale_0.pdf").exists()
    plt.close('all')


def test_plot_communities_matrix(graph, results, tmp_path):
    np.random.seed(42)
    plotting.plot_communities_matrix(graph.toarray(), results, tmp_path / "communities_matrix")
    assert (tmp_path / "communities_matrix/scale_0.pdf").exists()
    plt.close('all')

def test_plot_optimal_partitions(graph_nx, results, tmp_path):
    np.random.seed(42)
    results["selected_partitions"] = [1]
    plotting.plot_optimal_partitions(graph_nx, results, folder=tmp_path / "partitions")
    assert (tmp_path / "partitions/scale_1.pdf").exists()
    plt.close('all')
