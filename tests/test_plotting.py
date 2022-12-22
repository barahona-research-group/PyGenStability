"""Test plotting module."""
import numpy as np
from pathlib import Path
from diff_pdf_visually import pdf_similar

from pygenstability import plotting

DATA = Path(__file__).absolute().parent / "data"


def test_plot_scan(results, tmp_path):
    plotting.plot_scan(results, figure_name=tmp_path / "scan.pdf")
    assert pdf_similar(DATA / "scan.pdf", tmp_path / "scan.pdf")

    assert (
        plotting.plot_scan(
            results, use_plotly=True, live=False, plotly_filename=str(tmp_path / "scan.html")
        )
        is not None
    )


def test_plot_clustered_adjacency(graph, results, tmp_path):
    plotting.plot_clustered_adjacency(
        graph.toarray(), results, 0, figure_name=tmp_path / "clustered_adjacency.pdf"
    )
    assert pdf_similar(
        DATA / "clustered_adjacency.pdf", tmp_path / "clustered_adjacency.pdf", threshold=15
    )


def test_plot_communities(graph_nx, results, tmp_path):
    np.random.seed(42)
    plotting.plot_communities(graph_nx, results, tmp_path / "communities")
    assert pdf_similar(DATA / "scale_0.pdf", tmp_path / "communities/scale_0.pdf")


def test_plot_optimal_partitions(graph_nx, results, tmp_path):
    np.random.seed(42)
    results["selected_partitions"] = [1]
    plotting.plot_optimal_partitions(graph_nx, results, folder=tmp_path / "partitions")
    assert pdf_similar(DATA / "scale_1.pdf", tmp_path / "partitions/scale_1.pdf", threshold=10)
