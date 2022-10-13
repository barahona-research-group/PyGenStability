"""Test plotting module."""
from pathlib import Path
from diff_pdf_visually import pdfdiff
import filecmp

from pygenstability import plotting

DATA = Path(__file__).absolute().parent / "data"


def test_plot_scan(results, tmp_path):
    plotting.plot_scan(results, figure_name=tmp_path / "scan.pdf")
    pdfdiff(DATA / "scan.pdf", tmp_path / "scan.pdf")

    plotting.plot_scan(
        results, use_plotly=True, live=False, plotly_filename=str(tmp_path / "scan.html")
    )
    filecmp.cmp(DATA / "scan.html", tmp_path / "scan.html")


def test_plot_clustered_adjacency(graph, results, tmp_path):
    plotting.plot_clustered_adjacency(
        graph.toarray(), results, 0, figure_name=tmp_path / "clustered_adjacency.pdf"
    )
    pdfdiff(DATA / "clustered_adjacency.pdf", tmp_path / "clustered_adjacency.pdf")


def test_plot_communities(graph_nx, results, tmp_path):
    plotting.plot_communities(graph_nx, results, tmp_path / "communities")
    pdfdiff(DATA / "scale_0.pdf", tmp_path / "communities/scale_0.pdf")
