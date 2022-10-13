"""Test plotting module."""
from diff_pdf_visually import pdfdiff
import filecmp

from pygenstability import plotting
from pygenstability.constructors import load_constructor
from pygenstability import pygenstability as pgs


def test_plot_scan(graph, tmp_path):
    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(graph, constructor=constructor)
    plotting.plot_scan(results, figure_name=tmp_path / "scan.pdf")
    pdfdiff("data/scan.pdf", tmp_path / "scan.pdf")

    plotting.plot_scan(
        results, use_plotly=True, live=False, plotly_filename=str(tmp_path / "scan.html")
    )
    filecmp.cmp("data/scan.html", tmp_path / "scan.html")
