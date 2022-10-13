"""Test plotting module."""
from pathlib import Path
from diff_pdf_visually import pdfdiff
import filecmp

from pygenstability import plotting
from pygenstability.constructors import load_constructor
from pygenstability import pygenstability as pgs

DATA = Path(__file__).absolute().parent / "data"


def test_plot_scan(graph, tmp_path):
    constructor = load_constructor("continuous_combinatorial", graph)
    results = pgs.run(graph, constructor=constructor)
    plotting.plot_scan(results, figure_name=tmp_path / "scan.pdf")
    pdfdiff(DATA / "scan.pdf", tmp_path / "scan.pdf")

    plotting.plot_scan(
        results, use_plotly=True, live=False, plotly_filename=str(tmp_path / "scan.html")
    )
    filecmp.cmp(DATA / "scan.html", tmp_path / "scan.html")
