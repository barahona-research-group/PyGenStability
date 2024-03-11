"""Command line interface."""

import pickle
from pathlib import Path

import click
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse as sp

from pygenstability import load_results
from pygenstability import run as _run
from pygenstability.plotting import plot_communities as _plot_communities
from pygenstability.plotting import plot_scan as _plot_scan


def _load_graph(graph_file):
    try:
        # load pickle file
        if Path(graph_file).suffix == ".pkl":
            with open(graph_file, "rb") as pickle_file:  # pragma: no cover
                graph = pickle.load(pickle_file)
        else:
            # load text file with edge list
            edges = pd.read_csv(graph_file)
            n_nodes = len(np.unique(edges[edges.columns[:2]].to_numpy().flatten()))
            # pylint: disable=unsubscriptable-object,no-member
            graph = sp.csr_matrix(
                (edges[edges.columns[2]], tuple(edges[edges.columns[:2]].to_numpy().T)),
                shape=(n_nodes, n_nodes),
            )
    except Exception as exc:  # pragma: no cover
        raise Exception("Could not load the graph file.") from exc
    return graph


@click.group()
def cli():
    """App initialisation."""


@cli.command("run")
@click.argument("graph_file", type=click.Path(exists=True))
@click.option(
    "--constructor",
    default="linearized",
    show_default=True,
    help="Name of the quality constructor.",
)
@click.option(
    "--min-scale",
    default=-2.0,
    show_default=True,
    help="Minimum scale.",
)
@click.option(
    "--max-scale",
    default=0.5,
    show_default=True,
    help="Maximum scale.",
)
@click.option("--n-scale", default=20, show_default=True, help="Number of scale steps.")
@click.option(
    "--log-scale",
    default=True,
    show_default=True,
    help="Use linear or log scales.",
)
@click.option(
    "--n-tries",
    default=100,
    show_default=True,
    help="Number of Louvain evaluations.",
)
@click.option(
    "--NVI/--no-NVI",
    default=True,
    show_default=True,
    help="Compute the normalized variation of information between runs.",
)
@click.option(
    "--n-NVI",
    default=20,
    show_default=True,
    help="Number of randomly chosen runs to estimate the NVI.",
)
@click.option(
    "--postprocessing/--no-postprocessing",
    default=True,
    show_default=True,
    help="Apply the final postprocessing step.",
)
@click.option(
    "--ttprime/--no-ttprime",
    default=True,
    show_default=True,
    help="Compute the ttprime matrix.",
)
@click.option(
    "--spectral-gap/--no-spectral-gap",
    default=True,
    show_default=True,
    help="Normalize scale by spectral gap.",
)
@click.option(
    "--result-file",
    default="results.pkl",
    show_default=True,
    help="Path to the result file.",
)
@click.option(
    "--n-workers",
    default=4,
    show_default=True,
    help="Number of workers for multiprocessing.",
)
@click.option("--tqdm-disable", default=False, show_default=True, help="disable progress bars")
@click.option(
    "--method",
    default="louvain",
    show_default=True,
    help="Method to solve modularity, either Louvain or Leiden",
)
@click.option(
    "--with-optimal-scales/--no-with-optimal-scales",
    default=True,
    show_default=True,
    help="Search for optimal scales",
)
@click.option(
    "--exp-comp-mode",
    default="spectral",
    show_default=True,
    help="Method to compute matrix exponential, can be 'spectral' or 'expm'",
)
def run(
    graph_file,
    constructor,
    min_scale,
    max_scale,
    n_scale,
    log_scale,
    n_tries,
    nvi,
    n_nvi,
    postprocessing,
    ttprime,
    spectral_gap,
    result_file,
    n_workers,
    tqdm_disable,
    method,
    with_optimal_scales,
    exp_comp_mode,
):
    """Run pygenstability.

    graph_file: path to either a .pkl with adjacency matrix in sparse format,
    or a text file with three columns encoding node indices and edge weight.
    The columns need a header, or the first line will be dropped.
    Notice that doubled edges with opposite orientations are needed for symetric graph.

    See https://barahona-research-group.github.io/PyGenStability/ for more information.
    """
    graph = _load_graph(graph_file)
    _run(
        graph,
        constructor=constructor,
        min_scale=min_scale,
        max_scale=max_scale,
        n_scale=n_scale,
        log_scale=log_scale,
        n_tries=n_tries,
        with_NVI=nvi,
        n_NVI=n_nvi,
        with_postprocessing=postprocessing,
        with_ttprime=ttprime,
        with_spectral_gap=spectral_gap,
        result_file=result_file,
        n_workers=n_workers,
        tqdm_disable=tqdm_disable,
        method=method,
        with_optimal_scales=with_optimal_scales,
        exp_comp_mode=exp_comp_mode,
    )


@cli.command("plot_scan")
@click.argument("results_file", type=click.Path(exists=True))
def plot_scan(results_file):
    """Plot results in scan plot."""
    _plot_scan(load_results(results_file))


@cli.command("plot_communities")
@click.argument("graph_file", type=click.Path(exists=True))
@click.argument("results_file", type=click.Path(exists=True))
def plot_communities(graph_file, results_file):
    """Plot communities on networkx graph."""
    graph = _load_graph(graph_file)
    if not isinstance(graph, nx.Graph):
        graph = nx.from_scipy_sparse_array(graph)
    _plot_communities(graph, load_results(results_file))
