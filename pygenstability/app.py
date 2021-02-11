"""Command line interface."""
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy import sparse as sp

from pygenstability import load_results
from pygenstability import run as _run
from pygenstability.plotting import plot_communities as _plot_communities
from pygenstability.plotting import plot_scan as _plot_scan


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
    "--min-time",
    default=-2.0,
    show_default=True,
    help="Minimum Markov time.",
)
@click.option(
    "--max-time",
    default=0.5,
    show_default=True,
    help="Maximum Markov time.",
)
@click.option("--n-time", default=20, show_default=True, help="Number of time steps.")
@click.option(
    "--log-time",
    default=True,
    show_default=True,
    help="Use linear or log scales for times.",
)
@click.option(
    "--n-louvain",
    default=100,
    show_default=True,
    help="Number of Louvain evaluations.",
)
@click.option(
    "--VI/--no-VI",
    default=True,
    show_default=True,
    help="Compute the variation of information between Louvain runs.",
)
@click.option(
    "--n-louvain-VI",
    default=20,
    show_default=True,
    help="Number of randomly chosen Louvain run to estimate the VI.",
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
    help="Normalize time by spectral gap.",
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
def run(
    graph_file,
    constructor,
    min_time,
    max_time,
    n_time,
    log_time,
    n_louvain,
    vi,
    n_louvain_vi,
    postprocessing,
    ttprime,
    spectral_gap,
    result_file,
    n_workers,
    tqdm_disable,
):
    """Run pygenstability.

    graph_file: path to either a .pkl with adjacency matrix in sparse format,
    or a text file with three columns encoding node indices and edge weight.
    The columns need a header, or the first line will be dropped.
    Notice that doubled edges with opposite orientations are needed for symetric graph.

    See https://barahona-research-group.github.io/PyGenStability/ for more information.
    """
    try:
        # load pickle file
        if Path(graph_file).suffix == ".pkl":
            with open(graph_file, "rb") as pickle_file:
                graph = pickle.load(pickle_file)
        else:
            # load text file with edge list
            edges = pd.read_csv(graph_file)
            n_nodes = len(np.unique(edges[edges.columns[:2]].to_numpy().flatten()))
            graph = sp.csr_matrix(
                (edges[edges.columns[2]], tuple(edges[edges.columns[:2]].to_numpy().T)),
                shape=(n_nodes, n_nodes),
            )
    except Exception as exc:
        raise Exception("Could not load the graph file.") from exc

    _run(
        graph,
        constructor=constructor,
        min_time=min_time,
        max_time=max_time,
        n_time=n_time,
        log_time=log_time,
        n_louvain=n_louvain,
        with_VI=vi,
        n_louvain_VI=n_louvain_vi,
        with_postprocessing=postprocessing,
        with_ttprime=ttprime,
        with_spectral_gap=spectral_gap,
        result_file=result_file,
        n_workers=n_workers,
        tqdm_disable=tqdm_disable,
    )


@cli.command("plot_scan")
@click.argument("results_file", type=click.Path(exists=True))
def plot_scan(results_file):
    """Plot results in scan plot."""
    _plot_scan(load_results(results_file))


@cli.command("plot_communities")
@click.argument("graph_file", type=click.Path(exists=True))
@click.argument("results_file", type=click.Path(exists=True))
def plot_communities(results_file, graph_file):
    """Plot communities on networkx graph.

    Argument graph_file has to be a .gpickle compatible with network.
    """
    with open(graph_file, "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    _plot_communities(graph, load_results(results_file))
