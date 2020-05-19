"""click app"""
import pickle

import click

from .io import load_results

# pylint: disable=import-outside-toplevel


@click.group()
def cli():
    """App initialisation"""


@cli.command("run")
@click.argument("graph_file", type=click.Path(exists=True))
@click.option(
    "--constructor",
    default="linearized",
    show_default=True,
    help="name of the quality constructor",
)
@click.option(
    "--min-time", default=-2.0, show_default=True, help="minimum Markov time",
)
@click.option(
    "--max-time", default=0.5, show_default=True, help="maximum Markov time",
)
@click.option("--n-time", default=20, show_default=True, help="number of time steps")
@click.option(
    "--log-time",
    default=True,
    show_default=True,
    help="use linear or log scales for times",
)
@click.option(
    "--n-louvain", default=100, show_default=True, help="number of Louvain evaluations",
)
@click.option(
    "--MI/--no-MI",
    default=True,
    show_default=True,
    help="compute the mutual information between Louvain runs",
)
@click.option(
    "--n-louvain-MI",
    default=20,
    show_default=True,
    help="number of randomly chosen Louvain run to estimate MI",
)
@click.option(
    "--postprocessing/--no-postprocessing",
    default=True,
    show_default=True,
    help="apply the final postprocessing step",
)
@click.option(
    "--ttprime/--no-ttprime",
    default=True,
    show_default=True,
    help="compute the ttprime matrix",
)
@click.option(
    "--result-file",
    default="results.pkl",
    show_default=True,
    help="path to the result file",
)
@click.option(
    "--n-workers",
    default=4,
    show_default=True,
    help="number of workers for multiprocessing",
)
@click.option(
    "--tqdm-disable", default=False, show_default=True, help="disable progress bars"
)
def run(
    graph_file,
    constructor,
    min_time,
    max_time,
    n_time,
    log_time,
    n_louvain,
    mi,
    n_louvain_mi,
    postprocessing,
    ttprime,
    result_file,
    n_workers,
    tqdm_disable,
):
    """ Run pygenstability."""
    from .pygenstability import run

    with open(graph_file, "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    run(
        graph,
        constructor=constructor,
        min_time=min_time,
        max_time=max_time,
        n_time=n_time,
        log_time=log_time,
        n_louvain=n_louvain,
        with_MI=mi,
        n_louvain_MI=n_louvain_mi,
        with_postprocessing=postprocessing,
        with_ttprime=ttprime,
        result_file=result_file,
        n_workers=n_workers,
        tqdm_disable=tqdm_disable,
    )


@cli.command("plot_scan")
@click.argument("results_file", type=click.Path(exists=True))
def plot_scan(results_file):
    """Plot results in scan plot."""
    from .plotting import plot_scan

    plot_scan(load_results(results_file))


@cli.command("plot_communities")
@click.argument("graph_file", type=click.Path(exists=True))
@click.argument("results_file", type=click.Path(exists=True))
def plot_communities(results_file, graph_file):
    """Plot communities on networkx graph."""
    from .plotting import plot_communities

    with open(graph_file, "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    plot_communities(graph, load_results(results_file))
