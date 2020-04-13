"""click app"""
import pickle

import click

from .io import load_params, load_results
from .plotting import plot_communities, plot_scan


@click.group()
def cli():
    """App initialisation"""


@cli.command("run")
@click.argument("graph_file", type=click.Path(exists=True))
@click.argument("params_file", type=click.Path(exists=True))
def run(graph_file, params_file):
    """ Run pygenstability"""
    from .pygenstability import run

    with open(graph_file, "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    run(graph, load_params(params_file))


@cli.command("plot_scan")
@click.argument("results_file", type=click.Path(exists=True))
def plot(results_file):
    """Plot results in scan plot."""
    plot_scan(load_results(results_file))


@cli.command("plot_communities")
@click.argument("graph_file", type=click.Path(exists=True))
@click.argument("results_file", type=click.Path(exists=True))
def plot(results_file, graph_file):
    """Plot communities on networkx graph."""
    with open(graph_file, "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    plot_communities(graph, load_results(results_file))
