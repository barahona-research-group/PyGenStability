"""click app"""
import click

from .pygenstability import run as _run
from .io import load_graph, load_params, load
from .plotting import plot_scan


@click.group()
def cli():
    """App initialisation"""


@cli.command("run")
@click.argument("graph_path", type=click.Path(exists=True))
@click.argument("params_path", type=click.Path(exists=True))
def run(graph_path, params_path):
    """ Run pygenstability"""
    _run(load_graph(graph_path), load_params(params_path))


@cli.command("plot")
@click.argument("results", type=click.Path(exists=True))
def plot(results):
    """ Run pygenstability"""
    plot_scan(load(results))
