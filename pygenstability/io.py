"""i/o functions"""
import pickle
from pathlib import Path
import yaml

import networkx as nx


def load_graph(graph_path):
    """load various types of graphs files"""
    if Path(graph_path).suffix == ".gml":
        return nx.read_graphml(graph_path)
    if Path(graph_path).suffix == ".gpickle":
        return nx.read_gpickle(graph_path)
    raise Exception("Could not read graph {}".format(graph_path))


def load_params(params_path):
    """load params fro yaml file"""
    with open(params_path, "r") as params_file:
        try:
            return yaml.safe_load(params_file)
        except yaml.YAMLError:
            raise Exception("Could not read yaml params file {}".format(params_path))


def save(all_results, filename="all_results.pkl"):
    """save results in a pickle"""
    pickle.dump(all_results, open(filename, "wb"))


def load(filename="all_results.pkl"):
    """load results from a pickle"""
    return pickle.load(open(filename, "rb"))
