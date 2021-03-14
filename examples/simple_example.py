"""simple example"""
import pickle

from pygenstability import run, plotting
from create_graph import create_sbm


def simple_test():
    """run simple test"""
    with open("sbm_graph.pkl", "rb") as pickle_file:
        graph = pickle.load(pickle_file)

    all_results = run(graph)

    plotting.plot_scan(all_results, use_plotly=False)
    plotting.plot_scan(all_results, use_plotly=True, live=False)

    with open("sbm_graph.gpickle", "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    plotting.plot_communities(graph, all_results)


if __name__ == "__main__":
    create_sbm()
    simple_test()
