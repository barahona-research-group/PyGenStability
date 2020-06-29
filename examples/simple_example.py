"""simple example"""
import pickle

from pygenstability import run, plotting
from create_graph import create_sbm
import scipy.sparse as sp

def simple_test():
    """run simple test"""
    with open("sbm_graph.pkl", "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    graph = sp.triu(graph)
    all_results = run(graph, constructor='directed_normalized' )

    plotting.plot_scan(all_results, use_plotly=True)

    with open("sbm_graph.gpickle", "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    plotting.plot_communities(graph, all_results)

    plotting.plot_sankey(all_results)


if __name__ == "__main__":
    create_sbm()
    simple_test()
