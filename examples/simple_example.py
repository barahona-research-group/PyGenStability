"""simple example"""
import pygenstability as pgs
from pygenstability import plotting
from create_graph import create_sbm
import matplotlib.pyplot as plt

def simple_test():
    """run simple test"""
    params = pgs.load_params("params.yaml")
    graph = pgs.load_graph("sbm_graph.gpickle")

    all_results = pgs.run(graph, params)

    plotting.plot_scan(all_results, use_plotly=False)
    plt.show()
    #plotting.plot_communities(graph, all_results)


if __name__ == "__main__":

    create_sbm()
    simple_test()
