"""simple example"""
import pickle
import matplotlib.pyplot as plt

from pygenstability import run, plotting
from pygenstability.contrib.optimal_scales import identify_optimal_scales, plot_optimal_scales
from create_graph import create_sbm


def simple_test():
    """run simple test"""
    with open("sbm_graph.pkl", "rb") as pickle_file:
        graph = pickle.load(pickle_file)

    all_results = run(graph, n_time=50)
    all_results = identify_optimal_scales(
        all_results, window_size=5, VI_cutoff=0.1, criterion_threshold=0.9
    )

    plot_optimal_scales(all_results, use_plotly=True)
    plt.show()

    with open("sbm_graph.gpickle", "rb") as pickle_file:
        graph = pickle.load(pickle_file)
    plotting.plot_communities(graph, all_results)


if __name__ == "__main__":
    create_sbm()
    simple_test()
