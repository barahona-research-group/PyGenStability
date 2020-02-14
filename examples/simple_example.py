"""simple example"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pygenstability import run
from pygenstability.plotting import plot_scan, plot_communities

# import the constructor of your choice here,
# see available consttructors in module constructors
from pygenstability.constructors import (
    constructor_continuous_combinatorial as constructor,
)


def simple_test():
    """simple test"""
    # set a simple SBM model
    sizes = [15, 55, 25]
    probs = [[0.7, 0.08, 0.10], [0.08, 0.8, 0.02], [0.10, 0.02, 0.80]]

    graph = nx.stochastic_block_model(sizes, probs, seed=0)

    # need to set the weights to 1
    for i, j in graph.edges():
        graph[i][j]["weight"] = 1

    # ground truth
    community_labels = [graph.nodes[i]["block"] for i in graph]

    # spring layout
    pos = nx.spring_layout(graph, weight=None, scale=1)
    for u in graph:
        graph.nodes[u]["pos"] = pos[u]

    # draw the graph with ground truth
    plt.figure(figsize=(5, 4))
    nx.draw(graph, pos=pos, node_color=community_labels)
    plt.title("Ground truth communities")
    plt.savefig("ground_truth.png", bbox_inches="tight")

    times = np.logspace(-3, 0.5, 20)
    params = {}
    params["n_runs"] = 50
    params["n_workers"] = 4
    params["compute_mutual_information"] = True
    params["compute_ttprime"] = True
    params["apply_postprocessing"] = True
    params["save_qualities"] = True
    params["n_partitions"] = 10

    all_results = run(graph, times, constructor, params)

    plot_scan(all_results)

    plot_communities(graph, all_results)


if __name__ == "__main__":
    simple_test()
