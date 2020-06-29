"""create graph for simple example"""
import matplotlib.pyplot as plt
import pickle
import networkx as nx


def create_sbm():
    """simple test"""
    # set a simple SBM model
    sizes = [15, 35, 25]
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

    with open("sbm_graph.pkl", "wb") as pickle_file:
        pickle.dump(nx.adjacency_matrix(graph, weight="weight"), pickle_file)

    nx.write_gpickle(graph, "sbm_graph.gpickle")


if __name__ == "__main__":
    create_sbm()
