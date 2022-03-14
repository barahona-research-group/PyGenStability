import leidenalg
import igraph as ig
from pygenstability import run, plotting
import numpy as np
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
    #plt.figure(figsize=(5, 4))
    #nx.draw(graph, pos=pos, node_color=community_labels)
    #plt.title("Ground truth communities")
    #plt.savefig("ground_truth.png", bbox_inches="tight")

    # save adjacency with pickle
    adjacency =  nx.adjacency_matrix(graph, weight="weight")
    #with open("sbm_graph.pkl", "wb") as pickle_file:
    #    pickle.dump(adjacency, pickle_file)

    # save .gpickle for community plotting
    #nx.write_gpickle(graph, "sbm_graph.gpickle")

    # save with text file as alternative format
    #edges = pd.DataFrame()
    #edges["i"] = [e[0] for e in graph.edges] + [e[1] for e in graph.edges]
    #edges["j"] = [e[1] for e in graph.edges] + [e[0] for e in graph.edges]
    #edges["weight"] = 2 * [graph.edges[e]["weight"] for e in graph.edges]
    #edges.to_csv("edges.csv", index=False)

    return adjacency

if __name__ == "__main__":

    G = ig.Graph.Erdos_Renyi(100, 0.1)
    graph = G.get_adjacency().data
    graph = create_sbm()
    #all_results = run(graph, n_louvain=10, n_time=1, min_time=0, max_time=0)
    #print(all_results)
    all_results = run(graph, n_louvain=10, n_time=1, min_time=0, max_time=0, method='leiden')
    print(all_results)

    #part = leidenalg.find_partition(G, leidenalg.GeneralizedModularityVertexPartition)
    #print(part)
