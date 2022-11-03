import networkx as nx

from pygenstability import run, plotting

if __name__ == "__main__":
    G = nx.read_gpickle("2RH5.gpickle")
    adjacency = nx.adjacency_matrix(G)

    all_results = run(adjacency, min_scale=-1.5, max_scale=2, n_scale=30, constructor="linearized")

    plotting.plot_scan(all_results, use_plotly=False)
    plotting.plot_scan(all_results, use_plotly=True)
