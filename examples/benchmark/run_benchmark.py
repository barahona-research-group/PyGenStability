import networkx as nx
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pygenstability as pgs
import scipy.sparse as sp
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)


def get_comp_time(sizes, graph_type="SBM", constructor="linearized", method="louvain", n_tries=1):
    """Estimate computational times for simple benchmarking."""
    df = pd.DataFrame()
    filename = f"comp_time_{constructor}_{graph_type}_{method}.csv"
    if not Path(filename).exists():
        for size in sizes:
            print(f"Computing size {size}")
            _df = pd.DataFrame()

            _node_sizes = []
            _edge_sizes = []
            for _ in range(n_tries):
                if graph_type == "SBM":
                    base_sizes = np.array([20, 20, 20])
                    probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
                    G = nx.stochastic_block_model(size * base_sizes, probs, seed=0)
                if graph_type == "ER":
                    # played with number of edges to get something sort of consistent with SBM
                    G = nx.erdos_renyi_graph(size * 60, size * 1000.0 / math.comb(size * 60, 2))
                _node_sizes.append(len(G))
                _edge_sizes.append(len(G.edges))
                A = nx.to_numpy_array(G)
                A = sp.csgraph.csgraph_from_dense(A)

                if Path("timing.csv").exists():
                    os.remove("timing.csv")

                pgs.run(
                    A,
                    min_scale=-1.5,
                    max_scale=0.5,
                    n_scale=10,
                    with_NVI=True,
                    with_postprocessing=True,
                    with_ttprime=True,
                    with_optimal_scales=False,
                    method=method,
                    n_tries=50,
                    constructor=constructor,
                    n_workers=1,
                )
                __df = pd.read_csv("timing.csv", header=None)
                __df.columns = ["function", "time"]
                __df = __df.groupby("function").sum()
                __df.loc["without_run", "time"] = __df.loc[__df.index != "run", "time"].sum()
                _df = pd.concat([_df, __df])
            df[size] = _df["time"].groupby("function").mean()

            df.loc["node_size", size] = np.mean(_node_sizes)
            df.loc["edge_size", size] = np.mean(_edge_sizes)
        df = df.T
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename, index_col=0)
    print(df)
    plt.figure(figsize=(5, 4))
    data_low = np.zeros(len(sizes))
    data_high = np.zeros(len(sizes))
    for col in df.columns:
        if col != "run" and col != "without_run" and col != "node_size" and col != "edge_size":
            data_low = data_high.copy()
            data_high += df[col].to_numpy()
            if col.startswith("_"):
                col = col[1:]
            plt.fill_between(df["node_size"], data_low, data_high, label=col, alpha=0.5)
    plt.plot(df["node_size"], df["run"], "+-r", label="total")
    plt.yscale("log")
    plt.xscale("log")
    # plt.axis([df["node_size"].to_list()[0], df["node_size"].to_list()[-1], 0, 1.1 * max(df["run"])])
    x = np.linspace(200, 1000, 10)
    y = (0.015 * x) ** 2
    print(y)
    plt.plot(x, y, c="k", label="O(2)")
    y = 0.1 * x
    plt.plot(x, y, c="k", ls="--", label="O(1)")
    plt.axis([df["node_size"].to_list()[0], df["node_size"].to_list()[-1], 0, 3e2])
    plt.legend()
    plt.xlabel("Graph size [# nodes]")
    plt.ylabel("Computational time [s]")
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.plot(df["edge_size"], -np.ones(len(df)))
    ax2.set_xlabel("Graph size [# edges]")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(f"comp_time_{constructor}_{graph_type}_{method}.pdf")


if __name__ == "__main__":
    sizes = list(range(2, 20))

    get_comp_time(sizes, graph_type="SBM", constructor="linearized", method="louvain")
    get_comp_time(sizes, graph_type="SBM", constructor="continuous_combinatorial", method="louvain")
    """
    get_comp_time(sizes, graph_type="ER", constructor="linearized", method="louvain")
    get_comp_time(sizes, graph_type="ER", constructor="continuous_combinatorial", method="louvain")
    get_comp_time(sizes, graph_type="SBM", constructor="linearized", method="leiden")
    get_comp_time(sizes, graph_type="SBM", constructor="continuous_combinatorial", method="leiden")
    get_comp_time(sizes, graph_type="ER", constructor="linearized", method="leiden")
    get_comp_time(sizes, graph_type="ER", constructor="continuous_combinatorial", method="leiden")
    """
