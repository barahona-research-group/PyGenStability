import networkx as nx
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pygenstability as pgs
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)


def get_comp_time(sizes, graph_type="SBM", constructor="linearized", n_tries=5):
    """Estimate computational times for simple benchmarking."""
    df = pd.DataFrame()
    node_sizes = []
    edge_sizes = []
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
                # roughly played with number of edges to get something sort of consistent with SBM
                G = nx.erdos_renyi_graph(size * 60, size * 1000.0 / math.comb(size * 60, 2))
            _node_sizes.append(len(G))
            _edge_sizes.append(len(G.edges))
            A = nx.to_scipy_sparse_array(G)

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
                n_louvain=50,
                constructor=constructor,
                n_workers=1,
            )
            __df = pd.read_csv("timing.csv", header=None)
            __df.columns = ["function", "time"]
            __df = __df.groupby("function").sum()
            __df.loc["without_run", "time"] = __df.loc[__df.index != "run", "time"].sum()
            _df = pd.concat([_df, __df])
        df[size] = _df["time"].groupby("function").mean()

        node_sizes.append(np.mean(_node_sizes))
        edge_sizes.append(np.mean(_edge_sizes))
    df = df.T

    plt.figure(figsize=(6, 4))
    data_low = np.zeros(len(sizes))
    data_high = np.zeros(len(sizes))
    for col in df.columns:
        if col != "run" and col != "without_run":
            data_low = data_high.copy()
            data_high += df[col].to_numpy()
            if col.startswith("_"):
                col = col[1:]
            plt.fill_between(node_sizes, data_low, data_high, label=col, alpha=0.5)
    plt.plot(node_sizes, df["run"], "+-r", label="total")
    plt.axis([node_sizes[0], node_sizes[-1], 0, 1.1 * max(df["run"])])
    plt.legend()
    plt.xlabel("Graph size [# nodes]")
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.plot(edge_sizes, -np.ones(len(edge_sizes)))
    ax2.set_xlabel("Graph size [# edges]")
    df.to_csv(f"comp_time_{constructor}_{graph_type}.csv")
    plt.ylabel("Computational time [s]")
    plt.savefig(f"comp_time_{constructor}_{graph_type}.pdf")


if __name__ == "__main__":
    sizes = list(range(2, 11))

    get_comp_time(sizes, graph_type="SBM", constructor="linearized")
    get_comp_time(sizes, graph_type="ER", constructor="linearized")
    get_comp_time(sizes, graph_type="SBM", constructor="continuous_combinatorial")
    get_comp_time(sizes, graph_type="ER", constructor="continuous_combinatorial")
