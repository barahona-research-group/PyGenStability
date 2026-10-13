"""How many optimisation tries do Louvain and Leiden need, as a function of graph size?

For planted-partition graphs of increasing size, this runs many independent Louvain and
Leiden tries at a few scales and records, for each method:
- the time per try,
- the expected best stability after k tries (best-of-k, averaged over random subsets),
- the number of tries needed to get within 0.1% of the best stability found by any method,
- the agreement (NMI) of the best partition with the planted one.

Usage: python n_tries_louvain_leiden.py  (writes n_tries.csv, n_tries.md and n_tries.png)
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score

from pygenstability.constructors import load_constructor
from pygenstability.pygenstability import _evaluate_quality
from pygenstability.pygenstability import _optimise
from pygenstability.pygenstability import _to_indices

logging.disable(logging.CRITICAL)

SIZES = [500, 2000, 10000, 50000]
SCALES = [0.5, 1.0]
N_TRIES = {"louvain": 100, "leiden": 30}
KS = [1, 2, 5, 10, 20, 50, 100]
TOLERANCE = 1e-3  # relative gap to the best known stability
N_BLOCKS = 20
OUT = Path(__file__).parent


def planted_graph(n, seed=0):
    """20 equal blocks, average degree ~10 (8 inside, 2 outside), largest component."""
    block = n // N_BLOCKS
    G = nx.random_partition_graph([block] * N_BLOCKS, 8 / block, 2 / n, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len))
    nodes = sorted(G)
    truth = [G.nodes[u]["block"] for u in nodes]
    return sp.csr_matrix(nx.adjacency_matrix(G, nodelist=nodes), dtype=float), truth


def best_of_k(values, k, n_samples=2000, seed=0):
    """Expected maximum of k values drawn without replacement from the observed tries."""
    values = np.asarray(values)
    if k > len(values):
        return np.nan
    rng = np.random.default_rng(seed)
    return np.mean([values[rng.permutation(len(values))[:k]].max() for _ in range(n_samples)])


def run():
    rows = []
    for n in SIZES:
        adjacency, truth = planted_graph(n)
        for scale in SCALES:
            data = load_constructor("linearized", adjacency).get_data(scale)
            null_model = np.asarray(data["null_model"])
            louvain_indices, louvain_values = _to_indices(data["quality"])

            def stability(partition):
                # common yardstick for both methods
                return _evaluate_quality(
                    list(partition), louvain_indices, louvain_values, null_model, 0.0, "louvain"
                )

            for method, n_tries in N_TRIES.items():
                indices, values = _to_indices(data["quality"], directed=method == "leiden")
                for seed in range(n_tries):
                    start = time.perf_counter()
                    _, partition = _optimise(
                        seed, seed, indices, values, null_model, 0.0, method=method
                    )
                    rows.append(
                        {
                            "n_nodes": adjacency.shape[0],
                            "scale": scale,
                            "method": method,
                            "seed": seed,
                            "time": time.perf_counter() - start,
                            "stability": stability(partition),
                            "n_communities": len(set(partition)),
                            "nmi_planted": normalized_mutual_info_score(truth, partition),
                        }
                    )
                print(f"done n={adjacency.shape[0]} scale={scale} {method}", flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    summary = []
    for (n, scale), group in df.groupby(["n_nodes", "scale"]):
        best_known = group.stability.max()
        for method, runs in group.groupby("method"):
            curve = {k: best_of_k(runs.stability, k) for k in KS}
            target = best_known - TOLERANCE * abs(best_known)
            needed = next((k for k in KS if curve[k] >= target), None)
            best_run = runs.loc[runs.stability.idxmax()]
            summary.append(
                {
                    "n_nodes": n,
                    "scale": scale,
                    "method": method,
                    "time_per_try_s": runs.time.mean(),
                    "best_of_1": curve[1],
                    "best_of_10": curve[10],
                    "best_found": runs.stability.max(),
                    "best_known": best_known,
                    "tries_within_0.1%": needed if needed is not None else f">{len(runs)}",
                    "time_to_target_s": (
                        needed * runs.time.mean() if needed is not None else np.nan
                    ),
                    "nmi_best_partition": best_run.nmi_planted,
                    **{f"k{k}": v for k, v in curve.items()},
                }
            )
    return pd.DataFrame(summary)


def plot(summary):
    scales = sorted(summary.scale.unique())
    fig, axes = plt.subplots(1, len(scales), figsize=(5 * len(scales), 4), sharey=True)
    for ax, scale in zip(np.atleast_1d(axes), scales):
        for (n, method), row in summary[summary.scale == scale].groupby(["n_nodes", "method"]):
            row = row.iloc[0]
            gap = [(row.best_known - row[f"k{k}"]) / abs(row.best_known) for k in KS]
            ax.plot(
                KS,
                np.maximum(gap, 1e-6),
                "o-" if method == "louvain" else "s--",
                c=plt.cm.viridis(SIZES.index(min(SIZES, key=lambda s: abs(s - n))) / len(SIZES)),
                label=f"{method}, n={n}",
            )
        ax.axhline(TOLERANCE, c="k", lw=0.5, ls=":")
        ax.set(xscale="log", yscale="log", xlabel="number of tries k", title=f"scale {scale}")
    np.atleast_1d(axes)[0].set_ylabel("relative gap of best-of-k to best known")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "n_tries.png", dpi=150)


if __name__ == "__main__":
    df = run()
    df.to_csv(OUT / "n_tries.csv", index=False)
    summary = summarise(df)
    columns = [
        "n_nodes",
        "scale",
        "method",
        "time_per_try_s",
        "best_of_1",
        "best_of_10",
        "best_known",
        "tries_within_0.1%",
        "time_to_target_s",
        "nmi_best_partition",
    ]
    (OUT / "n_tries.md").write_text(summary[columns].to_markdown(index=False, floatfmt=".4g"))
    print(summary[columns].to_string(index=False))
    plot(summary)
