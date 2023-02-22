import pygenstability as pgs
from pygenstability import plotting
from pygenstability.pygenstability import evaluate_NVI

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from scipy.linalg import block_diag
import logging

logging.basicConfig(level=logging.INFO)


def block(n, th, rng):
    A = rng.uniform(0, 1, (n, n))
    A[A < th] = 0.0
    A[A > th] = 1.0
    A = (A + A.T) / 2
    return A


def create_graph():

    # define size and strength of multiscale structure
    n0 = 270
    th0 = 0.995

    n1 = 3
    th1 = 0.95

    n2 = 9
    th2 = 0.8

    n3 = 27
    th3 = 0.2

    # construct adjacency matrix
    rng = np.random.RandomState(42)
    A = block(n0, th0, rng)
    A += block_diag(*[block(int(n0 / n1), th1, rng) for i in range(n1)])
    A += block_diag(*[block(int(n0 / n2), th2, rng) for i in range(n2)])
    A += block_diag(*[block(int(n0 / n3), th3, rng) for i in range(n3)])

    # binarized
    A[A > 0] = 1

    # remove self-loops
    A -= np.diag(np.diag(A))

    # Multiscale structure
    coarse_scale_id = np.zeros(n0)
    middle_scale_id = np.zeros(n0)
    fine_scale_id = np.zeros(n0)

    for i in range(n1):
        coarse_scale_id[(i * n0 // n1) : ((i + 1) * n0 // n1)] = i

    for i in range(n2):
        middle_scale_id[(i * n0 // n2) : ((i + 1) * n0 // n2)] = i

    for i in range(n3):
        fine_scale_id[(i * n0 // n3) : ((i + 1) * n0 // n3)] = i

    return A, coarse_scale_id, middle_scale_id, fine_scale_id


if __name__ == "__main__":

    A, coarse_scale_id, middle_scale_id, fine_scale_id = create_graph()

    # plot matrix
    plt.figure()
    plt.imshow(A, interpolation="nearest")
    plt.savefig("adjacency_matrix.pdf", bbox_inches="tight")

    # Create nx graph
    G = nx.from_numpy_array(A)

    # Compute spring layout
    pos_G = nx.layout.spring_layout(G, seed=1)

    # Plot multiscale graph structure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    nx.draw(
        G,
        ax=axes[0],
        pos=pos_G,
        node_size=20,
        arrows=False,
        width=0.1,
        node_color=fine_scale_id,
        cmap="jet",
    )
    axes[0].set(title=r"Fine scale, n=27")

    nx.draw(
        G,
        ax=axes[1],
        pos=pos_G,
        node_size=20,
        arrows=False,
        width=0.1,
        node_color=middle_scale_id,
        cmap="jet",
    )
    axes[1].set(title=r"Middle scale, n=9")

    nx.draw(
        G,
        ax=axes[2],
        pos=pos_G,
        node_size=20,
        arrows=False,
        width=0.1,
        node_color=coarse_scale_id,
        cmap="jet",
    )
    axes[2].set(title=r"Coarse scale, n=3")

    plt.savefig("multiscale_structure.pdf", bbox_inches="tight")

    # converting to csgraph

    # run markov stability and identify optimal scales
    results = pgs.run(
        sp.csgraph.csgraph_from_dense(A),
        min_scale=-1.25,
        max_scale=0.75,
        n_scale=50,
        n_tries=20,
        constructor="continuous_combinatorial",
        n_workers=4,
        exp_comp_mode='expm',
    )
    print(results)
    # plots results
    plt.figure(figsize=(7, 5))
    axes = plotting.plot_scan(results, figure_name=None)
    axes[3].set_ylim(0, 50)
    axes[3].axhline(3, ls="--", color="k", zorder=-1, lw=0.5)
    axes[3].axhline(9, ls="--", color="k", zorder=-1, lw=0.5)
    axes[3].axhline(27, ls="--", color="k", zorder=-1, lw=0.5)
    plt.savefig("scan_results.pdf")
    plt.close()

    for i in results["selected_partitions"][1:]:
        data = results["community_id"][i]
        plt.figure()
        plt.imshow([data / max(data)], aspect=10.0, cmap="gist_rainbow")
        plt.savefig(f"partitions_{i}.pdf")

    # compare MS partitions to ground truth with NVI
    def _get_NVI(ref_ids):
        return [
            evaluate_NVI([0, i + 1], [ref_ids] + results["community_id"])
            for i in range(len(results["scales"]))
        ]

    NVI_scores_fine = _get_NVI(fine_scale_id)
    NVI_scores_middle = _get_NVI(middle_scale_id)
    NVI_scores_coarse = _get_NVI(coarse_scale_id)
    scales = results["scales"]

    # plot NVI scores
    fig, ax = plt.subplots(1, figsize=(7, 4))
    ax.plot(scales, NVI_scores_fine, label="Fine")
    ax.plot(scales, NVI_scores_middle, label="Middle")
    ax.plot(scales, NVI_scores_coarse, label="Coarse")

    # plot selected partitions
    selected_partitions = results["selected_partitions"]
    ax.axvline(
        x=results["scales"][selected_partitions[0]],
        ls="--",
        color="red",
        label="Selected Markov scales",
    )
    for i in selected_partitions[1:]:
        ax.axvline(x=results["scales"][i], ls="--", color="red")

    ax.set(xlabel=r"$log_{10}(t)$", ylabel="NVI")
    plt.axhline(0, c="k", ls="--")
    ax.legend(loc=3)
    plt.xscale("log")
    plt.savefig("NVI_comparison.pdf", bbox_inches="tight")
