import pygenstability as pgs
from pygenstability import plotting
from pygenstability.contrib import optimal_scales
from pygenstability.pygenstability import _evaluate_NVI
import pickle

from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from pathlib import Path
from scipy.linalg import block_diag
import logging

logging.basicConfig(level=logging.INFO)


def block(n, th, seed):
    rng = np.random.RandomState(seed)
    A = rng.uniform(0, 1, (n, n))
    A[A < th] = 0.0
    A[A > th] = 1.0
    A = (A + A.T) / 2
    return A


if __name__ == "__main__":

    # define root for storing figures
    root = str(Path.cwd())+'/examples/multiscale_example/'
    
    # define size and strength of multiscale structure
    n0 = 270
    th0 = 0.995

    n1 = 10
    th1 = 0.2

    n2 = 30
    th2 = 0.8

    n3 = 90
    th3 = 0.95

    # define list of seeds
    seeds = list(range(1 + int(n0 / n1) + int(n0 / n2) + int(n0 / n3)))

    # construct adjacency matrix
    A = block(n0, th0, seeds.pop())
    A += block_diag(*[block(n1, th1, seeds.pop()) for i in range(int(n0 / n1))])
    A += block_diag(*[block(n2, th2, seeds.pop()) for i in range(int(n0 / n2))])
    A += block_diag(*[block(n3, th3, seeds.pop()) for i in range(int(n0 / n3))])

    # binarized
    A[A > 0] = 1

    # remove self-loops
    A -= np.diag(np.diag(A))

    # binarized
    A[A > 0] = 1

    # remove self-loops
    A -= np.diag(np.diag(A))

    # plot matrix
    plt.figure()
    plt.imshow(A)
    plt.savefig(root + "adajcency_matrix.pdf", bbox_inches='tight')

    # Multiscale structure
    coarse_scale_id = np.zeros(n0)
    middle_scale_id = np.zeros(n0)
    fine_scale_id = np.zeros(n0)

    for i in range(3):
        coarse_scale_id[: i + (i * n0 // 3)] += 1

    for i in range(9):
        middle_scale_id[: i + (i * n0 // 9)] += 1

    for i in range(27):
        fine_scale_id[: i + (i * n0 // 27)] += 1

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

    plt.savefig(root + 'multiscale_structure.pdf', bbox_inches='tight')

    # converting to csgraph
    G = sp.csgraph.csgraph_from_dense(A)

    # run markov stability and identify optimal scales
    results = pgs.run(
        G,
        min_time=-1.5,
        max_time=0.5,
        n_time=100,
        n_louvain=500,
        constructor="continuous_combinatorial",
        n_workers=min(cpu_count(), 10),
    )

    # plots results
    plotting.plot_scan(results)
    plt.savefig(root + 'MS_scan.pdf', bbox_inches='tight')

    pickle.dump(results, open(root + "example_scan_results.pkl", "wb"))

    # with open(root + "example_scan_results.pkl", 'rb') as handle:
    #     results = pickle.load(handle)

    # get log times for x-axis
    min_time = results['run_params']['min_time']
    max_time = results['run_params']['max_time']
    n_time = results['run_params']['n_time']
    log_times = np.linspace(min_time,max_time,n_time)

    # compare MS partitions to ground truth with NVI
    NVI_scores_fine = np.array([_evaluate_NVI([0,i], [fine_scale_id]+results['community_id']) for i in range(1,n_time+1)])
    NVI_scores_middle = np.array([_evaluate_NVI([0,i], [middle_scale_id]+results['community_id']) for i in range(1,n_time+1)])
    NVI_scores_coarse = np.array([_evaluate_NVI([0,i], [coarse_scale_id]+results['community_id']) for i in range(1,n_time+1)])
    
    # plot NVI scores
    fig, ax = plt.subplots(1,figsize =(15,3.5))
    ax.plot(log_times,NVI_scores_fine, label = 'Fine')
    ax.plot(log_times,NVI_scores_middle, label = 'Middle')
    ax.plot(log_times,NVI_scores_coarse, label = 'Coarse')

    # plot minima of NVI scores
    ax.scatter(log_times[np.argmin(NVI_scores_fine)],NVI_scores_fine.min(),marker='.',s=300)
    ax.scatter(log_times[np.argmin(NVI_scores_middle)],NVI_scores_middle.min(),marker='.',s=300)
    ax.scatter(log_times[np.argmin(NVI_scores_coarse)],NVI_scores_coarse.min(),marker='.',s=300)

    # plot selected partitions
    selected_partitions = results['selected_partitions']

    for i in range(len(selected_partitions)):
        if i == 0:
            ax.axvline(x = log_times[selected_partitions[i]], ls="--", color = 'red', label = 'Selected Markov scales')
        else:
            ax.axvline(x = log_times[selected_partitions[i]], ls="--", color = 'red')

    ax.set(xlabel = r"$log_{10}(t)$",ylabel = 'NVI') #yticks = [0.2,0.4,0.6,0.8] ) 
    ax.legend(loc=3)
    plt.savefig(root + 'NVI_comparison.pdf', bbox_inches='tight')

 