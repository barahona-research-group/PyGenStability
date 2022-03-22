import pygenstability as pgs
from pygenstability import plotting
from pygenstability.contrib import optimal_scales
import pickle

from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from scipy.linalg import block_diag
import logging

logging.basicConfig(level=logging.INFO)

def block(n, th):
    A = np.random.uniform(0, 1, (n, n))
    A[A < th] = 0.
    A[A > th] = 1.
    A = (A + A.T) / 2
    return A


if __name__ == "__main__":

    n0 = 270
    th0 = 0.995

    n1 = 10
    th1 = 0.2

    n2 = 30
    th2 = 0.8

    n3 = 90
    th3 = 0.95

    A = block(n0, th0)
    A += block_diag(*[block(n1, th1) for i in range(int(n0 / n1))])
    A += block_diag(*[block(n2, th2) for i in range(int(n0 / n2))])
    A += block_diag(*[block(n3, th3) for i in range(int(n0 / n3))])

    # binarized
    A[A > 0] = 1

    # remove self-loops
    A -= np.diag(np.diag(A))

    plt.figure()
    plt.imshow(A)
    plt.savefig("adajcency_matrix.pdf")

    # converting to csgraph
    G = sp.csgraph.csgraph_from_dense(A)

    # run markov stability
    results = pgs.run(
        G,
        min_time=-1.5,
        max_time=0.5,
        n_time=100,
        n_louvain=500,
        constructor="continuous_combinatorial",
        n_workers=min(cpu_count(), 10),
    )
    results = optimal_scales.identify_optimal_scales(results)
    plotting.plot_scan(results)

    pickle.dump(results, open("example_scan_results.pkl", "wb"))
