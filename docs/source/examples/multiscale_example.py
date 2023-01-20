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