"""quality matrix and null model constructor functions"""
import sys
import logging
from functools import lru_cache, partial

import numpy as np
import scipy.sparse as sp

L = logging.getLogger(__name__)
_USE_CACHE = True
THRESHOLD = 1e-12
DTYPE = "float128"


def load_constructor(graph, constructor, with_spectral_gap=True, use_cache=_USE_CACHE):
    """Load constructor."""
    if isinstance(constructor, str):
        try:
            constructor = getattr(sys.modules[__name__], "constructor_%s" % constructor)
        except AttributeError:
            raise Exception("Could not load constructor %s" % constructor)

    if not use_cache:
        if hasattr(constructor, "with_spectral_gap"):
            return partial(constructor, graph, with_spectral_gap=with_spectral_gap)
        return partial(constructor, graph)

    @lru_cache()
    def cached_constructor(time):
        if hasattr(constructor, "with_spectral_gap"):
            return partial(constructor, graph, with_spectral_gap=with_spectral_gap)
        return constructor(graph, time)

    return cached_constructor


def threshold_matrix(matrix, threshold=THRESHOLD):
    """Threshold a matrix to remove small numbers for Louvain speed up."""
    matrix.data[np.abs(matrix.data) < threshold * np.max(matrix)] = 0
    matrix.eliminate_zeros()


def apply_expm(matrix):
    """Apply matrix exponential.

    TODO: implement other variants
    """
    exp = sp.csr_matrix(sp.linalg.expm(matrix.toarray().astype(DTYPE)))
    threshold_matrix(exp)
    return exp


def _check_tot_degree(degrees):
    """Ensures the sum(degree) > 0."""
    if degrees.sum() < 1e-10:
        raise Exception("The total degree = 0, we cannot proceed further")


def get_spectral_gap(laplacian):
    """Compute spectral gap."""
    spectral_gap = abs(sp.linalg.eigs(laplacian, which="SM", k=2)[0][1])
    L.info("Spectral gap = 10^{:.1f}".format(np.log10(spectral_gap)))
    return spectral_gap


def constructor_linearized(graph, time):
    """Constructor for continuous linearized Markov Stability."""
    degrees = _check_tot_degree(graph.sum(1).flatten())

    pi = degrees / degrees.sum()
    null_model = np.array([pi, pi])

    quality_matrix = time * (graph / degrees.sum()).astype(DTYPE)

    return quality_matrix, null_model, 1 - time


def constructor_continuous_combinatorial(graph, time, with_spectral_gap=True):
    """Constructor for continuous combinatorial Markov Stability."""
    laplacian, degrees = sp.csgraph.laplacian(graph, return_diag=True, normed=False)
    _check_tot_degree(degrees)
    laplacian /= degrees.mean()
    pi = np.ones(graph.shape[0]) / graph.shape[0]
    null_model = np.array([pi, pi], dtype=DTYPE)

    if with_spectral_gap:
        time /= get_spectral_gap(laplacian)

    exp = apply_expm(-time * laplacian)
    quality_matrix = sp.diags(pi).dot(exp)

    return quality_matrix, null_model


def constructor_continuous_normalized(graph, time, with_spectral_gap=True):
    """Constructor for continuous normalized Markov Stability."""
    laplacian, degrees = sp.csgraph.laplacian(graph, return_diag=True, normed=False)
    _check_tot_degree(degrees)
    normed_laplacian = sp.diags(1.0 / degrees).dot(laplacian)

    pi = degrees / degrees.sum()
    null_model = np.array([pi, pi], dtype=DTYPE)

    if with_spectral_gap:
        time /= get_spectral_gap(normed_laplacian)

    exp = apply_expm(-time * normed_laplacian)
    quality_matrix = sp.diags(pi).dot(exp)

    return quality_matrix, null_model


def constructor_signed_modularity(graph, time):
    """Constructor of signed modularity (Gomes, Jensen, Arenas, PRE 2009)

    The time only multiplies the quality matrix (this many not mean anything, use with care!)"""
    if np.min(graph) >= 0:
        return constructor_linearized(graph, time)

    adj_pos = graph.copy()
    adj_pos[graph < 0] = 0.0
    adj_neg = -graph.copy()
    adj_neg[graph > 0] = 0.0

    deg_plus = adj_pos.sum(1).flatten()
    deg_neg = adj_neg.sum(1).flatten()

    deg_norm = deg_plus.sum() + deg_neg.sum()
    null_model = np.array(
        [
            deg_plus / deg_norm,
            deg_plus / deg_plus.sum(),
            -deg_neg / deg_neg.sum(),
            deg_neg / deg_norm,
        ]
    )
    quality_matrix = time * graph / deg_norm
    return quality_matrix, null_model


def constructor_directed(graph, time, alpha=0.85):
    """Constructor for directed Markov stability.

    WIP"""
    dinv = np.asarray(np.divide(1, graph.sum(axis=1), where=graph.sum(axis=1) != 0))
    Dinv = np.diag(dinv.reshape(-1))
    ind_d = sp.csr_matrix([dinv == 0][0].reshape(-1) * 1)

    ones = sp.csr_matrix(np.ones(graph.shape[0]))
    M = (
        alpha * Dinv * graph
        + ((1 - alpha) * ones + alpha * ind_d).T * ones / graph.shape[0]
    )
    M = sp.csr_matrix(M)
    Q = sp.csc_matrix(M - sp.eye(M.shape[0]))

    exp = apply_expm(time * Q)
    pi = abs(sp.linalg.eigs(Q.transpose(), which="SM", k=1)[1][:, 0])
    pi /= pi.sum()

    threshold_matrix(exp)
    quality_matrix = sp.diags(pi).dot(exp)
    null_model = np.array([pi, pi])

    return quality_matrix, null_model
