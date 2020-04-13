"""quality matrix and null model constructor functions"""
import sys
from functools import lru_cache, partial

import numpy as np
import scipy.sparse as sp

THRESHOLD = 1e-6  # threshold quality matrix
USE_CACHE = True  # cache quality matrices for postprocessing


def load_constructor(graph, params, constructor_custom=None):
    """Load constructor."""
    if constructor_custom is None and "constructor" in params:
        try:
            constructor = getattr(
                sys.modules[__name__], "constructor_%s" % params["constructor"]
            )
        except:
            raise Exception("Could not load constructor %s" % params["constructor"])

    elif constructor_custom is None and "constructor" not in params:
        raise Exception(
            "Please provide either a constructor function or one in params."
        )

    elif callable(constructor_custom):
        constructor = constructor_custom
    else:
        raise Exception("Please pass a valid function as constructor.")

    if not USE_CACHE:
        return partial(constructor, graph)

    @lru_cache()
    def cached_constructor(time):
        return constructor(graph, time)

    return cached_constructor


def _threshold_matrix(matrix):
    mask = np.abs(matrix.data) < THRESHOLD * np.max(matrix)
    matrix.data[mask] = 0
    matrix.eliminate_zeros()


def constructor_continuous_linearized(graph, time):
    """constructor for continuous linearized"""
    print(
        "WARNING: Not fully working, need a shift of quality, could we include it here?"
    )

    degrees = graph.sum(1)
    if degrees.sum() < 1e-10:
        raise Exception("The total degree = 0, we cannot proceed further")
    pi = degrees / degrees.sum()

    null_model = np.array([pi, pi])
    quality_matrix = time * graph / degrees.sum()

    return quality_matrix, null_model


def constructor_continuous_combinatorial(graph, time):
    """constructor for continuous combinatorial"""
    laplacian = sp.csgraph.laplacian(graph).tocsc()

    exp = sp.linalg.expm(-time * laplacian)
    _threshold_matrix(exp)

    pi = np.ones(graph.shape[0]) / graph.shape[0]

    quality_matrix = sp.diags(pi).dot(exp)
    null_model = np.array([pi, pi])

    return quality_matrix, null_model


def constructor_continuous_normalized(graph, time):
    """constructor for continuous normalized"""
    laplacian, degrees = sp.csgraph.laplacian(graph, return_diag=True)
    normed_laplacian = sp.diags(1.0 / degrees).dot(laplacian).tocsc()

    exp = sp.linalg.expm(-time * normed_laplacian)
    _threshold_matrix(exp)

    pi = degrees / degrees.sum()

    quality_matrix = sp.diags(pi).dot(exp)
    null_model = np.array([pi, pi])

    return quality_matrix, null_model


def constructor_signed_modularity(graph, time):
    """constructor of signed mofularitye (Gomes, Jensen, Arenas, PRE 2009)
    the time only multiplies the quality matrix (this many not mean anything, use with care!)"""
    if np.min(graph) >= 0:
        return constructor_continuous_linearized(graph, time)

    adj_pos = graph.copy()
    adj_pos[graph < 0] = 0.0
    adj_neg = -graph.copy()
    adj_neg[graph > 0] = 0.0

    deg_plus = adj_pos.sum(1)
    deg_neg = adj_neg.sum(1)

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
