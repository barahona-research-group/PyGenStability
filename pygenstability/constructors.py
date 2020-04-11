"""quality matrix and null model constructor functions"""
import sys
import numpy as np
import scipy as sc
import networkx as nx

THRESHOLD = 1e-6


def _load_constructor(constructor_type):
    try:
        return getattr(sys.modules[__name__], "constructor_%s" % constructor_type)
    except:
        raise Exception("Could not load constructor %s" % constructor_type)


def _threshold_matrix(matrix):
    mask = np.abs(matrix.data) < THRESHOLD * np.max(matrix)
    matrix.data[mask] = 0
    matrix.eliminate_zeros()


def constructor_continuous_linearized(graph, time):
    """constructor for continuous linearized"""
    Warning("Not fully working, need a shift of quality, could we include it here?")
    adjacency_matrix = nx.adjacency_matrix(graph, weight="weight").toarray()

    degrees = adjacency_matrix.sum(1)
    if degrees.sum() < 1e-10:
        raise Exception("The total degree = 0, we cannot proceed further")
    pi = degrees / degrees.sum()

    null_model = np.array([pi, pi])
    quality_matrix = time * adjacency_matrix / degrees.sum()

    return quality_matrix, null_model


def constructor_continuous_combinatorial(graph, time):
    """constructor for continuous combinatorial"""
    pi = np.ones(len(graph)) / len(graph)

    laplacian = 1.0 * nx.laplacian_matrix(graph).tocsc()

    exp = sc.sparse.linalg.expm(-time * laplacian)

    _threshold_matrix(exp)

    null_model = np.array([pi, pi])
    quality_matrix = sc.sparse.diags(pi).dot(exp)

    return quality_matrix, null_model


def constructor_continuous_normalized(graph, time):
    """constructor for continuous normalized"""
    degrees = np.array([graph.degree[i] for i in graph.nodes])
    pi = degrees / degrees.sum()

    laplacian = sc.sparse.diags(1.0 / degrees).dot(nx.laplacian_matrix(graph)).tocsc()

    exp = sc.sparse.linalg.expm(-time * laplacian)

    _threshold_matrix(exp)

    null_model = np.array([pi, pi])
    quality_matrix = sc.sparse.diags(pi).dot(exp)

    return quality_matrix, null_model


def constructor_signed_modularity(graph, time):
    """constructor of signed mofularitye (Gomes, Jensen, Arenas, PRE 2009)
    the time only multiplies the quality matrix (this many not mean anything, use with care!)"""
    adjacency_matrix = nx.adjacency_matrix(graph, weight="weight").toarray()
    if np.min(adjacency_matrix) >= 0:
        return constructor_continuous_linearized(graph, time)

    adj_pos = adjacency_matrix.copy()
    adj_pos[adjacency_matrix < 0] = 0.0
    adj_neg = -adjacency_matrix.copy()
    adj_neg[adjacency_matrix > 0] = 0.0

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
    quality_matrix = time * adjacency_matrix / deg_norm

    return quality_matrix, null_model
