"""quality matrix and null model constructor functions"""
import numpy as np
import scipy as sc
import networkx as nx


def constructor_continuous_linearized(graph, time):
    """constructor for continuous linearized"""
    Warning("Not fully working, need a shift of quality, could we include it here?")
    degrees = np.array([graph.degree[i] for i in graph.nodes])
    pi = degrees / degrees.sum()

    adjacency_matrix = nx.adjacency_matrix(graph, weight="weight").toarray()

    null_model = np.array([pi, pi])
    quality_matrix = time * adjacency_matrix / degrees.sum()

    return quality_matrix, null_model


def constructor_continuous_combinatorial(graph, time):
    """constructor for continuous combinatorial"""
    pi = np.ones(len(graph)) / len(graph)

    laplacian = 1.0 * nx.laplacian_matrix(graph).tocsc()
    exp = sc.sparse.linalg.expm(-time * laplacian)

    null_model = np.array([pi, pi])
    quality_matrix = sc.sparse.diags(pi).dot(exp)

    return quality_matrix, null_model


def constructor_continuous_normalized(graph, time):
    """constructor for continuous normalized"""
    degrees = np.array([graph.degree[i] for i in graph.nodes])
    pi = degrees / degrees.sum()

    laplacian = sc.sparse.diags(1.0 / degrees).dot(nx.laplacian_matrix(graph)).tocsc()

    exp = sc.sparse.linalg.expm(-time * laplacian)

    null_model = np.array([pi, pi])
    quality_matrix = sc.sparse.diags(pi).dot(exp)

    return quality_matrix, null_model


def constructor_signed_modularity(graph, time):
    """constructor of signed mofularitye (Arenas 2008)"""
    adjacency_matrix = nx.adjacency_matrix(graph, weight="weight").toarray()

    adj_pos = adjacency_matrix.copy()
    adj_pos[adjacency_matrix < 0] = 0.0
    adj_neg = adjacency_matrix.copy()
    adj_neg[adjacency_matrix > 0] = 0.0

    deg_plus = adj_pos.sum(1)
    deg_neg = adj_neg.sum(1)
    deg_norm = deg_plus.sum() + deg_neg.sum()

    if deg_neg.sum() < 1e-10:
        deg_neg_norm = np.zeros(len(graph))
        deg_neg = np.zeros(len(graph))
    else:
        deg_neg_norm = deg_neg / deg_neg.sum()

    if deg_plus.sum() < 1e-10:
        deg_plus_norm = np.zeros(len(graph))
        deg_plus = np.zeros(len(graph))
    else:
        deg_plus_norm = deg_plus / deg_plus.sum()

    null_model = np.array([deg_plus, deg_plus_norm, deg_neg, -deg_neg_norm]) / deg_norm
    quality_matrix = time * adjacency_matrix / deg_norm

    return quality_matrix, null_model