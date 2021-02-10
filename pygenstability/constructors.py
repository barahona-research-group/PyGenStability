"""Quality matrix and null model constructor functions."""
import logging
import sys

import numpy as np
import scipy.sparse as sp

L = logging.getLogger(__name__)
_USE_CACHE = True
THRESHOLD = 1e-12
DTYPE = "float128"


def load_constructor(constructor):
    """Load a constructor from its name, or as a custom Constructor class."""
    if isinstance(constructor, str):
        try:
            return getattr(sys.modules[__name__], "constructor_%s" % constructor)
        except AttributeError as exc:
            raise Exception("Could not load constructor %s" % constructor) from exc
    if not isinstance(constructor, Constructor):
        raise Exception("Only Constructor class object can be used.")
    return constructor


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


def _check_total_degree(degrees):
    """Ensures the sum(degree) > 0."""
    if degrees.sum() < 1e-10:
        raise Exception("The total degree = 0, we cannot proceed further")


def get_spectral_gap(laplacian):
    """Compute spectral gap."""
    spectral_gap = abs(sp.linalg.eigs(laplacian, which="SM", k=2)[0][1])
    L.info("Spectral gap = 10^{:.1f}".format(np.log10(spectral_gap)))
    return spectral_gap


class Constructor:
    """Parent constructor class."""

    def __init__(self, graph, with_spectral_gap=False, kwargs={}):
        """Initialise constructor."""
        self.graph = graph
        self.with_spectral_gap = with_spectral_gap
        self.spectral_gap = None

        # these two variable can be used in prepare method
        self.partial_quality_matrix = None
        self.partial_null_model = None

        self.prepare(**kwargs)

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""

    def get_data(self, time):
        """Return quality and null model at given time."""


class constructor_linearized(Constructor):
    """Constructor for continuous linearized Markov Stability."""

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        degrees = np.array(self.graph.sum(1)).flatten()
        _check_total_degree(degrees)

        pi = degrees / degrees.sum()
        self.partial_null_model = np.array([pi, pi])

        if self.with_spectral_gap:
            laplacian = sp.csgraph.laplacian(self.graph, normed=False)
            self.spectral_gap = get_spectral_gap(laplacian)
        self.partial_quality_matrix = (self.graph / degrees.sum()).astype(DTYPE)

    def get_data(self, time):
        """Return quality and null model at given time."""
        if self.with_spectral_gap:
            time /= self.spectral_gap
        return time * self.partial_quality_matrix, self.partial_null_model, 1 - time


class constructor_continuous_combinatorial(Constructor):
    """Constructor for continuous combinatorial Markov Stability."""

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        laplacian, degrees = sp.csgraph.laplacian(self.graph, return_diag=True, normed=False)
        _check_total_degree(degrees)
        laplacian /= degrees.mean()
        pi = np.ones(self.graph.shape[0]) / self.graph.shape[0]
        self.partial_null_model = np.array([pi, pi], dtype=DTYPE)
        if self.with_spectral_gap:
            self.spectral_gap = get_spectral_gap(laplacian)
        self.partial_quality_matrix = laplacian

    def get_data(self, time):
        """Return quality and null model at given time."""
        if self.with_spectral_gap:
            time /= self.spectral_gap
        exp = apply_expm(-time * self.partial_quality_matrix)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return quality_matrix, self.partial_null_model


class constructor_continuous_normalized(Constructor):
    """Constructor for continuous normalized Markov Stability."""

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        laplacian, degrees = sp.csgraph.laplacian(self.graph, return_diag=True, normed=False)
        _check_total_degree(degrees)
        normed_laplacian = sp.diags(1.0 / degrees).dot(laplacian)

        pi = degrees / degrees.sum()
        self.partial_null_model = np.array([pi, pi], dtype=DTYPE)

        if self.with_spectral_gap:
            self.spectral_gap = get_spectral_gap(normed_laplacian)
        self.partial_quality_matrix = normed_laplacian

    def get_data(self, time):
        """Return quality and null model at given time."""
        if self.with_spectral_gap:
            time /= self.spectral_gap
        exp = apply_expm(-time * self.partial_quality_matrix)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return quality_matrix, self.partial_null_model


class constructor_signed_modularity(Constructor):
    """Constructor of signed modularity.

    Based on (Gomes, Jensen, Arenas, PRE 2009).
    The time only multiplies the quality matrix (this many not mean anything, use with care!).
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        adj_pos = self.graph.copy()
        adj_pos[self.graph < 0] = 0.0
        adj_neg = -self.graph.copy()
        adj_neg[self.graph > 0] = 0.0

        deg_plus = adj_pos.sum(1).flatten()
        deg_neg = adj_neg.sum(1).flatten()

        deg_norm = deg_plus.sum() + deg_neg.sum()
        self.partial_null_model = np.array(
            [
                deg_plus / deg_norm,
                deg_plus / deg_plus.sum(),
                -deg_neg / deg_neg.sum(),
                deg_neg / deg_norm,
            ]
        )
        self.partial_quality_matrix = self.graph / deg_norm

    def get_data(self, time):
        """Return quality and null model at given time."""
        return time * self.partial_quality_matrix, self.partial_null_model


def constructor_directed(Constructor):
    """Constructor for directed Markov stability."""

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        alpha = kwargs["alpha"]
        n_nodes = self.graph.shape[0]
        ones = np.ones((n_nodes, n_nodes)) / n_nodes

        out_degrees = self.graph.toarray().sum(axis=1).flatten()
        dinv = np.divide(1, out_degrees, where=out_degrees != 0)

        self.partial_quality_matrix = sp.csr_matrix(
            alpha * np.diag(dinv).dot(self.graph.toarray())
            + ((1 - alpha) * np.diag(np.ones(n_nodes)) + np.diag(alpha * (dinv == 0.0))).dot(ones)
            - np.eye(n_nodes)
        )

        pi = abs(sp.linalg.eigs(self.partial_quality_matrix.transpose(), which="SM", k=1)[1][:, 0])
        pi /= pi.sum()
        self.partial_null_model = np.array([pi, pi])

    def get_data(self, time):
        """Return quality and null model at given time."""
        exp = apply_expm(time * self.partial_quality_matrix)
        quality_matrix = sp.diags(self.partial_null_model).dot(exp)
        return quality_matrix, self.partial_null_model
