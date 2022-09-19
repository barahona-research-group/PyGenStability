"""Quality matrix and null model constructor functions."""
import logging
import sys

import numpy as np
import scipy.sparse as sp

L = logging.getLogger(__name__)
_USE_CACHE = True
THRESHOLD = 1e-8
DTYPE = "float128"


def load_constructor(constructor, graph, **kwargs):
    """Load a constructor from its name, or as a custom Constructor class."""
    if isinstance(constructor, str):
        if graph is None:
            raise Exception(f"No graph was provided with a generic constructor {constructor}")
        try:
            return getattr(sys.modules[__name__], f"constructor_{constructor}")(graph, **kwargs)
        except AttributeError as exc:
            raise Exception(f"Could not load constructor {constructor}") from exc
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
    spectral_gap = max(np.real(sp.linalg.eigs(laplacian, which="SM", k=2)[0]))
    L.info("Spectral gap = 10^%s", np.around(np.log10(spectral_gap), 2))
    return spectral_gap


class Constructor:
    """Parent constructor class.

    This class encodes method specific construction of quality matrix and null models.
    Use the method prepare to load and compute time independent quantities, and the method get_data
    to return quality matrix, null model, and possible global shift (for linearised stability).
    """

    def __init__(self, graph, with_spectral_gap=False, **kwargs):
        """The constructor calls te prepare method upon initialisation.

        Args:
            graph (csgraph): graph for which to run clustering
            with_spectral_gap (bool): set to True to use spectral gap time rescale if available
            kwargs (dict): any other properties to pass to the constructor.
        """
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
        """Return quality and null model at given time as well as global shift (or None)."""


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
        return {
            "quality": time * self.partial_quality_matrix,
            "null_model": self.partial_null_model,
            "shift": float(1 - time),
        }


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
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


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
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


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
        return {
            "quality": time * self.partial_quality_matrix,
            "null_model": self.partial_null_model,
        }


class constructor_directed(Constructor):
    """Constructor for directed Markov stability."""

    def prepare(self, **kwargs):
        """Prepare the constructor with non-time dependent computations."""
        alpha = kwargs.get("alpha", 0.8)
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
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}
