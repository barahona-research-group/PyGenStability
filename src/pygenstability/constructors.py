r"""Module to create constructors of quality matrix and null models.

The generalized modularity is given as

.. math::

    Q_{gen}(t,H) = \mathrm{Tr} \left [H^T \left (F(t)-\sum_{k=0}^m v_{2k} v_{2k+1}^T\right)H\right]

where :math:`F(t)` is the quality matrix and :math:`v_k` are null model vectors.
"""
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


def _threshold_matrix(matrix, threshold=THRESHOLD):
    """Threshold a matrix to remove small numbers for speed up."""
    matrix.data[np.abs(matrix.data) < threshold * np.max(matrix)] = 0.0
    matrix.eliminate_zeros()


def _apply_expm(matrix):
    """Apply matrix exponential.

    TODO: implement other variants
    """
    exp = sp.csr_matrix(sp.linalg.expm(matrix.toarray().astype(DTYPE)))
    _threshold_matrix(exp)
    return exp


def _check_total_degree(degrees):
    """Ensures the sum(degree) > 0."""
    if degrees.sum() < 1e-10:
        raise Exception("The total degree = 0, we cannot proceed further")


def _get_spectral_gap(laplacian):
    """Compute spectral gap."""
    spectral_gap = np.round(max(np.real(sp.linalg.eigs(laplacian, which="SM", k=2)[0])), 8)
    L.info("Spectral gap = 10^%s", np.around(np.log10(spectral_gap), 2))
    return spectral_gap


class Constructor:
    """Parent class for generalized modularity constructor.

    This class encodes generalized modularity through the quality matrix and null models.
    Use the method prepare to load and compute scale independent quantities, and the method get_data
    to return quality matrix, null model, and possible global shift.
    """

    def __init__(self, graph, with_spectral_gap=False, **kwargs):
        """The constructor calls the prepare method upon initialisation.

        Args:
            graph (csgraph): graph for which to run clustering
            with_spectral_gap (bool): set to True to use spectral gap to rescale
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
        """Prepare the constructor with non-scale dependent computations."""

    def get_data(self, scale):
        """Return quality and null model at given scale as well as global shift (or None)."""


class constructor_linearized(Constructor):
    r"""Constructor for continuous linearized Markov Stability.

    The quality matrix is:

    .. math::

        F(t) = tA

    and with null model :math:`v_k=\pi=A1`.
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        degrees = np.array(self.graph.sum(1)).flatten()
        _check_total_degree(degrees)

        pi = degrees / degrees.sum()
        self.partial_null_model = np.array([pi, pi])

        if self.with_spectral_gap:
            laplacian = sp.csgraph.laplacian(self.graph, normed=False)
            self.spectral_gap = _get_spectral_gap(laplacian)
        self.partial_quality_matrix = (self.graph / degrees.sum()).astype(DTYPE)

    def get_data(self, scale):
        """Return quality and null model at given scale."""
        if self.with_spectral_gap:
            scale /= self.spectral_gap
        return {
            "quality": scale * self.partial_quality_matrix,
            "null_model": self.partial_null_model,
            "shift": float(1 - scale),
        }


class constructor_continuous_combinatorial(Constructor):
    r"""Constructor for continuous combinatorial Markov Stability.

    The quality matrix is:

    .. math::

        F(t) = \Pi\exp(-Lt)

    where :math:`L=D-A` and :math:`\Pi=\mathrm{diag}(\pi)`, with null model :math:`v_k=\pi=A1`.
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        laplacian, degrees = sp.csgraph.laplacian(self.graph, return_diag=True, normed=False)
        _check_total_degree(degrees)
        laplacian /= degrees.mean()
        pi = np.ones(self.graph.shape[0]) / self.graph.shape[0]
        self.partial_null_model = np.array([pi, pi], dtype=DTYPE)
        if self.with_spectral_gap:
            self.spectral_gap = _get_spectral_gap(laplacian)
        self.partial_quality_matrix = laplacian

    def get_data(self, scale):
        """Return quality and null model at given scale."""
        if self.with_spectral_gap:
            scale /= self.spectral_gap
        exp = _apply_expm(-scale * self.partial_quality_matrix)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


class constructor_continuous_normalized(Constructor):
    r"""Constructor for continuous normalized Markov Stability.

    The quality matrix is:

    .. math::

        F(t) = \Pi\exp(-Lt)

    where :math:`L=D^{-1}(D-A)` and :math:`\Pi=\mathrm{diag}(\pi)`
    and null model :math:`v_k=\pi=A1`.
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        laplacian, degrees = sp.csgraph.laplacian(self.graph, return_diag=True, normed=False)
        _check_total_degree(degrees)
        normed_laplacian = sp.diags(1.0 / degrees).dot(laplacian)

        pi = degrees / degrees.sum()
        self.partial_null_model = np.array([pi, pi], dtype=DTYPE)

        if self.with_spectral_gap:
            self.spectral_gap = _get_spectral_gap(normed_laplacian)
        self.partial_quality_matrix = normed_laplacian

    def get_data(self, scale):
        """Return quality and null model at given scale."""
        if self.with_spectral_gap:
            scale /= self.spectral_gap
        exp = _apply_expm(-scale * self.partial_quality_matrix)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


class constructor_signed_modularity(Constructor):
    """Constructor of signed modularity.

    This implementation is equation (18) of [1]_, where quality is the adjacency matrix and
    the null model is the difference between the sstandard modularity null models based on
    positive and negative degree vectors.

    References:
        .. [1] Gómez, S., Jensen, P., & Arenas, A. (2009). Analysis of community structure in
                networks of correlated data. Physical Review E, 80(1), 016114.
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
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

    def get_data(self, scale):
        """Return quality and null model at given scale."""
        return {
            "quality": scale * self.partial_quality_matrix,
            "null_model": self.partial_null_model,
        }


class constructor_directed(Constructor):
    r"""Constructor for directed Markov stability.

    The quality matrix is:

    .. math::

        F(t)=\Pi \exp\left(-\alpha L-\left(\frac{1-\alpha}{N}+\alpha \mathrm{diag}(a)\right)I\right)

    where :math:`a` denotes the vector of dangling nodes, i.e. :math:`a_i=1` if the
    out-degree :math:`d_i=0` and :math:`a_i=0` otherwise, :math:`I` denotes the identity matrix
    and :math:`0\le \alpha < 1` the damping factor, and associated null model
    :math:`v_0=v_1=\pi` given by the PageRank vector :math:`\pi`.
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
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

    def get_data(self, scale):
        """Return quality and null model at given scale."""
        exp = _apply_expm(scale * self.partial_quality_matrix)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}
