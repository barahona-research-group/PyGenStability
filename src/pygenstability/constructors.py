r"""Module to create constructors of quality matrix and null models.

The generalized Markov Stability is given as

.. math::

    Q_{gen}(t,H) = \mathrm{Tr} \left [H^T \left (F(t)-\sum_{k=1}^m v_{2k-1} v_{2k}^T\right)H\right]

where :math:`F(t)` is the quality matrix and :math:`v_k` are null model vectors.

In the following we denote by :math:`A` the adjacency matrix of a graph with :math:`N` nodes and
:math:`M` edges. The out-degree of the graph is given by :math:`d=A\boldsymbol{1}`, where
:math:`\boldsymbol{1}` is the vector of ones, and we denote the diagonal degree matrix by
:math:`D=\mathrm{diag}(d)`.
"""

import logging
import sys

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

L = logging.getLogger(__name__)
THRESHOLD = 1e-8
_DTYPE = np.float64


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


def _limit_numpy(f):
    """Wrapper to limit threads used by numpy."""

    @threadpool_limits.wrap(limits=1, user_api="blas")
    @threadpool_limits.wrap(limits=1, user_api="openmp")
    def limit(*args, **kwargs):
        return f(*args, **kwargs)

    return limit


def _compute_spectral_decomp(matrix):
    """Solve eigenalue problem for symmetric matrix."""
    lambdas, v = la.eigh(matrix.toarray())
    vinv = la.inv(v)  # TODO: we could take v.T if we know that v is already orthonormal
    return lambdas, v, vinv


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
    """Parent class for generalized Markov Stability constructor.

    This class encodes generalized Markov Stability through the quality matrix and null models.
    Use the method prepare to load and compute scale independent quantities, and the method get_data
    to return quality matrix, null model, and possible global shift.
    """

    def __init__(self, graph, with_spectral_gap=False, exp_comp_mode="spectral", **kwargs):
        """The constructor calls the prepare method upon initialisation.

        Args:
            graph (csgraph): graph for which to run clustering
            with_spectral_gap (bool): set to True to use spectral gap to rescale
            kwargs (dict): any other properties to pass to the constructor.
            exp_comp_mode (str): mode to compute matrix exponential, can be expm or spectral
        """
        self.graph = sp.csr_matrix(graph)
        self.with_spectral_gap = with_spectral_gap
        self.spectral_gap = None
        self.exp_comp_mode = exp_comp_mode

        # these variables can be used in prepare method
        self.partial_quality_matrix = None
        self.partial_null_model = None
        self.spectral_decomp = None, None, None
        self.degrees = None
        self.threshold = THRESHOLD

        self.prepare(**kwargs)

    def _get_exp(self, scale):
        """Compute matrix exponential at a given scale."""
        if self.exp_comp_mode == "expm":
            # compute matrix exponential via Pade approximation
            exp = sp.linalg.expm(-scale * self.partial_quality_matrix.toarray().astype(_DTYPE))
        if self.exp_comp_mode == "spectral":
            # compute matrix exponential via spectral decomposition
            lambdas, v, vinv = self.spectral_decomp
            exp = v @ np.diag(np.exp(-scale * lambdas)) @ vinv

        # we cut values in exponential matrix that are smaller than 1e-8 the maximum value
        exp[np.abs(exp) < self.threshold * np.max(exp)] = 0.0
        return sp.csc_matrix(exp)

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""

    def get_data(self, scale):
        """Return quality and null model at given scale as well as global shift (or None).

        User has to define the _get_data so we can enure numpy does not use multiple threads
        """
        return self._get_data(scale)

    def _get_data(self, scale):
        """Method to be defined in child classes for get_data."""


class constructor_linearized(Constructor):
    r"""Constructor for continuous linearized Markov Stability.

    The quality matrix is:

    .. math::

        F(t) = t\frac{A}{2M},

    and the associated null model is :math:`v_1=v_2=\frac{d}{2M}`.
    """

    @_limit_numpy
    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        degrees = np.array(self.graph.sum(1)).flatten()
        _check_total_degree(degrees)

        pi = degrees / degrees.sum()
        self.partial_null_model = np.array([pi, pi])

        if self.with_spectral_gap:
            laplacian = sp.csgraph.laplacian(self.graph, normed=False)
            self.spectral_gap = _get_spectral_gap(laplacian)
        self.partial_quality_matrix = (self.graph / degrees.sum()).astype(_DTYPE)

    @_limit_numpy
    def _get_data(self, scale):
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

    This implementation follows equation (16) in [1]_. The quality matrix is:

    .. math::

        F(t) = \Pi\exp(-tL/<d>)

    where :math:`<d>=(\boldsymbol{1}^T D \boldsymbol{1})/N` is the average degree,
    :math:`L=D-A` is the combinatorial Laplacian and :math:`\Pi=\mathrm{diag}(\pi)`,
    with null model :math:`v_1=v_2=\pi=\frac{\boldsymbol{1}}{N}`.

    References:
        .. [1]  Lambiotte, R., Delvenne, J.-C., & Barahona, M. (2019). Random Walks, Markov
                  Processes and the Multiscale Modular Organization of Complex Networks.
                  IEEE Trans. Netw. Sci. Eng., 1(2), p. 76-90.
    """

    @_limit_numpy
    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        laplacian, degrees = sp.csgraph.laplacian(self.graph, return_diag=True, normed=False)
        _check_total_degree(degrees)
        laplacian /= degrees.mean()
        pi = np.ones(self.graph.shape[0]) / self.graph.shape[0]
        self.partial_null_model = np.array([pi, pi], dtype=_DTYPE)
        if self.with_spectral_gap:
            self.spectral_gap = _get_spectral_gap(laplacian)

        if self.exp_comp_mode == "spectral":
            self.spectral_decomp = _compute_spectral_decomp(laplacian)
        if self.exp_comp_mode == "expm":
            self.partial_quality_matrix = laplacian

    @_limit_numpy
    def _get_data(self, scale):
        """Return quality and null model at given scale."""
        if self.with_spectral_gap:
            scale /= self.spectral_gap

        exp = self._get_exp(scale)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


class constructor_continuous_normalized(Constructor):
    r"""Constructor for continuous normalized Markov Stability.

    This implementation follows equation (10) in [1]_. The quality matrix is:

    .. math::

        F(t) = \Pi\exp(-tL)

    where :math:`L=D^{-1}(D-A)` is the random-walk normalized Laplacian and
    :math:`\Pi=\mathrm{diag}(\pi)` with null model :math:`v_1=v_2=\pi=\frac{d}{2M}`.
    """

    @_limit_numpy
    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        # compute combinatorial Laplacian and degrees
        laplacian, degrees = sp.csgraph.laplacian(self.graph, return_diag=True, normed=False)
        _check_total_degree(degrees)

        if self.exp_comp_mode == "spectral":
            # store degrees
            self.degrees = degrees
            # compute symmetrically normalised Laplacian
            sym_normed_laplacian = sp.csgraph.laplacian(self.graph, normed=True)

        if self.exp_comp_mode == "expm" or self.with_spectral_gap:
            # compute random-walk normalised Laplacian
            D_inv = sp.diags(1.0 / degrees)
            rw_normed_laplacian = D_inv @ laplacian

        # define stationary distribution and set as null model
        pi = degrees / degrees.sum()
        self.partial_null_model = np.array([pi, pi], dtype=_DTYPE)

        if self.with_spectral_gap:
            # compute spectral gap of random-walk normalised Laplacian
            self.spectral_gap = _get_spectral_gap(rw_normed_laplacian)

        if self.exp_comp_mode == "spectral":
            # compute spectral decomposition of symmetric normalised Laplacian
            self.spectral_decomp = _compute_spectral_decomp(sym_normed_laplacian)
        if self.exp_comp_mode == "expm":
            self.partial_quality_matrix = rw_normed_laplacian

    @_limit_numpy
    def _get_data(self, scale):
        """Return quality and null model at given scale."""
        if self.with_spectral_gap:
            scale /= self.spectral_gap
        # compute matrix exponential
        exp = self._get_exp(scale)

        if self.exp_comp_mode == "spectral":
            # we need to transfrom exp of symmetrically normalised Laplacian to
            # obtain exp of random-walk normalised Laplacian
            D_sqrt_inv = sp.diags(1.0 / np.sqrt(self.degrees))
            D_sqrt = sp.diags(np.sqrt(self.degrees))
            exp = D_sqrt_inv @ exp @ D_sqrt

        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


class constructor_signed_modularity(Constructor):
    """Constructor of signed modularity.

    This implementation is equation (18) of [2]_, where quality is the adjacency matrix and
    the null model is the difference between the standard modularity null models based on
    positive and negative degree vectors.

    References:
        .. [2] Gomez, S., Jensen, P., & Arenas, A. (2009). Analysis of community structure in
                networks of correlated data. Physical Review E, 80(1), 016114.
    """

    @_limit_numpy
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

    @_limit_numpy
    def _get_data(self, scale):
        """Return quality and null model at given scale."""
        return {
            "quality": scale * self.partial_quality_matrix,
            "null_model": self.partial_null_model,
        }


class constructor_signed_combinatorial(Constructor):
    r"""Constructor for continuous signed combinatorial Markov Stability.

    This implementation follows equation (19) in [3]_. The quality matrix is:

    .. math::

        F(t) = \exp(-tL)^T\exp(-tL)

    where :math:`L=D_{\mathrm{abs}}-A` is the signed combinatorial Laplacian,
    :math:`D_{\mathrm{abs}}=\mathrm{diag}(d_\mathrm{abs})` the diagonal matrix of absolute node
    strengths :math:`d_\mathrm{abs}`, and the associated null model  is given by
    :math:`v_1=v_2=\boldsymbol{0}`, where :math:`\boldsymbol{0}` is the vector of zeros.

    References:
        .. [3]  Schaub, M., Delvenne, J.-C., Lambiotte, R., & Barahona, M. (2019). Multiscale
                  dynamical embeddings of complex networks. Physical Review E, 99(6), 062308.
    """

    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        degrees_abs = np.array(abs(self.graph).sum(1)).flatten()
        laplacian = sp.diags(degrees_abs) - self.graph

        if self.exp_comp_mode == "spectral":  # pragma: no cover
            self.spectral_decomp = _compute_spectral_decomp(laplacian)
        if self.exp_comp_mode == "expm":
            self.partial_quality_matrix = laplacian

        zeros = np.zeros(self.graph.shape[0])
        self.partial_null_model = np.array([zeros, zeros])

    def get_data(self, scale):
        """Return quality and null model at given scale."""
        exp = self._get_exp(scale)
        quality_matrix = exp.T.dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


class constructor_directed(Constructor):
    r"""Constructor for directed Markov stability.

    The quality matrix is:

    .. math::

        F(t)=\Pi \exp\left(t \left(M(\alpha)-I\right)\right)

    where :math:`I` denotes the identity matrix, :math:`M(\alpha)` is the transition matrix of a
    random walk with teleportation and damping factor :math:`0\le \alpha < 1`, and
    :math:`\Pi=\mathrm{diag}(\pi)` for the associated null model :math:`v_1=v_2=\pi` given by the
    eigenvector solving :math:`\pi M(\alpha) = \pi`, which is related to PageRank. See [1]_ for
    details.

    The transition matrix :math:`M(\alpha)` is given by

    .. math::

        M(\alpha) = \alpha D^{-1}A+\left((1-\alpha)I+\alpha \mathrm{diag}(a)\right)
        \frac{\boldsymbol{1}\boldsymbol{1}^T}{N},

    where :math:`D` denotes the diagonal matrix of out-degrees with :math:`D_{ii}=1` if the
    out-degree :math:`d_i=0` and :math:`a` denotes the vector of dangling nodes, i.e. :math:`a_i=1`
    if the out-degree :math:`d_i=0` and :math:`a_i=0` otherwise.
    """

    @_limit_numpy
    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        assert (
            self.exp_comp_mode == "expm"
        ), 'exp_comp_mode="expm" is required for "constructor_directed"'

        alpha = kwargs.get("alpha", 0.8)
        n_nodes = self.graph.shape[0]
        ones = np.ones((n_nodes, n_nodes)) / n_nodes

        out_degrees = np.array(self.graph.sum(1)).flatten()
        _check_total_degree(out_degrees)
        dinv = np.divide(1, out_degrees, where=out_degrees != 0)

        self.partial_quality_matrix = sp.csr_matrix(
            alpha * np.diag(dinv).dot(self.graph.toarray())
            + ((1 - alpha) * np.diag(np.ones(n_nodes)) + np.diag(alpha * (dinv == 0.0))).dot(ones)
            - np.eye(n_nodes)
        )

        pi = abs(sp.linalg.eigs(self.partial_quality_matrix.transpose(), which="SM", k=1)[1][:, 0])
        pi /= pi.sum()
        self.partial_null_model = np.array([pi, pi])

    @_limit_numpy
    def _get_data(self, scale):
        """Return quality and null model at given scale."""
        exp = self._get_exp(-scale)
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(exp)
        return {"quality": quality_matrix, "null_model": self.partial_null_model}


class constructor_linearized_directed(Constructor):
    r"""Constructor for linearized directed Markov stability.

    The quality matrix is:

    .. math::

        F(t)=\Pi t M(\alpha)

    where :math:`M(\alpha)` is the transition matrix of a random walk with teleportation and
    damping factor :math:`0\le \alpha < 1`, and :math:`\Pi=\mathrm{diag}(\pi)` for the associated
    null model :math:`v_1=v_2=\pi` given by the eigenvector solving :math:`\pi M(\alpha) = \pi`,
    which is related to PageRank.

    The transition matrix :math:`M(\alpha)` is given by

    .. math::

        M(\alpha) = \alpha D^{-1}A+\left((1-\alpha)I+\alpha \mathrm{diag}(a)\right)
        \frac{\boldsymbol{1}\boldsymbol{1}^T}{N},

    where :math:`I` denotes the identity matrix, :math:`D` denotes the diagonal matrix of
    out-degrees with :math:`D_{ii}=1` if the out-degree :math:`d_i=0` and :math:`a` denotes the
    vector of dangling nodes, i.e. :math:`a_i=1` if the out-degree :math:`d_i=0` and :math:`a_i=0`
    otherwise.
    """

    @_limit_numpy
    def prepare(self, **kwargs):
        """Prepare the constructor with non-scale dependent computations."""
        alpha = kwargs.get("alpha", 0.8)
        n_nodes = self.graph.shape[0]

        out_degrees = np.array(self.graph.sum(1)).flatten()
        _check_total_degree(out_degrees)
        dinv = np.divide(1, out_degrees, where=out_degrees != 0)

        if alpha < 1:
            ones = np.ones((n_nodes, n_nodes)) / n_nodes

            self.partial_quality_matrix = sp.csr_matrix(
                alpha * np.diag(dinv).dot(self.graph.toarray())
                + ((1 - alpha) * np.diag(np.ones(n_nodes)) + np.diag(alpha * (dinv == 0.0))).dot(
                    ones
                )
                - np.eye(n_nodes)
            )

        if alpha == 1:
            self.partial_quality_matrix = sp.csr_matrix(
                sp.diags(dinv).dot(self.graph) - sp.diags(np.ones(n_nodes))
            )

        pi = abs(sp.linalg.eigs(self.partial_quality_matrix.transpose(), which="SM", k=1)[1][:, 0])
        pi /= pi.sum()
        self.partial_null_model = np.array([pi, pi])

    @_limit_numpy
    def _get_data(self, scale):
        """Return quality and null model at given scale."""
        quality_matrix = sp.diags(self.partial_null_model[0]).dot(
            scale * self.partial_quality_matrix
        )
        return {"quality": quality_matrix, "null_model": self.partial_null_model}
