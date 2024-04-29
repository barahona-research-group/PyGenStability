"""Construct geometric graphs from data for multiscale clustering."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import kneighbors_graph

from pygenstability.contrib.sankey import plot_sankey as pgs_plot_sankey
from pygenstability.optimal_scales import identify_optimal_scales
from pygenstability.plotting import plot_scan as pgs_plot_scan
from pygenstability.pygenstability import run as pgs_run


def _compute_CkNN(D, k=5, delta=1):
    """Computes CkNN graph."""
    # obtain rescaled distance matrix, see CkNN paper
    darray_n_nbrs = np.partition(D, k)[:, [k]]
    ratio_matrix = D / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
    # threshold rescaled distance matrix by delta
    A = D * (ratio_matrix < delta)
    return A


class _GraphConstruction:
    """Graph construction."""

    def __init__(
        self,
        metric="euclidean",
        method="cknn-mst",
        k=5,
        delta=1.0,
        distance_threshold=np.inf,
    ):
        # parameters
        self.metric = metric
        self.method = method
        self.k = k
        self.delta = delta
        self.distance_threshold = distance_threshold

        # attributes
        self.adjacency_ = None

    def get_graph(self, X):
        """Construct graph from samples-by-features matrix."""
        # if precomputed take X as adjacency matrix
        if self.method == "precomputed":
            assert X.shape[0] == X.shape[1], "Precomputed matrix should be a square matrix."
            self.adjacency_ = X
            return self.adjacency_

        # compute normalised distance matrix
        D = squareform(pdist(X, metric=self.metric))
        D_norm = D / np.amax(D)

        # compute normalised similarity matrix
        S = 1 - D_norm

        # sparsify distance matrix with CkNN or kNN method
        if self.method == "cknn-mst":
            sparse = _compute_CkNN(D_norm, self.k, self.delta)

        elif self.method == "knn-mst":
            sparse = kneighbors_graph(D_norm, n_neighbors=self.k, metric="precomputed").toarray()

        # undirected distance backbone is given by sparse graph and MST
        mst = minimum_spanning_tree(D_norm)
        backbone = np.array((mst + mst.T + sparse + sparse.T) > 0, dtype=int)

        # apply distance threshold to backbone
        backbone *= np.array(D <= self.distance_threshold, dtype=int)

        # adjacency matrix has weights of similarity matrix
        self.adjacency_ = S * backbone

        return self.adjacency_


class DataClustering(_GraphConstruction):
    """Class for multiscale graph-based data clustering.

    This class provides an interface for multiscale graph-based data clustering [1]_
    with PyGenStability.

    Parameters
    ----------
    metric : str or function, default='euclidean'
        The distance metric to use. The distance function can be ‘braycurtis’, ‘canberra’,
        ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’,
        ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘yule’.

    graph_method : {'knn-mst', 'cknn-mst', 'precomputed'}, default='cknn-mst'
        Method to construct graph from sample-by-feature matrix:

        - 'knn-mst' will use k-Nearest Neighbor graph combined with Miniumus Spanning Tree.
        - 'cknn-mst' will use Continunous k-Nearest Neighbor graph [2]_ combined with
            Miniumus Spanning Tree.
        - 'precomputed' assumes that data is already provided as adjacency matrix of a
            sparse graph.

    k : int, default=5
        Number of neighbors considered in graph construction. This parameter is expected
        to be positive.

    delta : float, default=1.0
        Density parameter for Continunous k-Nearest Neighbor graph. This parameter is expected
        to be positive.

    distance_threshold : float, optional
        Optional thresholding of distance matrix.

    pgs_kwargs : dict, optional
        Parameters for PyGenStability, see documentation. Some possible arguments:

        - constructor (str/function): name of the generalized Markov Stability constructor,
            or custom constructor function. It must have two arguments, graph and scale.
        - min_scale (float): minimum Markov scale
        - max_scale (float): maximum Markov scale
        - n_scale (int): number of scale steps
        - with_spectral_gap (bool): normalise scale by spectral gap

    Attributes
    ----------
    adjacency_ : sparse matrix of shape (n_samples, n_samples)
        Sparse adjacency matrix of constructed graph.

    results_ : dict
        PyGenStability results dictionary, see documentation for all arguments.

    labels_ : list of ndarray
        List of robust partitions identified with optimal scale selection.

    References
    ----------
    .. [1] Z. Liu and M. Barahona, 'Graph-based data clustering via multiscale
        community detection', *Applied Network Science*, vol. 5, no. 1, p. 3,
        Dec. 2020, doi: 10.1007/s41109-019-0248-7.
    .. [2] T. Berry and T. Sauer, 'Consistent manifold representation for
        topological data analysis', *Foundations of Data Science*, vol. 1, no. 1,
        p. 1-38, Feb. 2019, doi: 10.3934/fods.2019001.
    """

    def __init__(
        self,
        metric="euclidean",
        graph_method="cknn-mst",
        k=5,
        delta=1.0,
        distance_threshold=np.inf,
        **pgs_kwargs,
    ):

        # initialise parameters for graph construction
        super().__init__(
            metric=metric,
            method=graph_method,
            k=k,
            delta=delta,
            distance_threshold=distance_threshold,
        )

        # store PyGenStability kwargs
        self.pgs_kwargs = pgs_kwargs

        # attributes
        self.results_ = {}

    @property
    def labels_(self):
        """Return labels for robust paritions."""
        labels = []

        assert (
            "selected_partitions" in self.results_.keys()
        ), "Run PyGenStability with optimal scale selection first."

        # store labels of robust partitions
        for i in self.results_["selected_partitions"]:

            # only return non-trivial robust partitions
            robust_partition = self.results_["community_id"][i]
            if not np.allclose(robust_partition, np.zeros(self.adjacency_.shape[0])):
                labels.append(robust_partition)

        return labels

    def fit(self, X):
        """Fit multiscale graph-based data clustering with PyGenStability from data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples,n_features) or \
            (n_samples,n_samples) if graph_method='precomputed'
            Data to fit

        Returns
        -------
        self : DataClustering
            The fitted multiscale graph-based data clustering.
        """
        # construct graph
        self.adjacency_ = csr_matrix(self.get_graph(X))

        # run PyGenStability
        self.results_ = pgs_run(self.adjacency_, **self.pgs_kwargs)

        return self

    def scale_selection(self, kernel_size=0.1, window_size=0.1, max_nvi=1, basin_radius=0.01):
        """Identify optimal scales [3].

        Parameters
        ----------
        kernel_size : int or float, default=0.1
            Size of kernel for average-pooling of the NVI(t,t') matrix. If float smaller
            than one it's the relative number of scales.

        window_size : int or float, default=0.1
            Size of window for moving mean, to smooth the pooled diagonal. If float smaller
            than one it's the relative number of scales.

        max_nvi: float, default=1
            Threshold for local minima of the pooled diagonal.

        basin_radius: int or float, default=0.01
            Radius of basin around local minima of the pooled diagonal. If float smaller
            than one it's the relative number of scales.

        Returns
        -------
        labels_ : list of ndarray
            List of robust partitions identified with optimal scale selection.

        References
        ----------
        .. [3] D. J. Schindler, J. Clarke, and M. Barahona, 'Multiscale Mobility Patterns and
               the Restriction of Human Movement', *arXiv:2201.06323*, 2023
        """
        # transform relative values to absolute values
        if kernel_size < 1:
            kernel_size = int(kernel_size * self.results_["run_params"]["n_scale"])
        if window_size < 1:
            window_size = int(window_size * self.results_["run_params"]["n_scale"])
        if basin_radius < 1:
            basin_radius = max(1, int(basin_radius * self.results_["run_params"]["n_scale"]))

        # apply scale selection algorithm
        self.results_ = identify_optimal_scales(
            self.results_,
            kernel_size=kernel_size,
            window_size=window_size,
            max_nvi=max_nvi,
            basin_radius=basin_radius,
        )

        return self.labels_

    def plot_scan(self, *args, **kwargs):
        """Plot summary figure for PyGenStability scan."""
        if self.results_ is None:
            return

        pgs_plot_scan(self.results_, *args, **kwargs)

    def plot_robust_partitions(
        self, x_coord, y_coord, edge_width=1.0, node_size=20.0, cmap="tab20", show=True
    ):
        """Plot robust partitions with graph layout.

        Parameters
        ----------
        x_coord : ndarray of shape (n_samples,)
            X-coordinates provided for samples.

        y_coord : ndarray of shape (n_samples,)
            Y-coordinates provided for samples.

        edge_width : float, default=1.0
            Edge width of graph. This parameter is expected to be positive.

        node_size : float, default=20.0
            Node size in graph. This parameter is expected to be positive.

        cmap : str, default='tab20'
            Color map for cluster colors.

        show : book, default=True
            Show the figures.

        Returns
        -------
        figs : All matplotlib figures

        """
        figs = []
        for m, partition in enumerate(self.labels_):
            fig, ax = plt.subplots(1, figsize=(10, 10))
            figs.append(fig)

            # plot edges
            for i in range(self.adjacency_.shape[0]):
                for j in range(i + 1, self.adjacency_.shape[0]):
                    if self.adjacency_[i, j] > 0:
                        ax.plot(
                            [x_coord[i], x_coord[j]],
                            [y_coord[i], y_coord[j]],
                            color="black",
                            alpha=0.5,
                            linewidth=edge_width,
                        )

            # plot nodes
            ax.scatter(x_coord, y_coord, s=node_size, c=partition, zorder=10, cmap=cmap)

            # set labels
            ax.set(
                xlabel="x",
                ylabel="y",
                title=f"Robust Partion {m+1} (with {len(np.unique(partition))} clusters)",
            )
        if show:
            plt.show()

    def plot_sankey(
        self,
        optimal_scales=True,
        live=False,
        filename="communities_sankey.html",
        scale_index=None,
    ):
        """Plot Sankey diagram.

        Parameters
        ----------
        optimal_scales : bool, default=True
            Plot Sankey diagram of robust partitions only or not.

        live : bool, default=False
            If True, interactive figure will appear in browser.

        filename : str, default="communities_sankey.html"
            Filename to save the plot.

        scale_index : bool
            Plot Sankey diagram for provided scale indices.

        Returns
        -------
        fig : plotly figure
            Sankey diagram figure.
        """
        # plot non-trivial optimal scales only
        if optimal_scales:
            n_partitions = len(self.labels_)
            # collect indices of non-trivial partitions
            scale_index = self.results_["selected_partitions"][:n_partitions]

        # plot Sankey diagram
        fig = pgs_plot_sankey(
            self.results_,
            optimal_scales=False,
            live=live,
            filename=filename,
            scale_index=scale_index,
        )

        return fig
