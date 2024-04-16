"""Construct geometric graphs from data for multiscale clustering."""

import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

from pygenstability.pygenstability import run as pgs_run
from pygenstability.plotting import plot_scan as pgs_plot_scan
from pygenstability.optimal_scales import identify_optimal_scales


def compute_kNN(D, k=5):
    """Computes kNN graph."""

    N = D.shape[0]

    # get k nearest neighbours for each point
    k_neighbours = np.argsort(D, axis=1)[:, 1 : k + 1]

    # initialise adjacency matrix
    A = np.zeros((N, N))

    # build kNN graph
    for i in range(N):
        for neighbour in k_neighbours[i]:
            A[i, neighbour] = D[i, neighbour]
            A[neighbour, i] = A[i, neighbour]

    return A


def compute_CkNN(D, k=5, delta=1):
    """Computes CkNN graph."""
    # obtain rescaled distance matrix, see CkNN paper
    darray_n_nbrs = np.partition(D, k)[:, [k]]
    ratio_matrix = D / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
    # threshold rescaled distance matrix by delta
    A = D * (ratio_matrix < delta)
    return A


class GraphConstruction:
    """Graph construction."""

    def __init__(
        self,
        metric="euclidean",
        method="cknn",
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
        self.adjacency_ = csr_matrix

    def fit(self, X):
        """Construct graph from samples-by-features matrix."""

        # compute normalised distance matrix
        D = squareform(pdist(X, metric=self.metric))
        D_norm = D / np.amax(D)

        # compute normalised similarity matrix
        S = 1 - D_norm

        # sparsify distance matrix with CkNN or kNN method
        if self.method == "cknn":
            sparse = compute_CkNN(D_norm, self.k, self.delta)

        elif self.method == "knn":
            sparse = compute_kNN(D_norm, self.k)

        # undirected distance backbone is given by sparse graph and MST
        mst = minimum_spanning_tree(D_norm)
        backbone = np.array((mst + mst.T + sparse + sparse.T) > 0, dtype=int)

        # apply distance threshold to backbone
        backbone *= np.array(D <= self.distance_threshold, dtype=int)

        # adjacency matrix has weights of similarity matrix
        self.adjacency_ = S * backbone

        return self.adjacency_


class DataClustering(GraphConstruction):
    """Data clustering."""

    def __init__(
        self,
        metric="euclidean",
        graph_method="cknn",
        k=5,
        delta=1.0,
        distance_threshold=np.inf,
        constructor="linearized",
        min_scale=-3.0,
        max_scale=0.0,
        n_scale=50,
        log_scale=True,
        scales=None,
        n_tries=100,
        with_NVI=True,
        n_NVI=20,
        with_postprocessing=True,
        with_ttprime=True,
        with_spectral_gap=True,
        exp_comp_mode="spectral",
        result_file="results.pkl",
        n_workers=4,
        tqdm_disable=False,
        with_optimal_scales=True,
        optimal_scales_kwargs=None,
        ms_method="louvain",
        constructor_kwargs=None,
    ):

        # initialise parameters for graph construction
        super().__init__(
            metric=metric,
            method=graph_method,
            k=k,
            delta=delta,
            distance_threshold=distance_threshold,
        )

        # initialise parameters for PyGenStability
        self.constructor = constructor
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scale = n_scale
        self.log_scale = log_scale
        self.scales = scales
        self.n_tries = n_tries
        self.with_NVI = with_NVI
        self.n_NVI = n_NVI
        self.with_postprocessing = with_postprocessing
        self.with_ttprime = with_ttprime
        self.with_spectral_gap = with_spectral_gap
        self.exp_comp_mode = exp_comp_mode
        self.result_file = result_file
        self.n_workers = n_workers
        self.tqdm_disable = tqdm_disable
        self.with_optimal_scales = with_optimal_scales
        self.optimal_scales_kwargs = optimal_scales_kwargs
        self.MS_method = ms_method
        self.constructor_kwargs = constructor_kwargs

        # attributes
        self.adjacency_ = csr_matrix
        self.results_ = {}
        self.labels_ = []

    def fit(self, X):

        # construct graph
        self.adjacency_ = csr_matrix(super().fit(X))

        # adapt optimal scales parameters
        if self.optimal_scales_kwargs is None:
            self.optimal_scales_kwargs = {"kernel_size": int(0.2 * self.n_scale)}

        # run PyGenStability
        self.results_ = pgs_run(
            self.adjacency_,
            self.constructor,
            self.min_scale,
            self.max_scale,
            self.n_scale,
            self.log_scale,
            self.scales,
            self.n_tries,
            self.with_NVI,
            self.n_NVI,
            self.with_postprocessing,
            self.with_ttprime,
            self.with_spectral_gap,
            self.exp_comp_mode,
            self.result_file,
            self.n_workers,
            self.tqdm_disable,
            self.with_optimal_scales,
            self.optimal_scales_kwargs,
            self.MS_method,
            self.constructor_kwargs,
        )

        # store labels of robust partitions
        self._postprocess_selected_partitions()

        return self.results_

    def _postprocess_selected_partitions(self):
        """Postprocess selected partitions."""

        self.labels_ = []

        # store labels of robust partitions
        for i in self.results_["selected_partitions"]:

            # only store non-trivial robust partitions
            robust_partition = self.results_["community_id"][i]
            if not np.allclose(robust_partition, np.zeros(self.adjacency_.shape[0])):
                self.labels_.append(robust_partition)

    def scale_selection(
        self, kernel_size=0.1, window_size=0.1, max_nvi=1, basin_radius=0.01
    ):
        """Identify optimal scales."""

        # transform relative values to absolute values
        if kernel_size < 1:
            kernel_size = int(kernel_size * self.results_["run_params"]["n_scale"])
        if window_size < 1:
            window_size = int(window_size * self.results_["run_params"]["n_scale"])
        if basin_radius < 1:
            basin_radius = int(basin_radius * self.results_["run_params"]["n_scale"])

        # apply scale selection algorithm
        self.results_ = identify_optimal_scales(
            self.results_,
            kernel_size=kernel_size,
            window_size=window_size,
            max_nvi=max_nvi,
            basin_radius=basin_radius,
        )

        # store labels of robust partitions
        self._postprocess_selected_partitions()

        return self.labels_

    def plot_scan(self):
        """Plot PyGenStability scan."""
        if self.results_ is None:
            return

        pgs_plot_scan(self.results_)

    def plot_robust_partitions(
        self, x_coord, y_coord, edge_width=1, node_size=20, cmap="tab20"
    ):
        """Plot robust partitions."""

        import matplotlib.pyplot as plt

        for m, partition in enumerate(self.labels_):

            # plot
            _, ax = plt.subplots(1, figsize=(10, 10))

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
            ax.set(xlabel="x", ylabel="y", title=f"Robust Partion {m+1}")
            plt.show()
