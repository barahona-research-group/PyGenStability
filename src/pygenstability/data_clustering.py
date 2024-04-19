"""Construct geometric graphs from data for multiscale clustering."""

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

from pygenstability.pygenstability import run as pgs_run
from pygenstability.plotting import plot_scan as pgs_plot_scan
from pygenstability.optimal_scales import identify_optimal_scales
from pygenstability.contrib.sankey import plot_sankey as pgs_plot_sankey


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
        self.adjacency_ = None

    def get_graph(self, X):
        """Construct graph from samples-by-features matrix."""

        # if precomputed take X as adjacency matrix
        if self.method == "precomputed":
            assert (
                X.shape[0] == X.shape[1]
            ), "Precomputed matrix should be a square matrix."
            self.adjacency_ = X
            return self.adjacency_

        # compute normalised distance matrix
        D = squareform(pdist(X, metric=self.metric))
        D_norm = D / np.amax(D)

        # compute normalised similarity matrix
        S = 1 - D_norm

        # sparsify distance matrix with CkNN or kNN method
        if self.method == "cknn":
            sparse = compute_CkNN(D_norm, self.k, self.delta)

        elif self.method == "knn":
            sparse = kneighbors_graph(
                D_norm, n_neighbors=self.k, metric="precomputed"
            ).toarray()

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
        """Construct graph and run PyGenStability for multiscale data clustering."""

        # construct graph
        self.adjacency_ = csr_matrix(self.get_graph(X))

        # run PyGenStability
        self.results_ = pgs_run(self.adjacency_, **self.pgs_kwargs)

        return self.results_

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
            basin_radius = max(
                1, int(basin_radius * self.results_["run_params"]["n_scale"])
            )

        # apply scale selection algorithm
        self.results_ = identify_optimal_scales(
            self.results_,
            kernel_size=kernel_size,
            window_size=window_size,
            max_nvi=max_nvi,
            basin_radius=basin_radius,
        )

    def plot_scan(self):
        """Plot PyGenStability scan."""
        if self.results_ is None:
            return

        pgs_plot_scan(self.results_)

    def plot_robust_partitions(
        self, x_coord, y_coord, edge_width=1, node_size=20, cmap="tab20"
    ):
        """Plot robust partitions."""

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
            ax.set(
                xlabel="x",
                ylabel="y",
                title=f"Robust Partion {m+1} (with {len(np.unique(partition))} clusters)",
            )
            plt.show()

    def plot_sankey(
        self,
        optimal_scales=True,
        live=False,
        filename="communities_sankey.html",
        scale_index=None,
    ):
        """Plot Sankey diagram."""

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
