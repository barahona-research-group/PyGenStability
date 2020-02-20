"""main functions"""
import multiprocessing
import logging
import os

import numpy as np
import scipy as sc
import networkx as nx

from sklearn.metrics.cluster import normalized_mutual_info_score

from generalizedLouvain_API import run_louvain, evaluate_quality
from .io import save
from .constructors import _load_constructor

L = logging.getLogger("pygenstability")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def _graph_checks(graph):
    """do some checks and preprocessing of the graph"""
    if not nx.is_connected(graph):
        L.warning("Graph not connected, so we will use the largest connected component")
        graph = max(nx.connected_components(graph), key=len)

    if nx.is_directed(graph):
        L.warning(
            "given graph is directed, we convert it to undirected, as directed not implemented yet"
        )
        graph = graph.to_undirected()

    return graph


def run(graph, params):
    """main funtion to compute clustering at various time scales"""

    graph = _graph_checks(graph)

    constructor = _load_constructor(params["constructor"])

    if params["log_time"]:
        times = np.logspace(params["min_time"], params["max_time"], params["n_time"])
    else:
        times = np.linspace(params["min_time"], params["max_time"], params["n_time"])

    if params["save_qualities"]:
        quality_matrices = []
        null_models = []

    pool = multiprocessing.Pool(params["n_workers"])

    all_results = {"times": []}
    for time in times:
        L.info("Computing time 10^" + str(np.round(np.log10(time), 3)))

        quality_matrix, null_model = constructor(graph, time)

        if params["save_qualities"]:
            quality_matrices.append(quality_matrix)
            null_models.append(null_model)

        louvain_results = run_several_louvains(
            quality_matrix, null_model, params["n_runs"], pool
        )

        process_louvain_run(time, np.array(louvain_results), all_results)

        if params["compute_mutual_information"]:
            compute_mutual_information(
                louvain_results, all_results, pool, n_partitions=params["n_partitions"]
            )

        save(all_results)

    if params["compute_ttprime"]:
        compute_ttprime(all_results, pool)

    if params["apply_postprocessing"]:
        if params["save_qualities"]:
            apply_postprocessing(
                all_results,
                pool,
                quality_matrices=quality_matrices,
                null_models=null_models,
            )
        else:
            apply_postprocessing(
                all_results, pool, graph=graph, constructor=constructor
            )

    return all_results


def process_louvain_run(time, louvain_results, all_results, mutual_information=None):
    """convert the louvain outputs to useful data and save it"""

    if "time" not in all_results:
        all_results["time"] = []
    if "number_of_communities" not in all_results:
        all_results["number_of_communities"] = []
    if "stability" not in all_results:
        all_results["stability"] = []
    if "community_id" not in all_results:
        all_results["community_id"] = []

    best_run_id = np.argmax(louvain_results[:, 0])
    all_results["times"].append(time)
    all_results["number_of_communities"].append(
        np.max(louvain_results[best_run_id, 1]) + 1
    )
    all_results["stability"].append(louvain_results[best_run_id, 0])
    all_results["community_id"].append(louvain_results[best_run_id, 1])

    if mutual_information is not None:
        if "mutual_information" not in all_results:
            all_results["mutual_information"] = []
        all_results["mutual_information"].append(mutual_information)


def compute_mutual_information(louvain_results, all_results, pool, n_partitions=10):
    """compute the mutual information between the top partitions"""
    top_run_ids = np.argsort(louvain_results[:, 0])[-n_partitions:]
    top_partitions = louvain_results[top_run_ids, 1]
    index_pairs = [[i, j] for i in range(n_partitions) for j in range(n_partitions)]

    worker = WorkerMI(top_partitions)

    if "mutual_information" not in all_results:
        all_results["mutual_information"] = []
    all_results["mutual_information"].append(np.mean(pool.map(worker, index_pairs)))


class WorkerMI:
    """worker for Louvain runs"""

    def __init__(self, top_partitions):
        self.top_partitions = top_partitions

    def __call__(self, index_pair):
        return normalized_mutual_info_score(
            self.top_partitions[index_pair[0]],
            self.top_partitions[index_pair[1]],
            average_method="arithmetic",
        )


def _to_indices(matrix):
    """convert a sparse matrix to indices and values"""
    indices = sc.sparse.tril(matrix).nonzero()
    values = [matrix[index[0], index[1]] for index in zip(*indices)]
    return indices, values


class WorkerLouvain:
    """worker for Louvain runs"""

    def __init__(self, quality_matrix, null_model):
        self.quality_indices, self.quality_values = _to_indices(quality_matrix)
        self.null_model = null_model

    def __call__(self, i):
        stability, community_id = run_louvain(
            self.quality_indices[0],
            self.quality_indices[1],
            self.quality_values,
            len(self.quality_values),
            self.null_model,
            np.shape(self.null_model)[0],
            1.0,
        )
        return stability, community_id


class WorkerQuality:
    """worker for Louvain runs"""

    def __init__(self, quality_matrix, null_model):
        self.quality_indices, self.quality_values = _to_indices(quality_matrix)
        self.null_model = null_model

    def __call__(self, partition_id):
        quality = evaluate_quality(
            self.quality_indices[0],
            self.quality_indices[1],
            self.quality_values,
            len(self.quality_values),
            self.null_model,
            np.shape(self.null_model)[0],
            1.0,
            partition_id,
        )
        return quality


def run_several_louvains(quality_matrix, null_model, n_runs, pool):
    """run several louvain on the current quality matrix"""
    worker = WorkerLouvain(quality_matrix, null_model)
    return np.array(pool.map(worker, range(n_runs)))


def compute_ttprime(all_results, pool):
    """compute ttprime from the stability results"""
    index_pairs = [
        [i, j]
        for i in range(len(all_results["times"]))
        for j in range(len(all_results["times"]))
    ]

    worker = WorkerMI(all_results["community_id"])
    ttprime_list = pool.map(worker, index_pairs)

    all_results["ttprime"] = np.zeros(
        [len(all_results["times"]), len(all_results["times"])]
    )
    for i, ttp in enumerate(ttprime_list):
        all_results["ttprime"][index_pairs[i][0], index_pairs[i][1]] = ttp


def apply_postprocessing(
    all_results,
    pool,
    graph=None,
    constructor=None,
    quality_matrices=None,
    null_models=None,
):
    """apply postprocessing"""

    all_results_raw = all_results.copy()

    for i, time in enumerate(all_results["times"]):
        L.info("Postprocessing, computing time 10^" + str(np.round(np.log10(time), 3)))

        if quality_matrices is None:
            quality_matrix, null_model = constructor(graph, time)
        else:
            quality_matrix, null_model = quality_matrices[i], null_models[i]

        worker = WorkerQuality(quality_matrix, null_model)
        qualities = pool.map(worker, all_results_raw["community_id"])

        best_quality_id = np.argmax(qualities)
        all_results["community_id"][i] = all_results_raw["community_id"][
            best_quality_id
        ]
        all_results["stability"][i] = all_results_raw["stability"][best_quality_id]
        all_results["number_of_communities"][i] = all_results_raw[
            "number_of_communities"
        ][best_quality_id]
