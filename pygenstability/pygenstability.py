"""main functions"""
import multiprocessing
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm

from . import generalized_louvain
from .constructors import load_constructor
from .io import save_results


def _get_chunksize(n_comp, pool):
    """Split jobs accross workers for speedup."""
    return max(1, int(n_comp / pool._processes))  # pylint: disable=protected-access


def _graph_checks(graph):
    """Do some checks and preprocessing of the graph."""

    if sp.csgraph.connected_components(graph)[0] > 1:
        raise Exception("Graph not connected, so we stop you here.")

    if sp.linalg.norm(graph - graph.T) > 0:
        print("Warning, your graph is directed!")

    if np.min(graph) <= 0:
        print("Warning, you have negative weights, consider using signed constructor.")

    return graph


def _get_times(min_time=-2.0, max_time=0.5, n_time=20, log_time=True, times=None):
    """Get the time vectors."""
    if times is not None:
        return times
    if log_time:
        return np.logspace(min_time, max_time, n_time)
    return np.linspace(min_time, max_time, n_time)


def _get_constructor_data(constructor, time):
    """Extract data from constructor."""
    data = constructor(time)
    quality_matrix, null_model = data[0], data[1]
    if len(data) == 3:
        global_shift = data[2]
    else:
        global_shift = None
    return quality_matrix, null_model, global_shift


def _get_params(all_locals):
    """Get run paramters from the local variables."""
    del all_locals["graph"]
    if callable(all_locals["constructor"]):
        all_locals["constructor"] = "custom constructor"
    return all_locals


def run(
    graph,
    constructor="linearized",
    min_time=-2.0,
    max_time=0.5,
    n_time=20,
    log_time=True,
    times=None,
    n_louvain=100,
    with_MI=True,
    n_louvain_MI=20,
    with_postprocessing=True,
    with_ttprime=True,
    result_file="results.pkl",
    n_workers=4,
    tqdm_disable=False,
):
    """Main funtion to compute clustering at various time scales.

    Args:
        graph (scipy.csgraph): graph to cluster
        constructor (str/function): name of the quality constructor,
            or custom constructor function. It must have two arguments, graph and time.
        min_time (float): minimum Markov time
        max_time (float): maximum Markov time
        n_time (int): number of time steps
        log_time (bool): use linear or log scales for times
        times (array): cutom time vector, if provided, it will overrid the other time arguments
        n_louvain (int): number of Louvain evaluations
        with_MI (bool): compute the mutual information between Louvain runs
        n_louvain_MI (int): number of randomly chosen Louvain run to estimate MI
        with_postprocessing (bool): apply the final postprocessing step
        with_ttprime (bool): compute the ttprime matrix
        results_file (str): path to the result file
        n_workers (int): number of workers for multiprocessing
        tqdm_disbale (bool): disable progress bars
    """
    run_params = _get_params(locals())
    graph = _graph_checks(graph)
    times = _get_times(
        min_time=min_time,
        max_time=max_time,
        n_time=n_time,
        log_time=log_time,
        times=times,
    )
    constructor = load_constructor(graph, constructor)
    pool = multiprocessing.Pool(n_workers)

    print("Start loop over times...")
    all_results = defaultdict(list)
    all_results["run_params"] = run_params
    for time in tqdm(times, disable=tqdm_disable):
        quality_matrix, null_model, global_shift = _get_constructor_data(
            constructor, time
        )
        louvain_results = run_several_louvains(
            quality_matrix, null_model, global_shift, n_louvain, pool
        )

        process_louvain_run(time, np.array(louvain_results), all_results)

        if with_MI:
            compute_mutual_information(
                louvain_results,
                all_results,
                pool,
                n_partitions=min(n_louvain_MI, n_louvain),
            )

        save_results(all_results, filename=result_file)

    if with_ttprime:
        print("Start computing ttprimes...")
        compute_ttprime(all_results, pool)

    if with_postprocessing:
        print("Apply postprocessing...")
        apply_postprocessing(all_results, pool, constructor=constructor)

    save_results(all_results, filename=result_file)
    pool.close()

    return all_results


def process_louvain_run(time, louvain_results, all_results, mutual_information=None):
    """convert the louvain outputs to useful data and save it"""
    best_run_id = np.argmax(louvain_results[:, 0])
    all_results["times"].append(time)
    all_results["number_of_communities"].append(
        np.max(louvain_results[best_run_id, 1]) + 1
    )
    all_results["stability"].append(louvain_results[best_run_id, 0])
    all_results["community_id"].append(louvain_results[best_run_id, 1])

    if mutual_information is not None:
        all_results["mutual_information"].append(mutual_information)


def compute_mutual_information(louvain_results, all_results, pool, n_partitions=10):
    """Compute the mutual information between the first n_partitions"""
    selected_partitions = louvain_results[:n_partitions, 1]
    worker = WorkerMI(selected_partitions)
    index_pairs = [[i, j] for i in range(n_partitions) for j in range(n_partitions)]
    chunksize = _get_chunksize(len(index_pairs), pool)
    all_results["mutual_information"].append(
        np.mean(list(pool.imap(worker, index_pairs, chunksize=chunksize)))
    )


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
    rows, cols, values = sp.find(sp.tril(matrix))
    return (rows, cols), values


class WorkerLouvain:
    """worker for Louvain runs"""

    def __init__(self, quality_indices, quality_values, null_model, global_shift):
        self.quality_indices = quality_indices
        self.quality_values = quality_values
        self.null_model = null_model
        self.global_shift = global_shift

    def __call__(self, i):
        stability, community_id = generalized_louvain.run_louvain(
            self.quality_indices[0],
            self.quality_indices[1],
            self.quality_values,
            len(self.quality_values),
            self.null_model,
            np.shape(self.null_model)[0],
            1.0,
        )
        if self.global_shift is not None:
            return stability + self.global_shift, community_id
        return stability, community_id


class WorkerQuality:
    """worker for Louvain runs"""

    def __init__(self, qualities_index, null_model, global_shift):
        self.quality_indices = qualities_index[0]
        self.quality_values = qualities_index[1]
        self.null_model = null_model
        self.global_shift = global_shift

    def __call__(self, partition_id):
        quality = generalized_louvain.evaluate_quality(
            self.quality_indices[0],
            self.quality_indices[1],
            self.quality_values,
            len(self.quality_values),
            self.null_model,
            np.shape(self.null_model)[0],
            1.0,
            partition_id,
        )
        if self.global_shift is not None:
            return quality + self.global_shift
        return quality


def run_several_louvains(quality_matrix, null_model, global_shift, n_runs, pool):
    """run several louvain on the current quality matrix"""

    quality_indices, quality_values = _to_indices(quality_matrix)
    worker = WorkerLouvain(quality_indices, quality_values, null_model, global_shift)

    chunksize = _get_chunksize(n_runs, pool)
    return np.array(
        list(pool.imap_unordered(worker, range(n_runs), chunksize=chunksize))
    )


def compute_ttprime(all_results, pool):
    """compute ttprime from the stability results"""
    index_pairs = [
        [i, j]
        for i in range(len(all_results["times"]))
        for j in range(len(all_results["times"]))
    ]

    worker = WorkerMI(all_results["community_id"])
    chunksize = _get_chunksize(len(index_pairs), pool)
    ttprime_list = list(
        tqdm(
            pool.imap(worker, index_pairs, chunksize=chunksize), total=len(index_pairs)
        )
    )

    all_results["ttprime"] = np.zeros(
        [len(all_results["times"]), len(all_results["times"])]
    )
    for i, ttp in enumerate(ttprime_list):
        all_results["ttprime"][index_pairs[i][0], index_pairs[i][1]] = ttp


def apply_postprocessing(all_results, pool, constructor, tqdm_disable=False):
    """apply postprocessing"""

    all_results_raw = all_results.copy()

    for i, time in tqdm(
        enumerate(all_results["times"]),
        total=len(all_results["times"]),
        disable=tqdm_disable,
    ):
        quality_matrix, null_model, global_shift = _get_constructor_data(
            constructor, time
        )
        worker = WorkerQuality(_to_indices(quality_matrix), null_model, global_shift)
        best_quality_id = np.argmax(
            list(
                pool.map(
                    worker,
                    all_results_raw["community_id"],
                    chunksize=_get_chunksize(
                        len(all_results_raw["community_id"]), pool
                    ),
                )
            )
        )

        all_results["community_id"][i] = all_results_raw["community_id"][
            best_quality_id
        ]
        all_results["stability"][i] = all_results_raw["stability"][best_quality_id]
        all_results["number_of_communities"][i] = all_results_raw[
            "number_of_communities"
        ][best_quality_id]
