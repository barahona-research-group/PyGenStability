"""main functions"""
import multiprocessing
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import itertools
from math import log

from . import generalized_louvain
from .constructors import load_constructor
from .io import save_results


def _get_chunksize(n_comp, pool):
    """Split jobs accross workers for speedup."""
    return max(1, int(n_comp / pool._processes))  # pylint: disable=protected-access


def _graph_checks(graph):
    """Do some checks and preprocessing of the graph."""

    if sp.csgraph.connected_components(graph)[0] > 1:
        raise Exception(
            "Graph not connected, with {} components".format(
                sp.csgraph.connected_components(graph)[0]
            )
        )

    if sp.linalg.norm(graph - graph.T) > 0:
        print("Warning, your graph is directed!")

    if np.min(graph) < 0:
        print("Warning, you have negative weights, consider using signed constructor.")

    graph.eliminate_zeros()
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
    with_VI=True,
    n_louvain_VI=20,
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
        with_VI (bool): compute the variation of information between Louvain runs
        n_louvain_VI (int): number of randomly chosen Louvain run to estimate VI
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
        communities = process_louvain_run(time, louvain_results, all_results)

        if with_VI:
            compute_variation_information(
                communities,
                all_results,
                pool,
                n_partitions=min(n_louvain_VI, n_louvain),
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


def process_louvain_run(time, louvain_results, all_results, variation_information=None):
    """convert the louvain outputs to useful data and save it"""
    stabilities = np.array([res[0] for res in louvain_results])
    communities = np.array([res[1] for res in louvain_results])

    best_run_id = np.argmax(stabilities)
    all_results["times"].append(time)
    all_results["number_of_communities"].append(np.max(communities[best_run_id]) + 1)
    all_results["stability"].append(stabilities[best_run_id])
    all_results["community_id"].append(communities[best_run_id])

    if variation_information is not None:
        all_results["variation_information"].append(variation_information)

    return communities


def compute_variation_information(communities, all_results, pool, n_partitions=10):
    """Compute an information measure between the first n_partitions"""
    selected_partitions = communities[:n_partitions]

    worker = WorkerVI(selected_partitions)
    index_pairs = [[i, j] for i in range(n_partitions) for j in range(n_partitions)]
    chunksize = _get_chunksize(len(index_pairs), pool)
    all_results["variation_information"].append(
        np.mean(list(pool.imap(worker, index_pairs, chunksize=chunksize)))
    )


class WorkerVI:
    """worker for VI evaluations"""

    def __init__(self, top_partitions):
        self.top_partitions = top_partitions

    def __call__(self, index_pair):

        MI = mutual_info_score(
            self.top_partitions[index_pair[0]], self.top_partitions[index_pair[1]],
        )
        Ex = entropy(self.top_partitions[index_pair[0]])
        Ey = entropy(self.top_partitions[index_pair[1]])
        JE = Ex + Ey - MI
        if abs(JE) < 1e-8:
            return 0.0
        return (JE - MI) / JE


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
    return list(pool.imap(worker, range(n_runs), chunksize=chunksize))


def compute_ttprime(all_results, pool):
    """compute ttprime from the stability results"""
    index_pairs = [
        [i, j]
        for i in range(len(all_results["times"]))
        for j in range(len(all_results["times"]))
    ]

    worker = WorkerVI(all_results["community_id"])
    chunksize = _get_chunksize(len(index_pairs), pool)
    ttprime_list = pool.map(worker, index_pairs, chunksize=chunksize)

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


def joint_entropy(x, y):
    """
    Calculate the entropy of a variable, or joint entropy of several variables.
    Parameters
    ----------
    x : array or list
    y : array or list

    """
    n_instances = len(x)
    H = 0
    X = [np.array(x), np.array(y)]
    for classes in itertools.product(*[set(x) for x in X]):
        v = np.array([True] * n_instances)
        for predictions, c in zip(X, classes):
            v = np.logical_and(v, predictions == c)
        p = np.mean(v)
        H += -p * np.log2(p) if p > 0 else 0
    return H


def entropy(labels):
    """Calculates the entropy for a labeling.
    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
