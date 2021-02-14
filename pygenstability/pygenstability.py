"""PyGenStability module."""
import logging
import multiprocessing
from collections import defaultdict
from functools import partial

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import entropy
from tqdm import tqdm

from pygenstability import generalized_louvain
from pygenstability.constructors import load_constructor
from pygenstability.io import save_results

L = logging.getLogger(__name__)
_DTYPE = np.float64


def _get_chunksize(n_comp, pool):
    """Split jobs accross workers for speedup."""
    return max(1, int(n_comp / pool._processes))  # pylint: disable=protected-access


def _graph_checks(graph, dtype=_DTYPE):
    """Do some checks and preprocessing of the graph."""
    graph = sp.csr_matrix(graph, dtype=dtype)
    if sp.csgraph.connected_components(graph)[0] > 1:
        raise Exception(
            "Graph not connected, with {} components".format(
                sp.csgraph.connected_components(graph)[0]
            )
        )

    if sp.linalg.norm(graph - graph.T) > 0:
        L.warning("Your graph is directed!")

    if np.min(graph) < 0:
        L.warning("You have negative weights, consider using signed constructor.")

    graph.eliminate_zeros()
    return graph


def _get_times(min_time=-2.0, max_time=0.5, n_time=20, log_time=True, times=None):
    """Get the time vectors."""
    if times is not None:
        return times
    if log_time:
        return np.logspace(min_time, max_time, n_time)
    return np.linspace(min_time, max_time, n_time)


def _get_params(all_locals):
    """Get run paramters from the local variables."""
    del all_locals["graph"]
    if callable(all_locals["constructor"]):
        all_locals["constructor"] = "custom constructor"
    return all_locals


def run(
    graph=None,
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
    with_spectral_gap=False,
    result_file="results.pkl",
    n_workers=4,
    tqdm_disable=False,
):
    """Main function to compute clustering at various time scales.

    Args:
        graph (scipy.csgraph): graph to cluster, if None, the constructor cannot be a str
        constructor (str/function): name of the quality constructor,
            or custom constructor function. It must have two arguments, graph and time.
        min_time (float): minimum Markov time
        max_time (float): maximum Markov time
        n_time (int): number of time steps
        log_time (bool): use linear or log scales for times
        times (array): custom time vector, if provided, it will override the other time arguments
        n_louvain (int): number of Louvain evaluations
        with_VI (bool): compute the variation of information between Louvain runs
        n_louvain_VI (int): number of randomly chosen Louvain run to estimate VI
        with_postprocessing (bool): apply the final postprocessing step
        with_ttprime (bool): compute the ttprime matrix
        with_spectral_gap (bool): normalise time by spectral gap
        result_file (str): path to the result file
        n_workers (int): number of workers for multiprocessing
        tqdm_disable (bool): disable progress bars
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
    constructor = load_constructor(constructor, graph, with_spectral_gap=with_spectral_gap)
    pool = multiprocessing.Pool(n_workers)

    L.info("Start loop over times...")
    all_results = defaultdict(list)
    all_results["run_params"] = run_params
    for time in tqdm(times, disable=tqdm_disable):
        quality_matrix, null_model, global_shift = constructor.get_data(time)
        louvain_results = _run_several_louvains(
            quality_matrix, null_model, global_shift, n_louvain, pool
        )
        communities = _process_louvain_run(time, louvain_results, all_results)

        if with_VI:
            _compute_variation_information(
                communities,
                all_results,
                pool,
                n_partitions=min(n_louvain_VI, n_louvain),
            )

        save_results(all_results, filename=result_file)

    if with_ttprime:
        L.info("Start computing ttprimes...")
        compute_ttprime(all_results, pool)

    if with_postprocessing:
        L.info("Apply postprocessing...")
        apply_postprocessing(all_results, pool, constructor=constructor)

    save_results(all_results, filename=result_file)
    pool.close()

    return all_results


def _process_louvain_run(time, louvain_results, all_results, variation_information=None):
    """Convert the louvain outputs to useful data and save it."""
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


def _compute_variation_information(communities, all_results, pool, n_partitions=10):
    """Compute an information measure between the first n_partitions."""
    selected_partitions = communities[:n_partitions]

    worker = partial(_evaluate_VI, top_partitions=selected_partitions)
    index_pairs = [[i, j] for i in range(n_partitions) for j in range(n_partitions)]
    chunksize = _get_chunksize(len(index_pairs), pool)
    all_results["variation_information"].append(
        np.mean(list(pool.imap(worker, index_pairs, chunksize=chunksize)))
    )


def _evaluate_VI(index_pair, top_partitions):
    """Worker for VI evaluations."""
    MI = mutual_info_score(
        top_partitions[index_pair[0]],
        top_partitions[index_pair[1]],
    )
    Ex = entropy(top_partitions[index_pair[0]])
    Ey = entropy(top_partitions[index_pair[1]])
    JE = Ex + Ey - MI
    if abs(JE) < 1e-8:
        return 0.0
    return (JE - MI) / JE


def _to_indices(matrix):
    """Convert a sparse matrix to indices and values."""
    rows, cols, values = sp.find(sp.tril(matrix))
    return (rows, cols), values


def _evaluate_louvain(_, quality_indices, quality_values, null_model, global_shift):
    """Worker for Louvain runs."""
    stability, community_id = generalized_louvain.run_louvain(
        quality_indices[0],
        quality_indices[1],
        quality_values,
        len(quality_values),
        null_model,
        np.shape(null_model)[0],
        1.0,
    )
    if global_shift is not None:
        return stability + global_shift, community_id
    return stability, community_id


def _evaluate_quality(partition_id, qualities_index, null_model, global_shift):
    """Worker for Louvain runs."""
    quality = generalized_louvain.evaluate_quality(
        qualities_index[0][0],
        qualities_index[0][1],
        qualities_index[1],
        len(qualities_index[1]),
        null_model,
        np.shape(null_model)[0],
        1.0,
        partition_id,
    )
    if global_shift is not None:
        return quality + global_shift
    return quality


def _run_several_louvains(quality_matrix, null_model, global_shift, n_runs, pool):
    """Run several louvain on the current quality matrix."""
    quality_indices, quality_values = _to_indices(quality_matrix)
    worker = partial(
        _evaluate_louvain,
        quality_indices=quality_indices,
        quality_values=quality_values,
        null_model=null_model,
        global_shift=global_shift,
    )

    chunksize = _get_chunksize(n_runs, pool)
    return list(pool.imap(worker, range(n_runs), chunksize=chunksize))


def compute_ttprime(all_results, pool):
    """Compute ttprime from the stability results."""
    index_pairs = [
        [i, j] for i in range(len(all_results["times"])) for j in range(len(all_results["times"]))
    ]

    worker = partial(_evaluate_VI, top_partitions=all_results["community_id"])
    chunksize = _get_chunksize(len(index_pairs), pool)
    ttprime_list = pool.map(worker, index_pairs, chunksize=chunksize)

    all_results["ttprime"] = np.zeros([len(all_results["times"]), len(all_results["times"])])
    for i, ttp in enumerate(ttprime_list):
        all_results["ttprime"][index_pairs[i][0], index_pairs[i][1]] = ttp


def apply_postprocessing(all_results, pool, constructor, tqdm_disable=False):
    """Apply postprocessing."""
    all_results_raw = all_results.copy()

    for i, time in tqdm(
        enumerate(all_results["times"]),
        total=len(all_results["times"]),
        disable=tqdm_disable,
    ):
        quality_matrix, null_model, global_shift = constructor.get_data(time)
        worker = partial(
            _evaluate_quality,
            qualities_index=_to_indices(quality_matrix),
            null_model=null_model,
            global_shift=global_shift,
        )
        best_quality_id = np.argmax(
            list(
                pool.map(
                    worker,
                    all_results_raw["community_id"],
                    chunksize=_get_chunksize(len(all_results_raw["community_id"]), pool),
                )
            )
        )

        all_results["community_id"][i] = all_results_raw["community_id"][best_quality_id]
        all_results["stability"][i] = all_results_raw["stability"][best_quality_id]
        all_results["number_of_communities"][i] = all_results_raw["number_of_communities"][
            best_quality_id
        ]
