r"""PyGenStability code to solve generalized Markov Stability including Markov stability.

The generalized Markov Stability is of the form

.. math::

    Q_{gen}(t,H) = \mathrm{Tr} \left [H^T \left (F(t)-\sum_{k=1}^m v_{2k-1} v_{2k}^T\right)H\right]

where :math:`F(t)` is the quality matrix and :math:`v_k` are null model vectors.
The choice of the quality matrix and null model vectors are arbitrary in the generalized
Markov Stability setting, and can be parametrised via built-in constructors, or specified by
the user via the constructor module.
"""

import itertools
import logging
import multiprocessing
from collections import defaultdict
from functools import partial
from functools import wraps
from time import time

try:
    import igraph as ig
    import leidenalg

    _NO_LEIDEN = False
except ImportError:  # pragma: no cover
    _NO_LEIDEN = True

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import entropy
from tqdm import tqdm

try:
    from pygenstability import generalized_louvain

    _NO_LOUVAIN = False
except ImportError:  # pragma: no cover
    _NO_LOUVAIN = True

from pygenstability.constructors import load_constructor
from pygenstability.io import save_results
from pygenstability.optimal_scales import identify_optimal_scales

L = logging.getLogger(__name__)
_DTYPE = np.float64


def _timing(f):  # pragma: no cover
    """Use as decorator to time a function excecution if logging is in DEBUG mode."""

    @wraps(f)
    def wrap(*args, **kw):
        if logging.root.level == logging.DEBUG:
            t_start = time()
            result = f(*args, **kw)
            t_end = time()
            with open("timing.csv", "a", encoding="utf-8") as file:
                print(f"{f.__name__}, {t_end - t_start}", file=file)
        else:
            result = f(*args, **kw)
        return result

    return wrap


def _get_chunksize(n_comp, pool):
    """Split jobs accross workers for speedup."""
    return max(1, int(n_comp / pool._processes))  # pylint: disable=protected-access


def _graph_checks(graph, dtype=_DTYPE):
    """Do some checks and preprocessing of the graph."""
    graph = sp.csr_matrix(graph, dtype=dtype)
    if sp.csgraph.connected_components(graph)[0] > 1:
        raise Exception(
            f"Graph not connected, with {sp.csgraph.connected_components(graph)[0]} components"
        )

    if sp.linalg.norm(graph - graph.T) > 0:
        L.warning("Your graph is directed!")

    if np.min(graph) < 0:
        L.warning("You have negative weights, consider using signed constructor.")

    graph.eliminate_zeros()
    return graph


def _get_scales(min_scale=-2.0, max_scale=0.5, n_scale=20, log_scale=True, scales=None):
    """Get the scale vectors."""
    if scales is not None:
        return scales
    if log_scale:
        return np.logspace(min_scale, max_scale, n_scale)
    return np.linspace(min_scale, max_scale, n_scale)


def _get_params(all_locals):
    """Get run paramters from the local variables."""
    del all_locals["graph"]
    if hasattr(all_locals["constructor"], "get_data"):
        all_locals["constructor"] = "custom constructor"
    return all_locals


@_timing
def _get_constructor_data(constructor, scales, pool, tqdm_disable=False):
    return list(
        tqdm(
            pool.imap(constructor.get_data, scales),
            total=len(scales),
            disable=tqdm_disable,
        )
    )


def _check_method(method):  # pragma: no cover
    if _NO_LEIDEN and _NO_LOUVAIN:
        raise Exception("Without Louvain or Leiden solver, we cannot run PyGenStability")

    if method == "louvain" and _NO_LOUVAIN:
        print("Louvain is not available, we fallback to leiden.")
        return "leiden"

    if method == "leiden" and _NO_LEIDEN:
        print("Leiden is not available, we fallback to louvain.")
        return "louvain"

    return method


@_timing
def run(
    graph=None,
    constructor="linearized",
    min_scale=-2.0,
    max_scale=0.5,
    n_scale=20,
    log_scale=True,
    scales=None,
    n_tries=100,
    with_all_tries=False,
    with_NVI=True,
    n_NVI=20,
    with_postprocessing=True,
    with_ttprime=True,
    with_spectral_gap=False,
    exp_comp_mode="spectral",
    result_file="results.pkl",
    n_workers=4,
    tqdm_disable=False,
    with_optimal_scales=True,
    optimal_scales_kwargs=None,
    method="louvain",
    constructor_kwargs=None,
):
    """This is the main function to compute graph clustering across scales with Markov Stability.

    This function needs a graph object  as an adjacency matrix encoded with scipy.csgraph.
    The default settings will provide a fast and generic run with linearized Markov Stability,
    which corresponds to modularity with a scale parameter. Other built-in constructors are
    available to perform Markov Stability with matrix exponential computations. Custom constructors
    can be added via the constructor module.
    Additional parameters can be used to set the range and number of scales, number of trials for
    generalized Markov Stability optimisation, with Louvain or Leiden algorithm.

    Args:
        graph (scipy.csgraph): graph to cluster, if None, the constructor cannot be a str
        constructor (str/function): name of the generalized Markov Stability constructor,
            or custom constructor function. It must have two arguments, graph and scale.
        min_scale (float): minimum Markov scale
        max_scale (float): maximum Markov scale
        n_scale (int): number of scale steps
        log_scale (bool): use linear or log scales for scales
        scales (array): custom scale vector, if provided, it will override the other scale arguments
        n_tries (int): number of generalized Markov Stability optimisation evaluations
        with_all_tries (bools): store all partitions with stability values found in different
            optimisation evaluations
        with_NVI (bool): compute NVI(t) between generalized Markov Stability optimisations
            at each scale t
        n_NVI (int): number of randomly chosen generalized Markov Stability optimisations
            to estimate NVI
        with_postprocessing (bool): apply the final postprocessing step
        with_ttprime (bool): compute the NVI(t,tprime) matrix to compare scales t and tprime
        with_spectral_gap (bool): normalise scale by spectral gap
        exp_comp_mode (str): mode to compute matrix exponential, can be expm or spectral
        result_file (str): path to the result file
        n_workers (int): number of workers for multiprocessing
        tqdm_disable (bool): disable progress bars
        with_optimal_scales (bool): apply optimal scale selection algorithm
        optimal_scales_kwargs (dict): kwargs to pass to optimal scale selection, see
            optimal_scale module.
        method (str): optimiation method, louvain or leiden
        constructor_kwargs (dict): additional kwargs to pass to constructor prepare method

    Returns:
        Results dict with the following entries
            - 'run_params': dict with parameters used to run the code
            - 'scales': scales of the scan
            - 'number_of_communities': number of communities at each scale
            - 'stability': value of stability cost function at each scale
            - 'community_id': community node labels at each scale
            - 'all_tries': all community node labels with stability values found in different
                optimisation evaluations at each scale (included if with_all_tries==True)
            - 'NVI': NVI(t) at each scale
            - 'ttprime': NVI(t,tprime) matrix
            - 'block_nvi': block NVI curve (included if with_optimal_scales==True)
            - 'selected_partitions': selected partitions (included if with_optimal_scales==True)

    """
    method = _check_method(method)
    run_params = _get_params(locals())
    graph = _graph_checks(graph)
    scales = _get_scales(
        min_scale=min_scale,
        max_scale=max_scale,
        n_scale=n_scale,
        log_scale=log_scale,
        scales=scales,
    )
    assert exp_comp_mode in ["spectral", "expm"]
    if constructor in ("directed", "linearized_directed", "signed"):
        L.info("We cannot use spectral exponential computation for directed contructor")
        exp_comp_mode = "expm"

    if constructor_kwargs is None:
        constructor_kwargs = {}
    constructor_kwargs.update(
        {"with_spectral_gap": with_spectral_gap, "exp_comp_mode": exp_comp_mode}
    )

    constructor = load_constructor(constructor, graph, **constructor_kwargs)
    with multiprocessing.Pool(n_workers) as pool:
        L.info("Precompute constructors...")
        constructor_data = _get_constructor_data(
            constructor, scales, pool, tqdm_disable=tqdm_disable
        )
        if method == "leiden":
            # pragma: no cover
            for data in constructor_data:
                assert all(data["null_model"][0] == data["null_model"][1])

        L.info("Optimise stability...")
        all_results = defaultdict(list)
        all_results["run_params"] = run_params

        # iterate through all Markov scales
        for i, t in tqdm(enumerate(scales), total=n_scale, disable=tqdm_disable):
            # run optimisation independently for n_tries
            results = _run_optimisations(constructor_data[i], n_tries, pool, method)
            communities = _process_runs(t, results, all_results)

            if with_NVI:
                _compute_NVI(communities, all_results, pool, n_partitions=min(n_NVI, n_tries))

            if with_all_tries:
                all_results["all_tries"].append(results)

            save_results(all_results, filename=result_file)

        if with_postprocessing:
            L.info("Apply postprocessing...")
            _apply_postprocessing(all_results, pool, constructor_data, tqdm_disable, method=method)

        if with_ttprime or with_optimal_scales:
            L.info("Compute ttprimes...")
            _compute_ttprime(all_results, pool)

            if with_optimal_scales:
                L.info("Identify optimal scales...")
                if optimal_scales_kwargs is None:
                    optimal_scales_kwargs = {
                        "kernel_size": max(2, int(0.1 * n_scale)),
                        "window_size": max(2, int(0.1 * n_scale)),
                        "basin_radius": max(1, int(0.01 * n_scale)),
                    }
                all_results = identify_optimal_scales(all_results, **optimal_scales_kwargs)

    save_results(all_results, filename=result_file)

    return dict(all_results)


def _process_runs(scale, results, all_results):
    """For each scale pick partition with highest stability among all iterations."""
    # collect results from different optimisation runs
    stabilities = np.array([res[0] for res in results])
    communities = np.array([res[1] for res in results])
    # find index for highest stability
    best_run_id = np.argmax(stabilities)
    # save results for partition with highest stability
    all_results["scales"].append(scale)
    all_results["number_of_communities"].append(len(np.unique(communities[best_run_id])))
    all_results["stability"].append(stabilities[best_run_id])
    # we assign strictly increasing community IDs
    all_results["community_id"].append(_assign_increasing_ids(communities[best_run_id]))

    return communities


def _assign_increasing_ids(community_id):
    """Assign strictly increasing community IDs starting from 0."""
    # get unique ids and their indices in input array
    unique_ids, ind = np.unique(community_id, return_index=True)
    # translate old ids to new ids
    new_id_dict = {unique_ids[np.argsort(ind)][i]: i for i in range(len(unique_ids))}
    return np.vectorize(new_id_dict.get)(community_id)


@_timing
def _compute_NVI(communities, all_results, pool, n_partitions=10):
    """Compute NVI measure between the first n_partitions."""
    selected_partitions = communities[:n_partitions]
    # prepare worker to compute NVI between selected partitions
    worker = partial(evaluate_NVI, partitions=selected_partitions)
    # we compute pairwise NVI only for i != j because NVI is a metric
    index_pairs = list(itertools.combinations(range(n_partitions), 2))
    chunksize = _get_chunksize(len(index_pairs), pool)
    # compute using pool of workers
    nvi_off_diagonal = list(pool.imap(worker, index_pairs, chunksize=chunksize))
    # we compute the mean NVI, using the fact that NVI is a metric
    nvi_mean = 2 * np.sum(nvi_off_diagonal) / n_partitions**2
    # append mean NVI to results
    all_results["NVI"].append(nvi_mean)


def evaluate_NVI(index_pair, partitions):
    r"""Evaluations of Normalized Variation of Information (NVI).

    NVI is defined for two partitions :math:`p1` and :math:`p2` as:

    .. math::

        NVI = \frac{E(p1) + E(p2) - 2MI(p1, p2)}{JE(p1,p2)}

    where :math:`E` is the entropy, :math:`JE` the joint entropy
    and :math:`MI` the mutual information.

    Args:
        index_pair (list): list of two indices to select pairs of partitions
        partitions (list): list of partitions

    Returns:
        float, Normalized Variation Information
    """
    MI = mutual_info_score(partitions[index_pair[0]], partitions[index_pair[1]])
    Ex = entropy(partitions[index_pair[0]])
    Ey = entropy(partitions[index_pair[1]])
    JE = Ex + Ey - MI
    if abs(JE) < 1e-8:
        return 0.0
    return (JE - MI) / JE


def _to_indices(matrix, directed=False):
    """Convert a sparse matrix to indices and values.

    Args:
        matrix (sparse): sparse matrix to convert
        directed (bool): used for Leiden, which works if graph is full
    """
    if not directed:
        matrix = sp.tril(matrix)
    rows, cols, values = sp.find(matrix)
    return (rows, cols), values


@_timing
def _optimise(_, quality_indices, quality_values, null_model, global_shift, method="louvain"):
    """Worker for generalized Markov Stability optimisation runs."""
    if method == "louvain":
        stability, community_id = generalized_louvain.run_louvain(
            quality_indices[0],
            quality_indices[1],
            quality_values,
            len(quality_values),
            null_model,
            np.shape(null_model)[0],
            1.0,
        )

    if method == "leiden":
        # this implementation uses the trick suggested by V. Traag here:
        # https://github.com/vtraag/leidenalg/pull/109#issuecomment-1283963065
        G = ig.Graph(edges=zip(*quality_indices), directed=True)

        partitions = []
        n_null = int(len(null_model) / 2)
        for null in null_model[::2]:
            partitions.append(
                leidenalg.CPMVertexPartition(
                    G,
                    weights=quality_values,
                    node_sizes=null.tolist(),
                    correct_self_loops=True,
                )
            )
        optimiser = leidenalg.Optimiser()
        optimiser.set_rng_seed(np.random.randint(1e8))
        # we initialise stability
        stability = sum(partition.quality() for partition in partitions) / n_null
        # we use Leiden to find optimal partition and update stability according to improvement
        stability += optimiser.optimise_partition_multiplex(
            partitions, layer_weights=n_null * [1.0 / n_null]
        )
        community_id = partitions[0].membership

    return stability + global_shift, community_id


def _evaluate_quality(
    partition_id,
    quality_indices,
    quality_values,
    null_model,
    global_shift,
    method="louvain",
):
    """Worker for generalized Markov Stability evaluations."""
    # evaluate using Louvain method
    if method == "louvain":
        quality = generalized_louvain.evaluate_quality(
            quality_indices[0],
            quality_indices[1],
            quality_values,
            len(quality_values),
            null_model,
            np.shape(null_model)[0],
            1.0,
            partition_id,
        )

    # evaluate using Leiden method
    if method == "leiden":
        n_null = int(len(null_model) / 2)
        quality = np.sum(
            [
                leidenalg.CPMVertexPartition(
                    ig.Graph(edges=zip(*quality_indices), directed=True),
                    initial_membership=partition_id,
                    weights=quality_values,
                    node_sizes=null.tolist(),
                    correct_self_loops=True,
                ).quality()
                for null in null_model[::2]
            ]
        )
        quality /= n_null

    return quality + global_shift


def _run_optimisations(constructor, n_runs, pool, method="louvain"):
    """Run several generalized Markov Stability optimisation on the current quality matrix."""
    quality_indices, quality_values = _to_indices(
        constructor["quality"], directed=method == "leiden"
    )
    worker = partial(
        _optimise,
        quality_indices=quality_indices,
        quality_values=quality_values,
        null_model=constructor["null_model"],
        global_shift=constructor.get("shift", 0.0),
        method=method,
    )

    chunksize = _get_chunksize(n_runs, pool)
    return pool.map(worker, range(n_runs), chunksize=chunksize)


@_timing
def _compute_ttprime(all_results, pool):
    """Compute NVI(t,t') from the Markov stability results."""
    # prepare worker to compute NVI between selected partitions
    worker = partial(evaluate_NVI, partitions=all_results["community_id"])
    # we compute NVI only for t < t' because NVI is a metric
    index_pairs = list(itertools.combinations(range(len(all_results["scales"])), 2))
    chunksize = _get_chunksize(len(index_pairs), pool)
    # compute NVI(t,t') for t < t'
    ttprime_list = pool.map(worker, index_pairs, chunksize=chunksize)
    # store NVI(t,'t) as symmetric matrix with zero diagonal
    all_results["ttprime"] = np.zeros([len(all_results["scales"]), len(all_results["scales"])])
    for i, ttp in enumerate(ttprime_list):
        all_results["ttprime"][index_pairs[i][0], index_pairs[i][1]] = ttp
    all_results["ttprime"] += all_results["ttprime"].T


@_timing
def _apply_postprocessing(all_results, pool, constructors, tqdm_disable=False, method="louvain"):
    """Apply postprocessing."""
    all_results_raw = all_results.copy()

    # iterate through all scales
    for i, constructor in tqdm(
        enumerate(constructors), total=len(constructors), disable=tqdm_disable
    ):
        quality_indices, quality_values = _to_indices(
            constructor["quality"], directed=method == "leiden"
        )

        # prepare _evaluate_quality() function for parallel processing
        worker = partial(
            _evaluate_quality,
            quality_indices=quality_indices,
            quality_values=quality_values,
            null_model=constructor["null_model"],
            global_shift=constructor.get("shift", 0.0),
            method=method,
        )

        # find index in sequence of partitions that leads to highest quality score
        quality_scores = pool.map(
            worker,
            all_results_raw["community_id"],
            chunksize=_get_chunksize(len(all_results_raw["community_id"]), pool),
        )
        best_quality_id = np.argmax(quality_scores)

        # replace old partition with new partition
        all_results["community_id"][i] = all_results_raw["community_id"][best_quality_id]
        # assign new quality score
        all_results["stability"][i] = quality_scores[best_quality_id]
        # update number of communities
        all_results["number_of_communities"][i] = all_results_raw["number_of_communities"][
            best_quality_id
        ]
