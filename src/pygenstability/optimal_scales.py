"""Detect optimal scales from a scale scan."""
import logging

import numpy as np
import pandas as pd

L = logging.getLogger(__name__)


def identify_optimal_scales(results, NVI_cutoff=0.1, window_size=2):
    """Identifies optimal scales in Markov Stability.

    Stable scales are found from the NVI(t, t') matrix by searching for large diagonal
    blocks of NVI below NVI_cutoff. A moving average of window size is then applied to smooth the
    values accros scales, and a criterion is computed as the norm between this value and a similarly
    smoothed version of the NVI(t). Optimal scales are then detected as the local minima of this
    criterion.

    Args:
        results (dict): the results from a Markov Stability calculation
        NVI_cutoff (float): cut-off parameter for identifying plateau
        window_size (int): size of window for moving mean, to smooth the criterion curve

    Returns:
        result dictionary with two new keys: 'selected_partitions' and 'optimal_scale_criterion'
    """
    # get diagonals of ttprime matrix
    ttprime = results["ttprime"]
    ttprime_diagonals = []
    n = len(ttprime)
    ttprime_flipped = np.rot90(ttprime)
    for i in range(1, 2 * n, 2):
        ttprime_diagonals.append(np.diagonal(ttprime_flipped, offset=i - n, axis1=0))

    # ttprime_metric is size of diagonal plateau using NVI_cutoff
    ttprime_metric = np.asarray([np.sum(diagonal < NVI_cutoff) for diagonal in ttprime_diagonals])

    # apply moving mean of given window size and normalise
    ttprime_metric = pd.Series(ttprime_metric)
    ttprime_metric = np.roll(
        np.asarray(ttprime_metric.rolling(window=window_size, win_type="triang").mean()),
        -int(window_size / 2),
    )
    ttprime_metric = 1 - ttprime_metric / np.max(np.nan_to_num(ttprime_metric))
    ttprime_metric = ttprime_metric / np.max(np.nan_to_num(ttprime_metric))

    # nvi_metric is moving mean of NVI(t)
    nvi_metric = pd.Series(results["NVI"])
    nvi_metric = np.roll(
        np.asarray(nvi_metric.rolling(window=window_size, win_type="triang").mean()),
        -int(window_size / 2),
    )
    nvi_metric = nvi_metric / np.max(np.nan_to_num(nvi_metric))

    # compute final criterion and normalise
    criterion = np.sqrt((ttprime_metric**2 + nvi_metric**2) / 2)
    results["optimal_scale_criterion"] = criterion / np.max(np.nan_to_num(criterion))

    # selected scales are local minima of criterion computed via gradient
    criterion_gradient = np.gradient(criterion)
    selected_partitions = []
    for i in range(len(criterion_gradient) - 1):
        if np.sign(criterion_gradient)[i] == -1 and np.sign(criterion_gradient)[i + 1] == 1:
            selected_partitions.append(i)
        elif i < len(criterion_gradient) - 2:
            if (
                np.sign(criterion_gradient)[i] == -1
                and np.sign(criterion_gradient)[i + 1] == 0
                and np.sign(criterion_gradient)[i + 2] == 1
            ):
                selected_partitions.append(i + 1)

    # return with results dict
    results["selected_partitions"] = selected_partitions

    return results
