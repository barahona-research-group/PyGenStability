"""Detect optimal scales from a time scan."""
import numpy as np


def identify_optimal_scales(results, window=2, beta=0.1):
    """Identifies optimal scales in Markov Stability.

    Args:
        results (dict): the results from a Markov Stability calculation
        window (int): size of window for moving average
        beta (float): cut-off parameter for identifying plateau
    """
    # extract ttprime and flip to identify diagonals
    ttprime_ = np.flipud(results["ttprime"])
    n_ = ttprime_.shape[0]

    # extract diagonals in lower triangular and identify plateaus
    plateau_size = np.zeros(n_)
    for i, shift in enumerate(range(-n_ + 1, n_, 2)):
        diagonal = np.diag(ttprime_, k=shift)
        plateau_size[i] = np.sum(diagonal < beta)

    # compute normalised ttprime
    plateau_moving_average = np.convolve(plateau_size, np.ones(window), "valid") / window
    plateau_moving_average_norm = 1 - (plateau_moving_average / plateau_moving_average.max())
    ttprime_metric = plateau_moving_average_norm / plateau_moving_average_norm.max()
    ttprime_metric = np.append(ttprime_metric, 0)

    # compute normalised VI
    nvi_moving_average = (
        np.convolve(results["variation_information"], np.ones(window), "valid") / window
    )
    vi_metric = nvi_moving_average / nvi_moving_average.max()
    vi_metric = np.append(vi_metric, 0)

    # define criterion
    criterion = np.sqrt((ttprime_metric ** 2 + vi_metric ** 2) / 2)
    criterion = criterion / criterion.max()

    # find gradient of criterion function
    criterion_gradient = np.gradient(criterion)

    # find minima in criterion
    index_minima = np.where(criterion_gradient[:-1] * criterion_gradient[1:] < 0)[0]
    index_minima = index_minima[criterion_gradient[index_minima] < 0]

    # return with results dict
    results["selected_partitions"] = index_minima
    results["optimal_scale_criterion"] = criterion

    return results
