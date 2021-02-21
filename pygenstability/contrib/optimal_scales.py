"""Detect optimal scales from a time scan."""
import logging
from copy import deepcopy
import numpy as np


L = logging.getLogger("contrib.optimal_scales")


def identify_optimal_scales(results, window=2, beta=0.1):
    """Identifies optimal scales in Markov Stability.

    Args:
        results (dict): the results from a Markov Stability calculation
        window (int): size of window for moving average
        beta (float): cut-off parameter for identifying plateau

    Returns:
        result dictionary with two new keys: 'selected_partitions' and 'optimal_scale_criterion'
    """
    results = deepcopy(results)

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


def plot_optimal_scales(
    all_results, time_axis=True, figure_name="scan_results.pdf", use_plotly=False
):
    """Pot scan results witht optitmal scales."""
    if len(all_results["times"]) == 1:
        L.info("Cannot plot the results if only one time point, we display the result instead:")
        L.info(all_results)
        return

    if use_plotly:
        try:
            plot_optimal_scales_plotly(all_results)
        except ImportError:
            L.warning(
                "Plotly is not installed, please install package with \
                 pip install pygenstabiliy[plotly], using matplotlib instead."
            )

    plot_optimal_scales_plt(all_results, time_axis=time_axis, figure_name=figure_name)


def plot_optimal_scales_plotly(all_results, time_axis=True, figure_name="scan_results.pdf"):
    pass


def plot_optimal_scales_plt(all_results, time_axis=True, figure_name="scan_results.pdf"):
    """Pot scan results witht optitmal scales."""
    from pygenstability.plotting import plot_scan_plt, get_times
    import matplotlib.pyplot as plt

    ax0, ax1, ax2, ax3 = plot_scan_plt(all_results, time_axis=time_axis, figure_name=None)

    times = get_times(all_results, time_axis=time_axis)
    ax2.plot(
        times,
        all_results["optimal_scale_criterion"],
        "-",
        lw=2.0,
        c="C4",
        label="optimal scale criterion",
    )
    ax2.plot(
        times[all_results["selected_partitions"]],
        all_results["optimal_scale_criterion"][all_results["selected_partitions"]],
        "o",
        lw=2.0,
        c="C4",
        label="optimal scales",
    )

    ax2.set_ylabel(r"Stability, Optimal scales", color="k")
    ax2.legend()
    if figure_name is not None:
        plt.savefig(figure_name, bbox_inches="tight")
