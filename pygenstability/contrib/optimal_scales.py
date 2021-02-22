"""Detect optimal scales from a time scan."""
import logging
from copy import deepcopy

import numpy as np

L = logging.getLogger("contrib.optimal_scales")


def identify_optimal_scales(results, window=2, beta=0.1):
    """Identifies optimal scales in Markov Stability.

    Stable scales are found from the normalized VI(t, t') matrix by searching for large diagonal
    blocks of uniform low VI values. The parameter 'beta' is used as a threshold to define the
    blocks, and the 'windows' sets the  size of the moving mean windows.

    Args:
        results (dict): the results from a Markov Stability calculation
        window (int): size of window for moving mean
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
    all_results,
    time_axis=True,
    figure_name="scan_results.pdf",
    use_plotly=False,
    live=True,
    plotly_filename="scan_results.html",
):
    """Plot scan results with optimal scales."""
    if len(all_results["times"]) == 1:
        L.info("Cannot plot the results if only one time point, we display the result instead:")
        L.info(all_results)
        return

    if use_plotly:
        try:
            plot_optimal_scales_plotly(all_results, live=live, filename=plotly_filename)
        except ImportError:
            L.warning(
                "Plotly is not installed, please install package with \
                 pip install pygenstabiliy[plotly], using matplotlib instead."
            )

    else:
        plot_optimal_scales_plt(all_results, time_axis=time_axis, figure_name=figure_name)


def plot_optimal_scales_plotly(all_results, live=False, filename="scan_results.pdf"):
    """Plot optimal scales on plotly."""
    from plotly.offline import plot as _plot

    from pygenstability.plotting import get_times
    from pygenstability.plotting import plot_scan_plotly

    fig, _ = plot_scan_plotly(all_results, live=False, filename=None)

    times = get_times(all_results, time_axis=True)

    fig.add_scatter(
        x=times,
        y=all_results["optimal_scale_criterion"],
        mode="lines+markers",
        name="Optimal Scale Criterion",
        yaxis="y5",
        xaxis="x",
        marker_color="orange",
    )

    fig.add_scatter(
        x=times[all_results["selected_partitions"]],
        y=all_results["optimal_scale_criterion"][all_results["selected_partitions"]],
        mode="markers",
        name="Optimal Scale",
        yaxis="y5",
        xaxis="x",
        marker_color="red",
    )

    fig.update_layout(
        yaxis5=dict(
            titlefont=dict(color="orange"),
            tickfont=dict(color="orange"),
            domain=[0.0, 0.28],
            overlaying="y",
        )
    )
    fig.update_layout(yaxis=dict(title="Stability, Optimal Scale Criterion"))
    if filename is not None:
        _plot(fig, filename=filename)

    if live:
        fig.show()


def plot_optimal_scales_plt(all_results, time_axis=True, figure_name="scan_results.pdf"):
    """Plot scan results with optimal scales with matplotlib."""
    import matplotlib.pyplot as plt

    from pygenstability.plotting import get_times
    from pygenstability.plotting import plot_scan_plt

    _, _, ax2, _ = plot_scan_plt(all_results, time_axis=time_axis, figure_name=None)

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
