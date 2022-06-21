"""Detect optimal scales from a time scan."""
import logging

import numpy as np
# from skimage.feature import peak_local_max
import pandas as pd
from scipy.misc import derivative

L = logging.getLogger("contrib.optimal_scales")


def identify_optimal_scales(results, NVI_cutoff=0.1, window_size=2):
    """Identifies optimal scales in Markov Stability.

    Stable scales are found from the normalized VI(t, t') matrix by searching for large diagonal
    blocks of VI below VI_cutoff. A moving average of window size is then applied to smooth the
    values accros time, and a criterion is computed as the norm between this value and a similarly
    smoothed version of the normalized VI(t). Optimal scales are then detected using the peak
    detection algorithm skimage.peak_local_max, with minima under criterion_thresholds are selected.

    Args:
        results (dict): the results from a Markov Stability calculation
        VI_cutoff (float): cut-off parameter for identifying plateau
        criterion_threshold (float): maximum value of criterion to be a valid scale
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
    nvi_metric = pd.Series(results["variation_information"])
    nvi_metric = np.roll(
        np.asarray(nvi_metric.rolling(window=window_size, win_type="triang").mean()),
        -int(window_size / 2),
    )
    nvi_metric = nvi_metric / np.max(np.nan_to_num(nvi_metric))

    # compute final criterion and normalise
    criterion = np.sqrt((ttprime_metric**2 + nvi_metric**2) / 2)
    criterion = criterion / np.max(np.nan_to_num(criterion))
    results["optimal_scale_criterion"] = criterion

    # selected scales are local minima of criterion
    criterion_gradient = np.gradient(criterion)
    selected_partitions = []
    for i in range(len(criterion_gradient) - 1):
        if np.sign(criterion_gradient)[i] == -1 and np.sign(criterion_gradient)[i + 1] == 1:
            selected_partitions.append(i)
    
    # return with results dict
    results["selected_partitions"] = selected_partitions

    return results


def plot_optimal_scales(
    results,
    time_axis=True,
    figure_name="scan_results.pdf",
    use_plotly=False,
    live=True,
    plotly_filename="scan_results.html",
):
    """Plot scan results with optimal scales."""
    if len(results["times"]) == 1:
        L.info("Cannot plot the results if only one time point, we display the result instead:")
        L.info(results)
        return

    if use_plotly:
        try:
            plot_optimal_scales_plotly(results, live=live, filename=plotly_filename)
        except ImportError:
            L.warning(
                "Plotly is not installed, please install package with \
                 pip install pygenstabiliy[plotly], using matplotlib instead."
            )

    else:
        plot_optimal_scales_plt(results, time_axis=time_axis, figure_name=figure_name)


def plot_optimal_scales_plotly(results, live=False, filename="scan_results.pdf"):
    """Plot optimal scales on plotly."""
    from plotly.offline import plot as _plot

    from pygenstability.plotting import get_times
    from pygenstability.plotting import plot_scan_plotly

    fig, _ = plot_scan_plotly(results, live=False, filename=None)

    times = get_times(results, time_axis=True)

    fig.add_scatter(
        x=times,
        y=results["optimal_scale_criterion"],
        mode="lines+markers",
        name="Optimal Scale Criterion",
        yaxis="y5",
        xaxis="x",
        marker_color="orange",
    )

    fig.add_scatter(
        x=times[results["selected_partitions"]],
        y=results["optimal_scale_criterion"][results["selected_partitions"]],
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


def plot_optimal_scales_plt(results, time_axis=True, figure_name="scan_results.pdf"):
    """Plot scan results with optimal scales with matplotlib."""
    import matplotlib.pyplot as plt

    from pygenstability.plotting import get_times
    from pygenstability.plotting import plot_scan_plt

    ax2 = plot_scan_plt(results, time_axis=time_axis, figure_name=None)[2]

    times = get_times(results, time_axis=time_axis)

    ax2.plot(
        times,
        results["optimal_scale_criterion"],
        "-",
        lw=2.0,
        c="C4",
        label="optimal scale criterion",
    )
    ax2.plot(
        times[results["selected_partitions"]],
        results["optimal_scale_criterion"][results["selected_partitions"]],
        "o",
        lw=2.0,
        c="C4",
        label="optimal scales",
    )

    ax2.set_ylabel(r"Stability, Optimal scales", color="k")
    ax2.legend()
    if figure_name is not None:
        plt.savefig(figure_name, bbox_inches="tight")