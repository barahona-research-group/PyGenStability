"""Detect optimal scales from a time scan."""
import logging

import numpy as np
from skimage.feature import peak_local_max

L = logging.getLogger("contrib.optimal_scales")


def identify_optimal_scales(results, VI_cutoff=0.1, criterion_threshold=0.8, window_size=2):
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
    window = np.ones(window_size) / window_size

    # compute ttprime criterion
    _flip = np.flipud(results["ttprime"])
    _n = results["ttprime"].shape[0]
    plateau_size = [np.sum(np.diag(_flip, k=shift) < VI_cutoff) for shift in range(-_n + 1, _n, 2)]
    plateau_moving_average = np.convolve(plateau_size, window, "same")
    ttprime_metric = 1.0 - plateau_moving_average / plateau_moving_average.max()
    ttprime_metric = ttprime_metric / ttprime_metric.max()

    # compute normalised VI criterion
    nvi_moving_average = np.convolve(results["variation_information"], window, "same")
    vi_metric = nvi_moving_average / nvi_moving_average.max()

    # compute final criterion
    criterion = np.sqrt(ttprime_metric**2 + vi_metric**2)
    results["optimal_scale_criterion"] = criterion / criterion.max()

    # return with results dict
    results["selected_partitions"] = sorted(
        peak_local_max(1.0 - criterion, min_distance=2, threshold_abs=criterion_threshold).flatten()
    )
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
    print(np.shape(times), np.shape(all_results["optimal_scale_criterion"]))
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
