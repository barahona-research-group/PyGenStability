"""plotting functions"""
import logging
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

L = logging.getLogger("pygenstability")


def plot_scan(
    all_results, time_axis=True, figure_name="scan_results.svg", use_plotly=True
):
    """Plot results of pygenstability with matplotlib or plotly"""

    if len(all_results["times"]) == 1:
        L.info(
            "Cannot plot the results if only one time point, we display the result instead:"
        )
        L.info(all_results)
        return

    if use_plotly:
        try:
            plot_scan_plotly(all_results)
        except ImportError:
            L.warning(
                "Plotly is not installed, please install package with \
                 pip install pygenstabiliy[plotly], using matplotlib instead."
            )

    plot_scan_plt(all_results, time_axis=time_axis, figure_name=figure_name)


def plot_scan_plotly(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    all_results,
):
    """Plot results of pygenstability with plotly"""
    from plotly.subplots import make_subplots  # pylint: disable=import-outside-toplevel
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    if all_results["params"]["log_time"]:
        times = np.log10(all_results["times"])
    else:
        times = all_results["times"]

    hovertemplate = str(
        "<b>Time</b>: %{x:.2f}"
        + "<br><i>Number of communities</i>: %{y}"
        + "<br>%{text}<extra></extra>"
    )

    if "mutual_information" in all_results:
        mi_data = all_results["mutual_information"]
        mi_opacity = 1.0
        mi_title = "Mutual information"
        mi_ticks = True
    else:
        mi_data = np.zeros(len(times))
        mi_opacity = 0.0
        mi_title = None
        mi_ticks = False

    text = [
        "Stability: {0:.3f}, <br> Mutual Information: {1:.3f}, <br> Index: {2}".format(
            s, mi, i
        )
        for s, mi, i in zip(
            all_results["stability"], mi_data, np.arange(0, len(times)),
        )
    ]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    ncom = go.Scatter(
        x=times,
        y=all_results["number_of_communities"],
        mode="lines+markers",
        hovertemplate=hovertemplate,
        name="Number of communities",
        xaxis="x2",
        yaxis="y4",
        text=text,
        marker_color="red",
    )

    if "ttprime" in all_results:
        z = all_results["ttprime"]
        showscale = True
        tprime_title = "tprime"
    else:
        z = np.nan + np.zeros([len(times), len(times)])
        showscale = False
        tprime_title = None

    ttprime = go.Heatmap(
        z=z,
        x=times,
        y=times,
        colorscale="YlOrBr",
        yaxis="y2",
        xaxis="x2",
        hoverinfo="skip",
        colorbar=dict(title="ttprime MI", len=0.2, yanchor="middle", y=0.5,),
        showscale=showscale,
    )

    stab = go.Scatter(
        x=times,
        y=all_results["stability"],
        mode="lines+markers",
        hovertemplate=hovertemplate,
        text=text,
        name="Stability",
        marker_color="blue",
    )

    mi = go.Scatter(
        x=times,
        y=mi_data,
        mode="lines+markers",
        hovertemplate=hovertemplate,
        text=text,
        name="Mutual information",
        yaxis="y3",
        xaxis="x",
        marker_color="green",
        opacity=mi_opacity,
    )

    layout = go.Layout(
        yaxis=dict(
            title="Stability",
            titlefont=dict(color="blue",),
            tickfont=dict(color="blue",),
            domain=[0, 0.28],
        ),
        yaxis2=dict(
            title=tprime_title,
            titlefont=dict(color="black",),
            tickfont=dict(color="black",),
            domain=[0.32, 1],
            side="right",
            range=[times[0], times[-1],],
        ),
        yaxis3=dict(
            title=mi_title,
            titlefont=dict(color="green",),
            tickfont=dict(color="green",),
            showticklabels=mi_ticks,
            overlaying="y",
            side="right",
        ),
        yaxis4=dict(
            title="Number of communities",
            titlefont=dict(color="red",),
            tickfont=dict(color="red",),
            overlaying="y2",
        ),
        xaxis=dict(range=[times[0], times[-1],],),
        xaxis2=dict(range=[times[0], times[-1],],),
    )

    fig = go.Figure(data=[stab, ncom, mi, ttprime], layout=layout)
    fig.show()


def plot_communities(
    graph, all_results, folder="communities", edge_color="0.5", edge_width=0.5
):
    """now plot the community structures at each time in a folder"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    pos = [graph.nodes[u]["pos"] for u in graph]

    mpl_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    for i in tqdm(range(len(all_results["times"]))):
        node_color = all_results["community_id"][i]

        plt.figure()
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            node_color=node_color,
            node_size=100,
            cmap=plt.get_cmap("tab20"),
        )
        nx.draw_networkx_edges(graph, pos=pos, width=edge_width, edge_color=edge_color)

        plt.axis("off")
        plt.title(
            str(r"$log_{10}(time) =$ ")
            + str(np.round(np.log10(all_results["times"][i]), 2))
            + ", with "
            + str(all_results["number_of_communities"][i])
            + " communities"
        )

        plt.savefig(
            os.path.join(folder, "time_" + str(i) + ".png"), bbox_inches="tight"
        )
        plt.close()
    matplotlib.use(mpl_backend)


def _get_times(all_results, time_axis=True):
    if all_results["params"]["log_time"]:
        return np.log10(all_results["times"])
    if time_axis:
        return np.arange(len(all_results["times"]))
    return all_results["times"]


def plot_number_comm(all_results, ax, time_axis=True):
    """Plot number of communities."""
    times = _get_times(all_results, time_axis)

    ax.plot(
        times, all_results["number_of_communities"], "-", c="C3", label="size", lw=2.0
    )
    ax.set_ylabel("Number of clusters", color="C3")
    ax.tick_params("y", colors="C3")


def plot_ttprime(all_results, ax):
    """Plot ttprime."""
    times = _get_times(all_results)

    ax.contourf(times, times, all_results["ttprime"], cmap="YlOrBr")
    ax.set_ylabel(r"$log_{10}(t^\prime)$")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.axis([times[0], times[-1], times[0], times[-1]])


def plot_mutual_information(all_results, ax, time_axis=True):
    """Plot mutual information."""
    times = _get_times(all_results, time_axis=time_axis)
    ax.plot(times, all_results["mutual_information"], "-", lw=2.0, c="C2", label="MI")

    ax.yaxis.tick_right()
    ax.tick_params("y", colors="C2")
    ax.set_ylabel(r"Mutual information", color="C2")
    ax.axhline(1, ls="--", lw=1.0, c="C2")
    ax.axis([times[0], times[-1], np.min(all_results["mutual_information"]) * 0.9, 1.1])


def plot_stability(all_results, ax, time_axis=True):
    """Plot stability."""
    times = _get_times(all_results, time_axis=time_axis)
    ax.plot(times, all_results["stability"], "-", label=r"$Q$", c="C0")

    ax.tick_params("y", colors="C0")
    ax.set_ylabel("Stability", color="C0")
    ax.yaxis.set_label_position("left")
    ax.set_xlabel(r"$log_{10}(t)$")


def plot_scan_plt(all_results, time_axis=True, figure_name="scan_results.svg"):
    """Plot results of pygenstability with matplotlib."""
    plt.figure(figsize=(5, 5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.5])
    gs.update(hspace=0)

    if "ttprime" in all_results:
        ax0 = plt.subplot(gs[0, 0])
        plot_ttprime(all_results, ax=ax0)
        ax1 = ax0.twinx()
    else:
        ax1 = plt.subplot(gs[0, 0])

    ax1.set_xticks([])

    plot_number_comm(all_results, ax=ax1, time_axis=time_axis)

    if "ttprime" in all_results:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

    ax2 = plt.subplot(gs[1, 0])

    if "stability" in all_results:
        plot_stability(all_results, ax=ax2, time_axis=time_axis)

    if "mutual_information" in all_results:
        ax3 = ax2.twinx()
        plot_mutual_information(all_results, ax=ax3, time_axis=time_axis)

    plt.savefig(figure_name, bbox_inches="tight")
