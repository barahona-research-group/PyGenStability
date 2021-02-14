"""Plotting functions."""
import logging
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
from tqdm import tqdm

L = logging.getLogger(__name__)

# pylint: disable=import-outside-toplevel


def plot_scan(
    all_results,
    time_axis=True,
    figure_name="scan_results.pdf",
    use_plotly=False,
    live=True,
    plotly_filename="scan_results.html",
):
    """Plot results of pygenstability with matplotlib or plotly.

    Args:
        all_results (dict): results of pygenstability scan
        time_axis (bool): display time of time index on time axis
        figure_name (str): name of matplotlib figure
        use_plotly (bool): use matplotlib or plotly backend
        live (bool): for plotly backend, open browser with pot
        plotly_filename (str): filename of .html figure from plotly
    """
    if len(all_results["times"]) == 1:
        L.info("Cannot plot the results if only one time point, we display the result instead:")
        L.info(all_results)
        return

    if use_plotly:
        try:
            _plot_scan_plotly(all_results, live=live, filename=plotly_filename)
        except ImportError:
            L.warning(
                "Plotly is not installed, please install package with \
                 pip install pygenstabiliy[plotly], using matplotlib instead."
            )
    else:
        _plot_scan_plt(all_results, time_axis=time_axis, figure_name=figure_name)


def _plot_scan_plotly(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    all_results,
    live=False,
    filename="clusters.html",
):
    """Plot results of pygenstability with plotly."""
    import plotly.graph_objects as go
    from plotly.offline import plot as _plot

    if all_results["run_params"]["log_time"]:
        times = np.log10(all_results["times"])
    else:
        times = all_results["times"]

    hovertemplate = str("<b>Time</b>: %{x:.2f}, <br>%{text}<extra></extra>")

    if "variation_information" in all_results:
        vi_data = all_results["variation_information"]
        vi_opacity = 1.0
        vi_title = "Variation of information"
        vi_ticks = True
    else:
        vi_data = np.zeros(len(times))
        vi_opacity = 0.0
        vi_title = None
        vi_ticks = False

    text = [
        f"""Number of communities: {n}, <br> Stability: {np.round(s, 3)},
        <br> Variation Information: {np.round(vi, 3)}, <br> Index: {i}"""
        for n, s, vi, i in zip(
            all_results["number_of_communities"],
            all_results["stability"],
            vi_data,
            np.arange(0, len(times)),
        )
    ]

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
        colorbar=dict(
            title="ttprime VI",
            len=0.2,
            yanchor="middle",
            y=0.5,
        ),
        showscale=showscale,
    )
    if "stability" in all_results:
        stab = go.Scatter(
            x=times,
            y=all_results["stability"],
            mode="lines+markers",
            hovertemplate=hovertemplate,
            text=text,
            name="Stability",
            marker_color="blue",
        )

    vi = go.Scatter(
        x=times,
        y=vi_data,
        mode="lines+markers",
        hovertemplate=hovertemplate,
        text=text,
        name="Variation information",
        yaxis="y3",
        xaxis="x",
        marker_color="green",
        opacity=vi_opacity,
    )

    layout = go.Layout(
        yaxis=dict(
            title="Stability",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            domain=[0.0, 0.28],
        ),
        yaxis2=dict(
            title=tprime_title,
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            domain=[0.32, 1],
            side="right",
            range=[times[0], times[-1]],
        ),
        yaxis3=dict(
            title=vi_title,
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            showticklabels=vi_ticks,
            overlaying="y",
            side="right",
        ),
        yaxis4=dict(
            title="Number of communities",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y2",
        ),
        xaxis=dict(range=[times[0], times[-1]]),
        xaxis2=dict(range=[times[0], times[-1]]),
        height=600,
        width=800,
    )

    fig = go.Figure(data=[stab, ncom, vi, ttprime], layout=layout)
    _plot(fig, filename=filename)

    if live:
        fig.show()


def plot_single_community(
    graph, all_results, time_id, edge_color="0.5", edge_width=0.5, node_size=100
):
    """Plot the community structures for a given time.

    Args:
        graph (networkx.Graph): graph to plot
        all_results (dict): results of pygenstability scan
        time_id (int): index of time to plot
        folder (str): folder to save figures
        edge_color (str): color of edges
        edge_width (float): width of edges
        node_size (float): size of nodes
        ext (str): extension of figures files
    """
    pos = {u: graph.nodes[u]["pos"] for u in graph}

    node_color = all_results["community_id"][time_id]

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        cmap=plt.get_cmap("tab20"),
    )
    nx.draw_networkx_edges(graph, pos=pos, width=edge_width, edge_color=edge_color)

    plt.axis("off")
    plt.title(
        str(r"$log_{10}(time) =$ ")
        + str(np.round(np.log10(all_results["times"][time_id]), 2))
        + ", with "
        + str(all_results["number_of_communities"][time_id])
        + " communities"
    )


def plot_communities(
    graph, all_results, folder="communities", edge_color="0.5", edge_width=0.5, ext=".pdf"
):
    """Plot the community structures at each time in a folder.

    Args:
        graph (networkx.Graph): graph to plot
        all_results (dict): results of pygenstability scan
        folder (str): folder to save figures
        edge_color (str): color of edges
        edge_width (float): width of edgs
        ext (str): extension of figures files
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    mpl_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    for time_id in tqdm(range(len(all_results["times"]))):
        plt.figure()
        plot_single_community(
            graph, all_results, time_id, edge_color=edge_color, edge_width=edge_width
        )
        plt.savefig(os.path.join(folder, "time_" + str(time_id) + ext), bbox_inches="tight")
        plt.close()
    matplotlib.use(mpl_backend)


def _get_times(all_results, time_axis=True):
    """Get the time vector."""
    if not time_axis:
        return np.arange(len(all_results["times"]))
    if all_results["run_params"]["log_time"]:
        return np.log10(all_results["times"])
    return all_results["times"]


def _plot_number_comm(all_results, ax, time_axis=True):
    """Plot number of communities."""
    times = _get_times(all_results, time_axis)

    ax.plot(times, all_results["number_of_communities"], "-", c="C3", label="size", lw=2.0)
    ax.set_ylabel("Number of clusters", color="C3")
    ax.tick_params("y", colors="C3")


def _plot_ttprime(all_results, ax, time_axis):
    """Plot ttprime."""
    times = _get_times(all_results, time_axis)

    ax.contourf(times, times, all_results["ttprime"], cmap="YlOrBr_r")
    ax.set_ylabel(r"$log_{10}(t^\prime)$")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.axis([times[0], times[-1], times[0], times[-1]])


def _plot_variation_information(all_results, ax, time_axis=True):
    """Plot variation information."""
    times = _get_times(all_results, time_axis=time_axis)
    ax.plot(times, all_results["variation_information"], "-", lw=2.0, c="C2", label="VI")

    ax.yaxis.tick_right()
    ax.tick_params("y", colors="C2")
    ax.set_ylabel(r"Variation information", color="C2")
    ax.axhline(1, ls="--", lw=1.0, c="C2")
    ax.axis([times[0], times[-1], 0.0, np.max(all_results["variation_information"]) * 1.1])


def _plot_stability(all_results, ax, time_axis=True):
    """Plot stability."""
    times = _get_times(all_results, time_axis=time_axis)
    ax.plot(times, all_results["stability"], "-", label=r"$Q$", c="C0")
    ax.tick_params("y", colors="C0")
    ax.set_ylabel("Stability", color="C0")
    ax.yaxis.set_label_position("left")
    ax.set_xlabel(r"$log_{10}(t)$")


def _plot_scan_plt(all_results, time_axis=True, figure_name="scan_results.svg"):
    """Plot results of pygenstability with matplotlib."""
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.5])
    gs.update(hspace=0)
    if "ttprime" in all_results:
        ax0 = plt.subplot(gs[0, 0])
        _plot_ttprime(all_results, ax=ax0, time_axis=time_axis)
        ax1 = ax0.twinx()
    else:
        ax1 = plt.subplot(gs[0, 0])

    ax1.set_xticks([])

    _plot_number_comm(all_results, ax=ax1, time_axis=time_axis)
    if "ttprime" in all_results:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

    ax2 = plt.subplot(gs[1, 0])

    if "stability" in all_results:
        _plot_stability(all_results, ax=ax2, time_axis=time_axis)

    if "variation_information" in all_results:
        ax3 = ax2.twinx()
        _plot_variation_information(all_results, ax=ax3, time_axis=time_axis)

    plt.savefig(figure_name, bbox_inches="tight")


def plot_clustered_adjacency(
    adjacency,
    all_results,
    time,
    labels=None,
    figsize=(12, 10),
    cmap="Blues",
    figure_name="clustered_adjacency.pdf",
):
    """Plot the clustered adjacency matrix of the graph at a given time.

    Args:
        adjacency (ndarray): adjacency matrix to plot
        all_results (dict): results of PyGenStability
        time (int): time index for clustering
        labels (list): node labels, or None
        figsize (tubple): figure size
        cmap (str): colormap for matrix elements
        figure_name (str): filename of the figure with extension
    """
    comms, counts = np.unique(all_results["community_id"][time], return_counts=True)

    node_ids = []
    for comm in comms:
        node_ids += list(np.where(all_results["community_id"][time] == comm)[0])

    adjacency = adjacency[np.ix_(node_ids, node_ids)]
    adjacency[adjacency == 0] = np.nan

    plt.figure(figsize=figsize)
    plt.imshow(adjacency, aspect="auto", origin="auto", cmap=cmap)

    ax = plt.gca()

    pos = 0
    for comm, count in zip(comms, counts):
        rect = patches.Rectangle(
            (pos - 0.5, pos - 0.5),
            count,
            count,
            linewidth=5,
            facecolor="none",
            edgecolor="g",
        )
        ax.add_patch(rect)
        pos += count

    ax.set_xticks(np.arange(len(adjacency)))
    ax.set_yticks(np.arange(len(adjacency)))

    if labels is not None:
        labels_plot = [labels[i] for i in node_ids]
        ax.set_xticklabels(labels_plot)
        ax.set_yticklabels(labels_plot)

    plt.colorbar()
    plt.xticks(rotation=90)
    plt.axis([-0.5, len(adjacency) - 0.5, -0.5, len(adjacency) - 0.5])
    plt.suptitle(
        "log10(time) = "
        + str(np.round(np.log10(all_results["times"][time]), 2))
        + ",  number_of_communities="
        + str(all_results["number_of_communities"][time])
    )

    plt.savefig(figure_name, bbox_inches="tight")


def plot_sankey(all_results, live=False, filename="communities_sankey.html", time_index=None):
    """Plot Sankey diagram of communities accros time (plotly only).

    Args:
        all_results (dict): results from run function
        live (bool): if True, interactive figure will appear in browser
        filename (str): filename to save the plot
        time_index (bool): plot time of indices
    """
    import plotly.graph_objects as go
    from plotly.offline import plot as _plot

    sources = []
    targets = []
    values = []
    shift = 0

    if not time_index:
        all_results["community_id_reduced"] = all_results["community_id"]
    else:
        all_results["community_id_reduced"] = [all_results["community_id"][i] for i in time_index]

    for i in range(len(all_results["community_id_reduced"]) - 1):
        community_source = np.array(all_results["community_id_reduced"][i])
        community_target = np.array(all_results["community_id_reduced"][i + 1])
        source_ids = set(community_source)
        target_ids = set(community_target)
        for source in source_ids:
            for target in target_ids:
                value = sum(community_target[community_source == source] == target)
                if value > 0:
                    values.append(value)
                    sources.append(source + shift)
                    targets.append(target + len(source_ids) + shift)
        shift += len(source_ids)

    layout = go.Layout(autosize=True)
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=1,
                    thickness=1,
                    line=dict(color="black", width=0.0),
                ),
                link=dict(source=sources, target=targets, value=values),
            )
        ],
        layout=layout,
    )

    _plot(fig, filename=filename)

    if live:
        fig.show()
