"""Plotting functions."""

import logging
import os

import matplotlib
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    print('Please install networkx via pip install "pygenstability[networkx]" for full plotting.')

import numpy as np
from matplotlib import gridspec
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as _plot
except ImportError:  # pragma: no cover
    pass


from pygenstability.optimal_scales import identify_optimal_scales

L = logging.getLogger(__name__)


def plot_scan(
    all_results,
    figsize=(6, 5),
    scale_axis=True,
    figure_name="scan_results.pdf",
    use_plotly=False,
    live=True,
    plotly_filename="scan_results.html",
):
    """Plot results of pygenstability with matplotlib or plotly.

    Args:
        all_results (dict): results of pygenstability scan
        figsize (tuple): matplotlib figure size
        scale_axis (bool): display scale of scale index on scale axis
        figure_name (str): name of matplotlib figure
        use_plotly (bool): use matplotlib or plotly backend
        live (bool): for plotly backend, open browser with pot
        plotly_filename (str): filename of .html figure from plotly
    """
    if len(all_results["scales"]) == 1:  # pragma: no cover
        L.info("Cannot plot the results if only one scale point, we display the result instead:")
        L.info(all_results)
        return None

    if use_plotly:
        return plot_scan_plotly(all_results, live=live, filename=plotly_filename)
    return plot_scan_plt(
        all_results, figsize=figsize, scale_axis=scale_axis, figure_name=figure_name
    )


def plot_scan_plotly(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    all_results,
    live=False,
    filename="clusters.html",
):
    """Plot results of pygenstability with plotly."""
    scales = _get_scales(all_results, scale_axis=True)

    hovertemplate = str("<b>scale</b>: %{x:.2f}, <br>%{text}<extra></extra>")

    if "NVI" in all_results:
        nvi_data = all_results["NVI"]
        nvi_opacity = 1.0
        nvi_title = "Variation of information"
        nvi_ticks = True
    else:  # pragma: no cover
        nvi_data = np.zeros(len(scales))
        nvi_opacity = 0.0
        nvi_title = None
        nvi_ticks = False

    text = [
        f"""Number of communities: {n}, <br> Stability: {np.round(s, 3)},
        <br> Normalised Variation Information: {np.round(vi, 3)}, <br> Index: {i}"""
        for n, s, vi, i in zip(
            all_results["number_of_communities"],
            all_results["stability"],
            nvi_data,
            np.arange(0, len(scales)),
        )
    ]
    ncom = go.Scatter(
        x=scales,
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
        tprime_title = "log10(scale)"
    else:  # pragma: no cover
        z = np.nan + np.zeros([len(scales), len(scales)])
        showscale = False
        tprime_title = None

    ttprime = go.Heatmap(
        z=z,
        x=scales,
        y=scales,
        colorscale="YlOrBr_r",
        yaxis="y2",
        xaxis="x2",
        hoverinfo="skip",
        colorbar={"title": "VI", "len": 0.2, "yanchor": "middle", "y": 0.5},
        showscale=showscale,
    )
    if "stability" in all_results:
        stab = go.Scatter(
            x=scales,
            y=all_results["stability"],
            mode="lines+markers",
            hovertemplate=hovertemplate,
            text=text,
            name="Stability",
            marker_color="blue",
        )

    vi = go.Scatter(
        x=scales,
        y=nvi_data,
        mode="lines+markers",
        hovertemplate=hovertemplate,
        text=text,
        name="NVI",
        yaxis="y3",
        xaxis="x",
        marker_color="green",
        opacity=nvi_opacity,
    )

    layout = go.Layout(
        yaxis={
            "title": "Stability",
            "title_font": {"color": "blue"},
            "tickfont": {"color": "blue"},
            "domain": [0.0, 0.28],
        },
        yaxis2={
            "title": tprime_title,
            "title_font": {"color": "black"},
            "tickfont": {"color": "black"},
            "domain": [0.32, 1],
            "side": "right",
            "range": [scales[0], scales[-1]],
        },
        yaxis3={
            "title": nvi_title,
            "title_font": {"color": "green"},
            "tickfont": {"color": "green"},
            "showticklabels": nvi_ticks,
            "overlaying": "y",
            "side": "right",
        },
        yaxis4={
            "title": "Number of communities",
            "title_font": {"color": "red"},
            "tickfont": {"color": "red"},
            "overlaying": "y2",
        },
        xaxis={"range": [scales[0], scales[-1]]},
        xaxis2={"range": [scales[0], scales[-1]]},
    )

    fig = go.Figure(data=[stab, ncom, vi, ttprime], layout=layout)
    fig.update_layout(xaxis_title="log10(scale)")

    if filename is not None:
        _plot(fig, filename=filename, auto_open=live)

    return fig, layout


def plot_single_partition(
    graph, all_results, scale_id, edge_color="0.5", edge_width=0.5, node_size=100
):
    """Plot the community structures for a given scale.

    Args:
        graph (networkx.Graph): graph to plot
        all_results (dict): results of pygenstability scan
        scale_id (int): index of scale to plot
        folder (str): folder to save figures
        edge_color (str): color of edges
        edge_width (float): width of edges
        node_size (float): size of nodes
        ext (str): extension of figures files
    """
    if any("pos" not in graph.nodes[u] for u in graph):
        pos = nx.spring_layout(graph)
        for u in graph:
            graph.nodes[u]["pos"] = pos[u]

    pos = {u: graph.nodes[u]["pos"] for u in graph}

    node_color = all_results["community_id"][scale_id]

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
        str(r"$log_{10}(scale) =$ ")
        + str(np.round(np.log10(all_results["scales"][scale_id]), 2))
        + ", with "
        + str(all_results["number_of_communities"][scale_id])
        + " communities"
    )


def plot_optimal_partitions(
    graph,
    all_results,
    edge_color="0.5",
    edge_width=0.5,
    folder="optimal_partitions",
    ext=".pdf",
    show=False,
):
    """Plot the community structures at each optimal scale.

    Args:
        graph (networkx.Graph): graph to plot
        all_results (dict): results of pygenstability scan
        edge_color (str): color of edges
        edge_width (float): width of edgs
        folder (str): folder to save figures
        ext (str): extension of figures files
        show (bool): show each plot with plt.show() or not
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    if "selected_partitions" not in all_results:  # pragma: no cover
        identify_optimal_scales(all_results)

    selected_scales = all_results["selected_partitions"]
    n_selected_scales = len(selected_scales)

    if n_selected_scales == 0:  # pragma: no cover
        return

    for optimal_scale_id in selected_scales:
        plot_single_partition(
            graph,
            all_results,
            optimal_scale_id,
            edge_color=edge_color,
            edge_width=edge_width,
        )
        plt.savefig(f"{folder}/scale_{optimal_scale_id}{ext}", bbox_inches="tight")
        if show:  # pragma: no cover
            plt.show()


def plot_communities(
    graph,
    all_results,
    folder="communities",
    edge_color="0.5",
    edge_width=0.5,
    ext=".pdf",
):
    """Plot the community structures at each scale in a folder.

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
    for scale_id in tqdm(range(len(all_results["scales"]))):
        plt.figure()
        plot_single_partition(
            graph, all_results, scale_id, edge_color=edge_color, edge_width=edge_width
        )
        plt.savefig(os.path.join(folder, "scale_" + str(scale_id) + ext), bbox_inches="tight")
        plt.close()
    matplotlib.use(mpl_backend)


def plot_communities_matrix(graph, all_results, folder="communities_matrix", ext=".pdf"):
    """Plot communities at all scales in matrix form.

    Args:
        graph (array): as a numpy matrix
        all_results (dict): clustring results
        folder (str): folder to save figures
        ext (str): figure file format
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    for scale_id in tqdm(range(len(all_results["scales"]))):
        plt.figure()
        com_ids = all_results["community_id"][scale_id]
        ids = []
        lines = [0]
        for i in range(len(set(com_ids))):
            _ids = list(np.argwhere(com_ids == i).flatten())
            lines.append(len(_ids))
            ids += _ids
        plt.imshow(graph[ids][:, ids], origin="lower")
        lines = np.cumsum(lines)
        for i in range(len(lines) - 1):
            plt.plot((lines[i], lines[i + 1]), (lines[i], lines[i]), c="k")
            plt.plot((lines[i], lines[i]), (lines[i], lines[i + 1]), c="k")
            plt.plot((lines[i + 1], lines[i + 1]), (lines[i + 1], lines[i]), c="k")
            plt.plot((lines[i + 1], lines[i]), (lines[i + 1], lines[i + 1]), c="k")

        plt.savefig(os.path.join(folder, "scale_" + str(scale_id) + ext), bbox_inches="tight")


def _get_scales(all_results, scale_axis=True):
    """Get the scale vector."""
    if not scale_axis:  # pragma: no cover
        return np.arange(len(all_results["scales"]))
    if all_results["run_params"]["log_scale"]:
        return np.log10(all_results["scales"])
    return all_results["scales"]  # pragma: no cover


def _plot_number_comm(all_results, ax, scales):
    """Plot number of communities."""
    ax.plot(scales, all_results["number_of_communities"], "-", c="C3", label="size", lw=2.0)
    ax.set_ylim(0, 1.1 * max(all_results["number_of_communities"]))
    ax.set_ylabel("# clusters", color="C3")
    ax.tick_params("y", colors="C3")


def _plot_ttprime(all_results, ax, scales):
    """Plot ttprime."""
    contourf_ = ax.contourf(scales, scales, all_results["ttprime"], cmap="YlOrBr_r", extend="min")
    ax.set_ylabel(r"$log_{10}(t^\prime)$")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.axis([scales[0], scales[-1], scales[0], scales[-1]])
    ax.set_xlabel(r"$log_{10}(t)$")

    axins = inset_axes(
        ax,
        width="3%",
        height="40%",
        loc="lower left",
        bbox_to_anchor=(0.05, 0.45, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    axins.tick_params(labelsize=7)
    plt.colorbar(contourf_, cax=axins, label="NVI(t,t')")


def _plot_NVI(all_results, ax, scales):
    """Plot variation information."""
    ax.plot(scales, all_results["NVI"], "-", lw=2.0, c="C2", label="VI")

    ax.yaxis.tick_right()
    ax.tick_params("y", colors="C2")
    ax.set_ylabel(r"NVI", color="C2")
    ax.axhline(1, ls="--", lw=1.0, c="C2")
    ax.axis([scales[0], scales[-1], 0.0, np.max(all_results["NVI"]) * 1.1])
    ax.set_xlabel(r"$log_{10}(t)$")


def _plot_stability(all_results, ax, scales):
    """Plot stability."""
    ax.plot(scales, all_results["stability"], "-", label=r"Stability", c="C0")
    ax.tick_params("y", colors="C0")
    ax.set_ylabel("Stability", color="C0")
    ax.set_ylim(0, 1.1 * max(all_results["stability"]))
    ax.yaxis.set_label_position("left")


def _plot_optimal_scales(all_results, ax, scales, ax1, ax2):
    """Plot stability."""
    ax.plot(
        scales,
        all_results["block_nvi"],
        "-",
        lw=2.0,
        c="C4",
        label="Block NVI",
    )
    ax.plot(
        scales[all_results["selected_partitions"]],
        all_results["block_nvi"][all_results["selected_partitions"]],
        "o",
        lw=2.0,
        c="C4",
        label="optimal scales",
    )

    ax.tick_params("y", colors="C4")
    ax.set_ylabel("Block NVI", color="C4")
    ax.yaxis.set_label_position("left")
    ax.set_xlabel(r"$log_{10}(t)$")

    for scale in scales[all_results["selected_partitions"]]:
        ax.axvline(scale, ls="--", color="C4")
        ax1.axvline(scale, ls="--", color="C4")
        ax2.axvline(scale, ls="--", color="C4")


def plot_scan_plt(all_results, figsize=(6, 5), scale_axis=True, figure_name="scan_results.svg"):
    """Plot results of pygenstability with matplotlib."""
    scales = _get_scales(all_results, scale_axis=scale_axis)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 1.0, 0.5])
    gs.update(hspace=0)
    axes = []

    if "ttprime" in all_results:
        ax0 = plt.subplot(gs[1, 0])
        axes.append(ax0)
        _plot_ttprime(all_results, ax=ax0, scales=scales)
        ax1 = ax0.twinx()
    else:  # pragma: no cover
        ax1 = plt.subplot(gs[1, 0])

    axes.append(ax1)
    ax1.set_xticks([])

    _plot_NVI(all_results, ax=ax1, scales=scales)

    if "ttprime" in all_results:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

    ax2 = plt.subplot(gs[0, 0])

    if "stability" in all_results:
        _plot_stability(all_results, ax=ax2, scales=scales)
        ax2.set_xticks([])
        axes.append(ax2)

    if "NVI" in all_results:
        ax3 = ax2.twinx()
        _plot_number_comm(all_results, ax=ax3, scales=scales)
        axes.append(ax3)

    if "block_nvi" in all_results:
        ax4 = plt.subplot(gs[2, 0])
        _plot_optimal_scales(all_results, ax=ax4, scales=scales, ax1=ax1, ax2=ax2)
        axes.append(ax4)

    for ax in axes:
        ax.set_xlim(scales[0], scales[-1])

    if figure_name is not None:
        plt.savefig(figure_name)

    return axes


def plot_clustered_adjacency(
    adjacency,
    all_results,
    scale,
    labels=None,
    figsize=(12, 10),
    cmap="Blues",
    figure_name="clustered_adjacency.pdf",
):
    """Plot the clustered adjacency matrix of the graph at a given scale.

    Args:
        adjacency (ndarray): adjacency matrix to plot
        all_results (dict): results of PyGenStability
        scale (int): scale index for clustering
        labels (list): node labels, or None
        figsize (tubple): figure size
        cmap (str): colormap for matrix elements
        figure_name (str): filename of the figure with extension
    """
    comms, counts = np.unique(all_results["community_id"][scale], return_counts=True)

    node_ids = []
    for comm in comms:
        node_ids += list(np.where(all_results["community_id"][scale] == comm)[0])

    adjacency = adjacency[np.ix_(node_ids, node_ids)]
    adjacency[adjacency == 0] = np.nan

    plt.figure(figsize=figsize)
    plt.imshow(adjacency, aspect="auto", cmap=cmap)

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

    if labels is not None:  # pragma: no cover
        labels_plot = [labels[i] for i in node_ids]
        ax.set_xticklabels(labels_plot)
        ax.set_yticklabels(labels_plot)

    plt.colorbar()
    plt.xticks(rotation=90)
    plt.axis([-0.5, len(adjacency) - 0.5, -0.5, len(adjacency) - 0.5])
    plt.suptitle(
        "log10(scale) = "
        + str(np.round(np.log10(all_results["scales"][scale]), 2))
        + ",  number_of_communities="
        + str(all_results["number_of_communities"][scale])
    )

    plt.savefig(figure_name, bbox_inches="tight")
