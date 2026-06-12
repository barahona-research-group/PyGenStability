"""Plotting functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    logging.getLogger(__name__).warning(
        'Please install networkx via pip install "pygenstability[networkx]" for full plotting.'
    )

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
    all_results: dict[str, Any],
    figsize: tuple[float, float] = (6, 5),
    scale_axis: bool = True,
    figure_name: str | Path | None = "scan_results.pdf",
    use_plotly: bool = False,
    live: bool = True,
    plotly_filename: str = "scan_results.html",
    n_clusters_log_scale: bool = True,
) -> Any:
    """Plot results of pygenstability with matplotlib or plotly.

    Args:
        all_results (dict): results of pygenstability scan
        figsize (tuple): matplotlib figure size
        scale_axis (bool): display scale of scale index on scale axis
        figure_name (str): name of matplotlib figure
        use_plotly (bool): use matplotlib or plotly backend
        live (bool): for plotly backend, open browser with pot
        plotly_filename (str): filename of .html figure from plotly
        n_clusters_log_scale (bool): draw the # clusters axis on a log scale
    """
    if len(all_results["scales"]) == 1:  # pragma: no cover
        L.info("Cannot plot the results if only one scale point, we display the result instead:")
        L.info(all_results)
        return None

    if use_plotly:
        return plot_scan_plotly(all_results, live=live, filename=plotly_filename)
    return plot_scan_plt(
        all_results,
        figsize=figsize,
        scale_axis=scale_axis,
        figure_name=figure_name,
        n_clusters_log_scale=n_clusters_log_scale,
    )


def plot_scan_plotly(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    all_results: dict[str, Any],
    live: bool = False,
    filename: str | None = "clusters.html",
) -> tuple[Any, Any]:
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
    graph: Any,
    all_results: dict[str, Any],
    scale_id: int,
    edge_color: str = "0.5",
    edge_width: float = 0.5,
    node_size: float = 100,
) -> None:
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

    log10_scale = np.round(np.log10(all_results["scales"][scale_id]), 2)
    n_comm = all_results["number_of_communities"][scale_id]
    plt.axis("off")
    plt.title(rf"$log_{{10}}(scale) =$ {log10_scale}, with {n_comm} communities")


def plot_optimal_partitions(
    graph: Any,
    all_results: dict[str, Any],
    edge_color: str = "0.5",
    edge_width: float = 0.5,
    folder: str | Path = "optimal_partitions",
    ext: str = ".pdf",
    show: bool = False,
) -> None:
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
    Path(folder).mkdir(parents=True, exist_ok=True)

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
        plt.savefig(Path(folder) / f"scale_{optimal_scale_id}{ext}", bbox_inches="tight")
        if show:  # pragma: no cover
            plt.show()


def plot_communities(
    graph: Any,
    all_results: dict[str, Any],
    folder: str | Path = "communities",
    edge_color: str = "0.5",
    edge_width: float = 0.5,
    ext: str = ".pdf",
) -> None:
    """Plot the community structures at each scale in a folder.

    Args:
        graph (networkx.Graph): graph to plot
        all_results (dict): results of pygenstability scan
        folder (str): folder to save figures
        edge_color (str): color of edges
        edge_width (float): width of edgs
        ext (str): extension of figures files
    """
    Path(folder).mkdir(parents=True, exist_ok=True)

    for scale_id in tqdm(range(len(all_results["scales"]))):
        plt.figure()
        plot_single_partition(
            graph, all_results, scale_id, edge_color=edge_color, edge_width=edge_width
        )
        plt.savefig(Path(folder) / f"scale_{scale_id}{ext}", bbox_inches="tight")
        plt.close()


def plot_communities_matrix(
    graph: Any,
    all_results: dict[str, Any],
    folder: str | Path = "communities_matrix",
    ext: str = ".pdf",
) -> None:
    """Plot communities at all scales in matrix form.

    Args:
        graph (array): as a numpy matrix
        all_results (dict): clustring results
        folder (str): folder to save figures
        ext (str): figure file format
    """
    Path(folder).mkdir(parents=True, exist_ok=True)

    for scale_id in tqdm(range(len(all_results["scales"]))):
        plt.figure()
        com_ids = all_results["community_id"][scale_id]
        ids: list[int] = []
        line_lengths: list[int] = [0]
        for i in range(len(set(com_ids))):
            _ids = [int(x) for x in np.argwhere(com_ids == i).flatten()]
            line_lengths.append(len(_ids))
            ids += _ids
        plt.imshow(graph[ids][:, ids], origin="lower")
        lines = np.cumsum(line_lengths)
        for i in range(len(lines) - 1):
            plt.plot((lines[i], lines[i + 1]), (lines[i], lines[i]), c="k")
            plt.plot((lines[i], lines[i]), (lines[i], lines[i + 1]), c="k")
            plt.plot((lines[i + 1], lines[i + 1]), (lines[i + 1], lines[i]), c="k")
            plt.plot((lines[i + 1], lines[i]), (lines[i + 1], lines[i + 1]), c="k")

        plt.savefig(Path(folder) / f"scale_{scale_id}{ext}", bbox_inches="tight")
        plt.close()


def _get_scales(all_results: dict[str, Any], scale_axis: bool = True) -> np.ndarray:
    """Get the scale vector."""
    if not scale_axis:  # pragma: no cover
        return np.arange(len(all_results["scales"]))
    if all_results["run_params"]["log_scale"]:
        return np.log10(all_results["scales"])
    return all_results["scales"]  # pragma: no cover


def _plot_number_comm(
    all_results: dict[str, Any],
    ax: Any,
    scales: np.ndarray,
    log_scale: bool = True,
) -> None:
    """Plot number of communities as a step plot, optionally on a log y-axis."""
    n_clusters = np.asarray(all_results["number_of_communities"])
    ax.step(scales, n_clusters, where="post", c="0.4", lw=1.5, label="# clusters")
    ax.set_ylabel("# clusters", color="0.4")
    ax.tick_params("y", colors="0.4")
    if log_scale:
        ax.set_yscale("log")
        lo = max(1, int(n_clusters.min()))
        hi = max(lo + 1, int(n_clusters.max()))
        ax.set_ylim(lo, hi * 1.5)
    else:
        ax.set_ylim(0, 1.1 * max(1, n_clusters.max()))


def _plot_ttprime(all_results: dict[str, Any], ax: Any, scales: np.ndarray) -> None:
    """Plot ttprime."""
    contourf_ = ax.contourf(scales, scales, all_results["ttprime"], cmap="YlOrBr_r", extend="min")
    ax.set_ylabel(r"$log_{10}(t^\prime)$")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
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


def _plot_NVI(all_results: dict[str, Any], ax: Any, scales: np.ndarray) -> None:
    """Plot variation information."""
    ax.plot(scales, all_results["NVI"], "-", lw=2.0, c="C2", label="VI")

    ax.tick_params("y", colors="C2")
    ax.set_ylabel(r"NVI", color="C2")
    ax.axhline(1, ls="--", lw=1.0, c="C2")
    nvi_max = max(np.max(all_results["NVI"]) * 1.1, 1e-3)
    ax.axis([scales[0], scales[-1], 0.0, nvi_max])
    ax.set_xlabel(r"$log_{10}(t)$")


def _plot_stability(all_results: dict[str, Any], ax: Any, scales: np.ndarray) -> None:
    """Plot stability."""
    ax.plot(scales, all_results["stability"], "-", label=r"Stability", c="C0")
    ax.tick_params("y", colors="C0")
    ax.set_ylabel("Stability", color="C0")
    ax.set_ylim(0, 1.1 * max(all_results["stability"]))
    ax.yaxis.set_label_position("left")


def _plot_block_nvi(all_results: dict[str, Any], ax: Any, scales: np.ndarray) -> None:
    """Plot the block-NVI curve with dots at the selected scales."""
    block_nvi = np.asarray(all_results["block_nvi"])
    ax.plot(scales, block_nvi, "-", lw=1.5, c="k", label="Block NVI")
    if "selected_partitions" in all_results:
        sel = list(all_results["selected_partitions"])
        ax.plot(scales[sel], block_nvi[sel], "o", c="k", ms=5)
    ax.set_ylabel("Block NVI", color="k")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")


def _draw_selected_scale_markers(
    all_results: dict[str, Any],
    axes: list[Any],
    label_ax: Any,
    scales: np.ndarray,
) -> None:
    """Dashed red verticals across all panels + k=N text labels above the top panel."""
    if "selected_partitions" not in all_results:
        return
    sel = list(all_results["selected_partitions"])
    n_clusters = all_results.get("number_of_communities")
    for i in sel:
        for ax in axes:
            ax.axvline(scales[i], ls="--", color="red", lw=1.0)
        if n_clusters is not None:
            label_ax.text(
                scales[i],
                1.02,
                f"k={n_clusters[i]}",
                transform=label_ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="red",
                fontsize=8,
            )


def plot_scan_plt(
    all_results: dict[str, Any],
    figsize: tuple[float, float] = (6, 5),
    scale_axis: bool = True,
    figure_name: str | Path | None = "scan_results.svg",
    n_clusters_log_scale: bool = True,
) -> list[Any]:
    """Plot results of pygenstability with matplotlib.

    Layout (top → bottom): stability + #clusters; ttprime heatmap with block-NVI
    overlay; NVI(t). Selected scales are marked by red dashed verticals across
    all panels and `k=N` text labels above the top panel.
    """
    scales = _get_scales(all_results, scale_axis=scale_axis)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 1.0, 0.5])
    gs.update(hspace=0)
    axes: list[Any] = []

    # top: stability (left) + #clusters (right)
    ax_top = plt.subplot(gs[0, 0])
    if "stability" in all_results:
        _plot_stability(all_results, ax=ax_top, scales=scales)
    ax_top.set_xticks([])
    axes.append(ax_top)
    if "number_of_communities" in all_results:
        ax_nk = ax_top.twinx()
        _plot_number_comm(all_results, ax=ax_nk, scales=scales, log_scale=n_clusters_log_scale)
        axes.append(ax_nk)

    # middle: ttprime heatmap + block-NVI overlay
    ax_mid = plt.subplot(gs[1, 0])
    axes.append(ax_mid)
    ax_mid.set_xticks([])
    # Create the block-NVI twin axis *before* plotting ttprime. matplotlib's
    # twinx() forces the parent y-axis ticks back to the left, which would
    # otherwise undo the right-side log10(t') ticks set in _plot_ttprime. Doing
    # it first also adds the ttprime colorbar inset last, so the NVI(t,t')
    # legend draws on top of the block-NVI curve rather than behind it.
    ax_block = None
    if "block_nvi" in all_results:
        ax_block = ax_mid.twinx() if "ttprime" in all_results else ax_mid
    if "ttprime" in all_results:
        _plot_ttprime(all_results, ax=ax_mid, scales=scales)
    if ax_block is not None:
        _plot_block_nvi(all_results, ax=ax_block, scales=scales)
        axes.append(ax_block)

    # bottom: NVI(t)
    ax_bot = plt.subplot(gs[2, 0])
    if "NVI" in all_results:
        _plot_NVI(all_results, ax=ax_bot, scales=scales)
    axes.append(ax_bot)

    _draw_selected_scale_markers(all_results, axes=axes, label_ax=ax_top, scales=scales)

    for ax in axes:
        ax.set_xlim(scales[0], scales[-1])

    if figure_name is not None:
        plt.savefig(figure_name)

    return axes


def plot_clustered_adjacency(
    adjacency: Any,
    all_results: dict[str, Any],
    scale: int,
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (12, 10),
    cmap: str = "Blues",
    figure_name: str | Path = "clustered_adjacency.pdf",
) -> None:
    """Plot the clustered adjacency matrix of the graph at a given scale.

    Args:
        adjacency (ndarray or sparse matrix): adjacency matrix to plot
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

    # densify sparse inputs so np.ix_ fancy indexing works
    if hasattr(adjacency, "toarray"):
        adjacency = adjacency.toarray()
    adjacency = np.asarray(adjacency)[np.ix_(node_ids, node_ids)].astype(float)
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
    plt.axis((-0.5, len(adjacency) - 0.5, -0.5, len(adjacency) - 0.5))
    log10_scale = np.round(np.log10(all_results["scales"][scale]), 2)
    n_comm = all_results["number_of_communities"][scale]
    plt.suptitle(f"log10(scale) = {log10_scale},  number_of_communities={n_comm}")

    plt.savefig(figure_name, bbox_inches="tight")
