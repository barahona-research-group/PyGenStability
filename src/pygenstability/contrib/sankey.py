"""Sankey diagram plots."""

import numpy as np  # pragma: no cover

try:
    import plotly.graph_objects as go
    from plotly.offline import plot

    with_plotly = True
except ImportError:  # pragma: no cover
    with_plotly = False


def plot_sankey(
    all_results,
    optimal_scales=True,
    live=False,
    filename="communities_sankey.html",
    scale_index=None,
):  # pragma: no cover
    """Plot Sankey diagram of communities accros scale (plotly only).

    Args:
        all_results (dict): results from run function
        optimal_scales (bool): use optimal scales or not
        live (bool): if True, interactive figure will appear in browser
        filename (str): filename to save the plot
        scale_index (bool): plot scale of indices
    """
    sources = []
    targets = []
    values = []
    shift = 0

    if not scale_index:
        all_results["community_id_reduced"] = all_results["community_id"]
    else:
        all_results["community_id_reduced"] = [all_results["community_id"][i] for i in scale_index]

    community_ids = all_results["community_id_reduced"]
    if optimal_scales and ("selected_partitions" in all_results.keys()):
        community_ids = [community_ids[u] for u in all_results["selected_partitions"]]

    for i in range(len(community_ids) - 1):
        community_source = np.array(community_ids[i])
        community_target = np.array(community_ids[i + 1])
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
                node={
                    "pad": 1,
                    "thickness": 1,
                    "line": {"color": "black", "width": 0.0},
                },
                link={"source": sources, "target": targets, "value": values},
            )
        ],
        layout=layout,
    )

    if with_plotly:
        plot(fig, filename=filename)
    else:
        print("Plotly not installed, we cannot plot the figure")

    if live:
        fig.show()

    return fig
