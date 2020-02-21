"""plotting functions"""
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

L = logging.getLogger("pygenstability")


def plot_scan(all_results, time_axis=True, figure_name="scan_results.svg"):
    """
    Simple plot of a scan
    """

    # get the times paramters
    n_t = len(all_results["times"])
    if n_t == 1:
        L.info(
            "Cannot plot the results if only one time point, we display the result instead:"
        )
        L.info(all_results)
        return

    times = np.log10(all_results["times"])

    plt.figure(figsize=(5, 5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.5])  # , width_ratios = [1,0.2] )
    gs.update(hspace=0)

    # plot tt'
    if "ttprime" in all_results:
        ax0 = plt.subplot(gs[0, 0])
        ttprime = np.zeros([n_t, n_t])
        for i, tt in enumerate(all_results["ttprime"]):
            ttprime[i] = tt

        ax0.contourf(times, times, ttprime, cmap="YlOrBr")
        ax0.set_ylabel(r"$log_{10}(t^\prime)$")
        ax0.yaxis.tick_left()
        ax0.yaxis.set_label_position("left")
        ax0.axis([times[0], times[-1], times[0], times[-1]])

        ax1 = ax0.twinx()
    else:
        ax1 = plt.subplot(gs[0, 0])

    # plot the number of clusters
    if time_axis:
        ax1.plot(
            times, all_results["number_of_communities"], c="C0", label="size", lw=2.0
        )
    else:
        ax1.plot(all_results["number_of_communities"], c="C0", label="size", lw=2.0)

    ax1.tick_params("y", colors="C0")
    if "ttprime" in all_results:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
    ax1.set_ylabel("Number of clusters", color="C0")

    # make a subplot for stability and MI
    ax2 = plt.subplot(gs[1, 0])

    # first plot the stability
    if time_axis:
        ax2.plot(times, all_results["stability"], label=r"$Q$", c="C2")
    else:
        ax2.plot(all_results["stability"], label=r"$Q$", c="C2")

    # ax2.set_yscale('log')
    ax2.tick_params("y", colors="C2")
    ax2.set_ylabel("Modularity", color="C2")
    ax2.yaxis.set_label_position("left")
    # ax2.legend(loc='center right')
    ax2.set_xlabel(r"$log_{10}(t)$")

    # ax2.axis([0,n_t,0,self.stability_results.at[0,'number_of_communities']])

    # plot the MMI
    if "mutual_information" in all_results:
        ax3 = ax2.twinx()
        if time_axis:
            ax3.plot(
                times,
                all_results["mutual_information"],
                "-",
                lw=2.0,
                c="C3",
                label="MI",
            )
        else:
            ax3.plot(all_results["mutual_information"], "-", lw=2.0, c="C3", label="MI")

        ax3.yaxis.tick_right()
        ax3.tick_params("y", colors="C3")
        ax3.set_ylabel(r"Mutual information", color="C3")
        ax3.axhline(1, ls="--", lw=1.0, c="C3")
        ax3.axis(
            [times[0], times[-1], np.min(all_results["mutual_information"]) * 0.9, 1.1]
        )

    plt.savefig(figure_name, bbox_inches="tight")


def plot_communities(graph, all_results, folder="communities"):
    """now plot the community structures at each time in a folder"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    pos = [graph.nodes[u]["pos"] for u in graph]

    for i in range(len(all_results["times"])):
        node_color = all_results["community_id"][i]

        plt.figure()
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            node_color=node_color,
            node_size=100,
            cmap=plt.get_cmap("tab20"),
        )
        nx.draw_networkx_edges(graph, pos=pos, width=0.5, edge_color="0.5")

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
