#pragma once

#include <vector>
#include "graphhelpers.h"
#include "linearised_internals_norm.h"
#include "linearised_internals_comb.h"

namespace clq
{
/**
 @brief  Functor for finding normalised linearised stability of weighted graph
 */
struct find_linearised_normalised_stability {
    double markov_time;

    find_linearised_normalised_stability (double markov_time) : markov_time (markov_time)
    {
    }

    template<typename G, typename P, typename W>
    double operator () (G& graph, P& partition, W& weights)
    {
        clq::LinearisedInternals internals(graph, weights, partition);
        return (*this) (internals);
    }

    template<typename I>
    double operator () (I& internals)
    {
        double q = 1.0 - markov_time;
        int size = internals.comm_w_tot.size();

        for (int i = 0; i < size; i++) {
            if (internals.comm_w_tot[i] > 0) {
                q += markov_time * double (internals.comm_w_in[i] / internals.two_m) -
                     ( (double (internals.comm_w_tot[i]) / internals.two_m) *
                       (double (internals.comm_w_tot[i]) / internals.two_m) );
            }

        }

        return q;
    }

    template<typename I>
    double operator () (I& internals, int comm_id)
    {
        double q = -1;
        int i = comm_id;

        if (internals.comm_w_tot[i] > 0) {
            // "constant" term from linearisation
            q = internals.comm_w_tot[i] / internals.two_m * (1.0 - markov_time);
            // gain - loss
            q += markov_time * double (internals.comm_w_in[i] / internals.two_m)
                 - ( (double (internals.comm_w_tot[i]) / internals.two_m)
                     * (double (internals.comm_w_tot[i]) / internals.two_m) );
        } else {
            std::cout << "This community does not exist!!!" << std::endl;
        }

        if (internals.comm_w_tot[i] == internals.two_m) {
            return q;
        } else {
            return q*internals.two_m / internals.comm_w_tot[i];
        }

    }

};

/**
 @brief  Functor for finding stability gain (normalised Laplacian) with for weighted graph
 */
struct linearised_normalised_stability_gain {
    double markov_time;

    linearised_normalised_stability_gain (double mtime) : markov_time (mtime)
    {
    }

    double operator () (double tot_w_comm, double w_node_to_comm, double two_m, double w_deg_node)
    {
        return (markov_time * w_node_to_comm - tot_w_comm * w_deg_node / two_m);
    }

    template<typename I>
    double operator () (I& internals, int comm_id_neighbour, int node_id)
    {
        double tot_w_comm = internals.comm_w_tot[comm_id_neighbour];
        double w_node_to_comm = internals.node_weight_to_communities[comm_id_neighbour];
        double w_deg_node = internals.node_to_w[node_id];

        return (markov_time * w_node_to_comm - tot_w_comm * w_deg_node / internals.two_m) * 2 / internals.two_m;
    }
};

// Linearised combinatorial stability with partition given
template<typename G, typename M, typename P>
clq::LinearisedInternals gen_internals( find_linearised_normalised_stability X,
                                  G& graph, M& weights, P& partition, P& partition_init )
{
    LinearisedInternals internals( graph, weights, partition );
    return internals;
}


//////////////////////////////////////////////////////////////////////////////
// Combinatorial Stability
//////////////////////////////////////////////////////////////////////////////
/**
 @brief  Functor for finding combinatorial linearised stability of weighted graph
 */
struct find_linearised_combinatorial_stability {
    double markov_time;

    find_linearised_combinatorial_stability (double markov_time) :
        markov_time (markov_time)
    {
    }

    template<typename G, typename P, typename W>
    double operator () (G& graph, P& partition, W& weights, P& partition_init)
    {
        clq::LinearisedInternalsComb internals (graph, weights, partition,
                                                partition_init);
        return (*this) (internals);
    }

    template<typename I>
    double operator () (I& internals)
    {
        double q = 1.0 - markov_time;
        int size = internals.comm_tot_nodes.size();
        //				clq::output("time", markov_time, "size", size);

        for (int i = 0; i < size; i++) {
            //			clq::print_collection(internals.comm_tot_nodes);
            if (internals.comm_tot_nodes[i] > 0) {
                q += markov_time * double (internals.comm_w_in[i]
                                           / internals.two_m)
                     - ( (double (internals.comm_tot_nodes[i])
                          / internals.num_nodes_init)
                         * (double (internals.comm_tot_nodes[i])
                            / internals.num_nodes_init) );
            }

        }

        return q;
    }
};

/**
 @brief  Functor for finding stability gain (combinatorial Laplacian) with for weighted graph
 */
struct linearised_combinatorial_stability_gain {
    double markov_time;

    linearised_combinatorial_stability_gain (double mtime) :
        markov_time (mtime)
    {
    }

    double operator () (double comm_tot_nodes, double w_node_to_comm,
                        double two_m, double node_nr_nodes_init,
                        unsigned int num_nodes_init)
    {
        return (markov_time * w_node_to_comm / two_m - (comm_tot_nodes
                / num_nodes_init) * (node_nr_nodes_init / num_nodes_init) );
    }

    template<typename I>
    double operator () (I& internals, int comm_id_neighbour, int node_id)
    {
        double comm_tot_nodes = internals.comm_tot_nodes[comm_id_neighbour];
        double w_node_to_comm =
            internals.node_weight_to_communities[comm_id_neighbour];
        double node_nr_nodes_init = internals.node_to_nr_nodes_init[node_id];
        return (markov_time * w_node_to_comm / internals.two_m
                - (comm_tot_nodes / internals.num_nodes_init)
                * (node_nr_nodes_init / internals.num_nodes_init) ) * 2;
    }
};

// Linearised combinatorial stability with partition given
template<typename G, typename M, typename P>
clq::LinearisedInternalsComb gen_internals( find_linearised_combinatorial_stability X,
                                  G& graph, M& weights, P& partition, P& partition_init )
{
    LinearisedInternalsComb internals( graph, weights, partition, partition_init );
    return internals;
}
}
