#pragma once

#include <vector>
//#include "../lemon-lib/lemon/concepts/graph.h"
//#include <lemon/smart_graph.h>
//#include <time.h>
#include "graphhelpers.h"
#include "linearised_internals_generic.h"

namespace clq
{

/**
 @brief  A functor for evaluating the generic stability of a weighted graph
 */

struct find_linearised_generic_stability {
    // Markov time state variable
    double markov_time;

    // Constructor with default Markov time of 1
    find_linearised_generic_stability (double markov_time = 1.0) :
        markov_time (markov_time)
    {
    }

    template<typename I>
    double operator () (I& internals)
    {

        // first part equals 1-t
        double q = 1 - markov_time;
        double q2 = 0;

        // loop over all communities and sum up contributions (gain - loss)
        unsigned int num_null_model_vec = internals.num_null_model_vectors;
        unsigned int num_nodes = internals.num_nodes;

        //little helper to keep track of non-empty communities..
        bool check;

        // loop over all possible community indices
        for (unsigned int i = 0; i < num_nodes; i++) {
            // for each possible index check if the community is non-empty, i.e. if there is a loss term
            double gain = markov_time * double (internals.comm_w_in[i]);

            //TODO: check if this is the right contruction here.. it should be really..
            if (gain != 0) {
                q2 += gain;
                check = true;
            } else {
                check = true;
            }

            if (check) {
                for (unsigned int j = 0; j < num_null_model_vec; j = j + 2) {
                    //clq::output ("loss terms: ");
                    //clq::print_2d_vector (internals.comm_loss_vectors);
                    q2 -= double (internals.comm_loss_vectors[j][i])
                          * double (internals.comm_loss_vectors[j + 1][i]);

                }
            }

        }

        return q + q2;
    }

};


/**
 @brief  Functor for finding stability gain (normalised Laplacian) with for weighted graph
 */
struct linearised_generic_stability_gain {

    //state variable Markov time
    double markov_time;

    //constructor
    linearised_generic_stability_gain (double mtime = 1.0) :
        markov_time (mtime)
    {
    }

    template<typename I>
    double operator () (I& internals, int comm_id_neighbour, int node_id)
    {
        // compute loss factor incurred by moving node...
        double comm_loss = 0;

        //clq::output(comm_loss,internals.num_null_model_vectors,"loss\n");
        for (unsigned int i = 0; i < internals.num_null_model_vectors; i = i + 2) {
            //clq::output("DEBUG", i, node_id, comm_id_neighbour);
            comm_loss +=
                internals.null_model_vectors[i][node_id] * internals.comm_loss_vectors[i + 1][comm_id_neighbour]
                + internals.null_model_vectors[i + 1][node_id] * internals.comm_loss_vectors[i][comm_id_neighbour];
        }

        //clq::output(comm_loss,internals.num_null_model_vectors,"loss\n");

        // gain resulting from adding to community
        double w_node_to_comm = internals.node_weight_to_communities[comm_id_neighbour];
        return markov_time * w_node_to_comm * 2 - comm_loss;
    }
};

}
