#pragma once

#include "graphhelpers.h"
#include "io.h"

namespace clq
{
// Define internal structure to carry statistics for generalised stability

struct LinearisedInternalsGeneric {


    // typedef for convenience
    typedef std::vector<std::vector<double>> vec_of_vec;
    unsigned int num_nodes;

    // how many outer products do we have? must be multiple of 2
    unsigned int num_null_model_vectors;
    // contains the outer product vectors of the null model
    vec_of_vec null_model_vectors;

    // loss for each community (second matrix / null model terms per community)
    vec_of_vec comm_loss_vectors;
    // gain for each community (first matrix / gain term per community)
    std::vector<double> comm_w_in;
    // mapping: node weight to each community (gain of adding node to a community, dyn. updated)
    std::vector<double> node_weight_to_communities;
    // associated list of neighbouring communities
    std::vector<unsigned int> neighbouring_communities_list;

    //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    // simple constructor, no partition given
    //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    template<typename G, typename M>
        LinearisedInternalsGeneric (G& graph, M& weights, vec_of_vec null_model_input) :
        num_nodes (lemon::countNodes (graph) ),
        num_null_model_vectors (null_model_input.size() ),
        null_model_vectors (null_model_input),
        comm_loss_vectors (null_model_input),
        comm_w_in ( num_nodes, 0),
        node_weight_to_communities (num_nodes, 0), neighbouring_communities_list()
    {
        if (num_null_model_vectors % 2 != 0 ) {
            clq::output ("Null model vectors must be provided as pairs!");
            exit (EXIT_FAILURE);
        }

        for (unsigned int i = 0; i < num_nodes; ++i) {
            typename G::Node temp_node = graph.nodeFromId (i);
            comm_w_in[i] = find_weight_selfloops (graph, weights, temp_node);

        }

        //clq::output("CONSTRUCTOR Internals");
        //clq::output("num_nodes",num_nodes,"num_null_model",num_null_model_vectors);
        //clq::output("null model internals");
        //print_2d_vector(null_model_vectors);
        //clq::output("loss vectors internals");
        //print_2d_vector(comm_loss_vectors);
        //clq::output("gain internals");
        //print_collection(comm_w_in);
        //clq::output("end constructor\n");

    }

    //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    // full constructor with reference to partition
    //$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    template<typename G, typename M, typename P>
        LinearisedInternalsGeneric (G& graph, M& weights, P& partition, vec_of_vec null_model_input) :
        num_nodes (lemon::countNodes (graph) ),
        num_null_model_vectors (null_model_input.size() ),
        null_model_vectors (null_model_input),
        comm_loss_vectors (num_null_model_vectors, std::vector<double> (num_nodes, 0) ),
        comm_w_in (num_nodes, 0),
        node_weight_to_communities (num_nodes, 0), neighbouring_communities_list()
    {
        typedef typename G::EdgeIt EdgeIt;

        if (num_null_model_vectors % 2 != 0 ) {
            clq::output ("Null model vectors must be provided as pairs!");
            exit (EXIT_FAILURE);
        }

        // find internal statistics based on graph, weights and partitions
        // consider all edges
        for (EdgeIt edge (graph); edge != lemon::INVALID; ++edge) {
            int node_u_id = graph.id (graph.u (edge) );
            int node_v_id = graph.id (graph.v (edge) );

            // this is to distinguish within community weight with total weight
            int comm_of_node_u = partition.find_set (node_u_id);
            int comm_of_node_v = partition.find_set (node_v_id);

            // weight of edge
            double weight = weights[edge];

            // if selfloop, only half of the weight has to be considered (multiplication by two afterwards)
            if (node_u_id == node_v_id) {
                weight = weight / 2;
            }

            if (comm_of_node_u == comm_of_node_v) {
                // in case the weight stems from within the community add to internal weights
                comm_w_in[comm_of_node_u] += 2 * weight;
            }

        }

        // setup internal loss vectors (summing up all the null model terms per group)
        for (unsigned int i = 0; i < num_nodes; ++i) {
            int comm_id = partition.find_set (i);

            for (unsigned int k = 0; k < num_null_model_vectors; ++k) {
                comm_loss_vectors[k][comm_id] += null_model_vectors[k][i];
            }
        }


        //clq::output("CONSTRUCTOR Internals");
        //clq::output("num_nodes",num_nodes,"num_null_model",num_null_model_vectors);
        //clq::output("null model internals");
        //print_2d_vector(null_model_vectors);
        //clq::output("loss vectors internals");
        //print_2d_vector(comm_loss_vectors);
        //clq::output("gain internals");
        //print_collection(comm_w_in);
        //clq::output("end constructor\n");
    }
};


/**
 @brief  isolate a node into its singleton set & update internals
 */
template<typename G, typename M, typename P>
void isolate_and_update_internals (G& graph, M& weights, typename G::Node node,
                                   LinearisedInternalsGeneric& internals, P& partition)
{
    int node_id = graph.id (node);
    int comm_id = partition.find_set (node_id);

    // reset weights
    while (!internals.neighbouring_communities_list.empty() ) {
        //clq::output("empty");
        unsigned int old_neighbour =
            internals.neighbouring_communities_list.back();
        internals.neighbouring_communities_list.pop_back();
        internals.node_weight_to_communities[old_neighbour] = 0;
    }

    // get weights from node to each community
    for (typename G::IncEdgeIt e (graph, node); e != lemon::INVALID; ++e) {
        // check that you do not get a self-loop
        if (graph.u (e) != graph.v (e) ) {
            // get the edge weight
            double edge_weight = weights[e];
            // get the other node
            typename G::Node opposite_node = graph.oppositeNode (node, e);
            // get community id of the other node
            int comm_node = partition.find_set (graph.id (opposite_node) );

            // check if we have seen this community already
            if (internals.node_weight_to_communities[comm_node] == 0) {
                internals.neighbouring_communities_list.push_back (comm_node);
            }

            // add weights to vector
            internals.node_weight_to_communities[comm_node] += edge_weight;
        }
    }

//    clq::print_collection(internals.node_weight_to_communities);
//    clq::print_partition_line(partition);
//    clq::output("loss", internals.comm_loss_vectors[1][comm_id]);
    for (unsigned int j = 0; j < internals.num_null_model_vectors; ++j) {
        internals.comm_loss_vectors[j][comm_id] -= internals.null_model_vectors[j][node_id];
    }


//	clq::output("loss", internals.comm_loss[comm_id]);
//  clq::output("in", internals.comm_w_in[comm_id]);
    internals.comm_w_in[comm_id] -= 2 * internals.node_weight_to_communities[comm_id]
                                    + find_weight_selfloops (graph, weights, node);
//  clq::output("in", internals.comm_w_in[comm_id]);

    partition.isolate_node (node_id);
}


/**
 @brief  insert a node into the best set & update internals
 */
template<typename G, typename M, typename P>
void insert_and_update_internals (G& graph, M& weights, typename G::Node node,
                                  LinearisedInternalsGeneric& internals, P& partition, int best_comm)
{
    // node id and std dev
    int node_id = graph.id (node);

    // update loss
    for (unsigned int j = 0; j < internals.num_null_model_vectors; ++j) {
        internals.comm_loss_vectors[j][best_comm] += internals.null_model_vectors[j][node_id];
    }

    // update gain
    internals.comm_w_in[best_comm] += 2 * internals.node_weight_to_communities[best_comm]
                                      + find_weight_selfloops (graph, weights, node);

    partition.add_node_to_set (node_id, best_comm);
//  clq::print_partition_line(partition);
}

}
