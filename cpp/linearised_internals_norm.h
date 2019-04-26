#pragma once

#include "graphhelpers.h"


namespace clq
{

struct LinearisedInternals {
    //    typedef lemon::RangeMap<double> range_map;
    typedef std::vector<double> range_map;
    unsigned int num_nodes;
    double two_m;
    range_map node_to_w;
    range_map comm_w_tot;
    range_map comm_w_in;
    //std::map<int, double> node_weight_to_communities;
    std::vector<double> node_weight_to_communities;
    std::vector<unsigned int> neighbouring_communities_list;

    template<typename G, typename M>
    LinearisedInternals( G& graph, M& weights ) :
        num_nodes( lemon::countNodes( graph ) ), node_to_w( num_nodes, 0 ),
        comm_w_tot( num_nodes, 0 ), comm_w_in( num_nodes, 0 ),
        node_weight_to_communities( lemon::countNodes( graph ), 0 ),
        neighbouring_communities_list()
    {
        two_m = 2 * find_total_weight( graph, weights );

        for( unsigned int i = 0; i < num_nodes; ++i ) {
            typename G::Node temp_node = graph.nodeFromId( i );
            comm_w_tot[i] = node_to_w[i] = find_weighted_degree( graph, weights,
                                           temp_node );
            comm_w_in[i] = find_weight_selfloops( graph, weights, temp_node );
        }
    }

    template<typename G, typename M, typename P>
    LinearisedInternals( G& graph, M& weights, P& partition ) :
        num_nodes( lemon::countNodes( graph ) ), node_to_w( num_nodes, 0 ),
        comm_w_tot( num_nodes, 0 ), comm_w_in( num_nodes, 0 ),
        node_weight_to_communities( lemon::countNodes( graph ), 0 ),
        neighbouring_communities_list()
    {
        two_m = 2 * find_total_weight( graph, weights );

        typedef typename G::EdgeIt EdgeIt;

        // find internal statistics based on graph, weights and partitions
        // consider all edges
        for( EdgeIt edge( graph ); edge != lemon::INVALID; ++edge ) {
            int node_u_id = graph.id( graph.u( edge ) );
            int node_v_id = graph.id( graph.v( edge ) );

            // this is to distinguish within community weight with total weight
            int comm_of_node_u = partition.find_set( node_u_id );
            int comm_of_node_v = partition.find_set( node_v_id );

            // weight of edge
            double weight = weights[edge];

            // if selfloop, only half of the weight has to be considered
            if( node_u_id == node_v_id ) {
                weight = weight / 2;
            }

            // add weight to node weight
            node_to_w[node_u_id] += weight;
            node_to_w[node_v_id] += weight;
            // add weight to total weight of community
            comm_w_tot[comm_of_node_u] += weight;
            comm_w_tot[comm_of_node_v] += weight;

            if( comm_of_node_u == comm_of_node_v ) {
                // in case the weight stems from within the community add to internal weights
                comm_w_in[comm_of_node_u] += 2 * weight;
            }
        }
    }
};

/**
 @brief  isolate a node into its singleton set & update internals
 */
template<typename G, typename M, typename P>
void isolate_and_update_internals( G& graph, M& weights,
                                   typename G::Node node, LinearisedInternals& internals, P& partition )
{
    int node_id = graph.id( node );
    int comm_id = partition.find_set( node_id );

    // reset weights
    while( !internals.neighbouring_communities_list.empty() ) {
        //clq::output("empty");
        unsigned int old_neighbour =
            internals.neighbouring_communities_list.back();
        internals.neighbouring_communities_list.pop_back();
        internals.node_weight_to_communities[old_neighbour] = 0;

    }

    //clq::output(internals.neighbouring_communities_list.size());
    // get weights from node to each community
    for( typename G::IncEdgeIt e( graph, node ); e != lemon::INVALID; ++e ) {
        if( graph.u( e ) != graph.v( e ) ) {
            double edge_weight = weights[e];
            typename G::Node opposite_node = graph.oppositeNode( node, e );
            int comm_node = partition.find_set( graph.id( opposite_node ) );

            if( internals.node_weight_to_communities[comm_node]==0 ) {
                internals.neighbouring_communities_list.push_back( comm_node );
            }

            internals.node_weight_to_communities[comm_node] += edge_weight;
        }
    }

    //clq::print_collection(internals.node_weight_to_communities);
    internals.comm_w_tot[comm_id] -= internals.node_to_w[node_id];
    //clq::output("in", internals.comm_w_in[comm_id]);
    internals.comm_w_in[comm_id] -= 2
                                    * internals.node_weight_to_communities[comm_id]
                                    + find_weight_selfloops( graph, weights, node );
    //clq::output("in", internals.comm_w_in[comm_id]);

    partition.isolate_node( node_id );
}

/**
 @brief  insert a node into the best set & update internals
 */
template<typename G, typename M, typename P>
void insert_and_update_internals( G& graph, M& weights, typename G::Node node,
                                  LinearisedInternals& internals, P& partition, int best_comm )
{
    int node_id = graph.id( node );
    // insert node to partition/bookkeeping
    //              std::cout << "node "<<node_id << " comm: to "<< best_comm << std::endl;
    internals.comm_w_tot[best_comm] += internals.node_to_w[node_id];
    //              std::cout << "new_weight tot " <<internals.comm_w_tot[best_comm] << std::endl;
    internals.comm_w_in[best_comm] += 2
                                      * internals.node_weight_to_communities[best_comm]
                                      + find_weight_selfloops( graph, weights, node );
    //              std::cout << "new_weight int " <<internals.comm_w_in[best_comm] << std::endl;

    partition.add_node_to_set( node_id, best_comm );
    //    clq::output("in", internals.comm_w_in[best_comm], "tot",internals.comm_w_tot[best_comm], "nodew", internals.node_to_w[best_comm], "2m", internals.two_m);
    //    clq::print_partition_line(partition);
}

}
