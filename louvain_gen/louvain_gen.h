#pragma once

#include <vector>
#include "graphhelpers.h"
#include "stability_gen.h"
#include "stability.h"
#include "linearised_internals_generic.h"
#include "linearised_internals_norm.h"
#include "linearised_internals_comb.h"


namespace clq
{

// For convenience let's get rid of some cumbersome notation..
typedef std::vector<std::vector<double>> vec2;

/**
 @brief  find_best_comm_move -- find the community with the largest gain in quality 
 for a particular node.

 @param[in]  compute_quality_diff -- functor to compute difference in quality
 @param[in]  node_id -- Id of the node to move (into another community)
 @param[in]  internals -- Internals object storing the statistics for the current partition
 @param[in] community_id -- current community_id of the node 
 
 @param[out] (int) -- id of the community with the largest gain in quality 

 */
template <typename QD, typename L>
unsigned int find_best_comm_move( QD compute_quality_diff, unsigned int node_id, L internals,
                            unsigned int comm_id )
{
    unsigned int num_neighbour_communities = internals.neighbouring_communities_list.size();

    //default option for re-inclusion of node
    unsigned int best_comm = comm_id;
    double best_gain = 0;

    // loop over neighbouring communities
    for( unsigned int k = 0; k < num_neighbour_communities; ++k ) {

        unsigned int comm_id_neighbour = internals.neighbouring_communities_list[k];
        double gain = compute_quality_diff( internals, comm_id_neighbour, node_id );

        if( gain > best_gain ) {
            best_comm = comm_id_neighbour;
            best_gain = gain;
            // avoid not necessary movements, place node in old community if possible
        } else if( gain == best_gain && comm_id == comm_id_neighbour ) {
            best_comm = comm_id;
        }

    }

    return best_comm;
}


/**
 @brief fold_partition_into_orginal_graph_size -- given a partition for an aggregated graph, create 
 the corresponding partition for the graph in the original size.

 @param[in]  optimal_partitions -- vector of partitions for the different levels of the 
                                   Louvain algorithm   
 @param[in]  partition -- partition of aggregated graph, to be transformed/expanded to the original
                          graph size

 @param[out] (partition) -- expanded partition with size matching the original graph
 */
template <typename P>
P fold_partition_into_orginal_graph_size( std::vector<P>& optimal_partitions, P partition )
{
    int hierarchy = optimal_partitions.size();

    // if optimal partition is empty there is nothing to fold back..
    if( hierarchy == 0 ) {
        return partition;
    } else {
        // get size of partition one level below, i.e. number of nodes in original graph
        unsigned int original_number_of_nodes = optimal_partitions[hierarchy - 1].element_count();
        // create new empty partition of this size
        P partition_original_nodes( original_number_of_nodes );

        // loop over nodes one level below
        int old_comm, new_comm;

        for( unsigned int id = 0; id < original_number_of_nodes; id++ ) {
            // get the communities for each node one level below
            old_comm = optimal_partitions[hierarchy - 1].find_set( id );
            // use this as node_id in the current partition as old community id
            // is equivalent to new node id and read out new community id
            new_comm = partition.find_set( old_comm );
            // include pair (node, new community) id in the newly created partition
            partition_original_nodes.add_node_to_set( id, new_comm );
        }

        return partition_original_nodes;
    }
}


/**
 @brief create_reduced_null_model_vec -- created an aggregated set of 'null-model' vectors 
 to be used for a corresponding aggregated graph

 @param[in]  new_comm_id_to_old_comm_id -- mapping from new to old community ids
 @param[in]  weights   EdgeMap of edge weights
 @param[in]  num_null_model_vectors -- number of null model vectors to be aggregated.
 @param[in]  num_nodes_reduced_graph -- number of nodes in the reduced graph
 
 @param[out] (vec2) -- vector of vectors,  containing the reduced null model vectors
 */
vec2 create_reduced_null_model_vec( std::map<int, int> new_comm_id_to_old_comm_id, LinearisedInternalsGeneric internals,
                                    unsigned int num_null_model_vectors, unsigned int num_nodes_reduced_graph )
{
    vec2 reduced_null_model_vec( num_null_model_vectors, std::vector<double> ( num_nodes_reduced_graph, 0 ) );

    for( unsigned int k = 0; k < num_nodes_reduced_graph; ++k ) {
        for( unsigned int j = 0; j < num_null_model_vectors; ++j ) {
            reduced_null_model_vec[j][k] = internals.comm_loss_vectors[j][new_comm_id_to_old_comm_id[k]];
        }
    }

    return reduced_null_model_vec;
}



/**
 @brief  generalise Louvain method - greedy algorithm to find community structure of a network.

 @param[in]  graph    --    graph to partition
 @param[in]  weights  --    map of edge weights
 @param[in]  null_model_vec -- null model vectors
 @param[in]  compute_quality -- functor to compute the quality function
 @param[in]  compute_quality_diff -- functor to compute change in quality function
 @param[in]  initial_partition -- partition to start from
 @param[in]  optimal_partitions -- set of optimal_partitions at each level of the algorithm (initially empty)
 @param[in]  minimum_improve -- minimum_improvement necessary in each iteration / stopping criterion

 @param[out]  (double) quality of the optimal partition

 */
template<typename P, typename T, typename W, typename QF, typename QFDIFF>
double find_optimal_partition_louvain_gen( T& graph, W& weights, vec2 null_model_vec,
        QF compute_quality, QFDIFF compute_quality_diff, P initial_partition,
        std::vector<P>& optimal_partitions, double minimum_improve )
{

    typedef typename T::Node Node;
    typedef typename T::NodeIt NodeIt;
    
    // set up initial partitions
    P partition( initial_partition );

    double current_quality, old_quality;
    bool did_nodes_move = false;
    bool do_construct_new_graph = false;

    //clq::output( "\n\nFirst part of Louvain, current_quality", current_quality );
    LinearisedInternalsGeneric internals( graph, weights, partition, null_model_vec );

    current_quality = compute_quality( internals );
    //clq::output( "\n\nFirst part of Louvain after internals, current_quality", current_quality );
    //clq::print_partition_list( partition );
    //clq::output( "end partition list \n" );

    // Randomise the looping over nodes. You should nitialise random number generator
    // outside louvain when calling externally multiple times!
    // srand(std::time(0));
    std::vector<Node> nodes_ordered_randomly;

    for( NodeIt temp_node( graph ); temp_node != lemon::INVALID; ++temp_node ) {
        nodes_ordered_randomly.push_back( temp_node );
    }

    //clq::output( "Reshuffling ", lemon::countNodes( graph ), "Nodes" );
    std::random_shuffle( nodes_ordered_randomly.begin(), nodes_ordered_randomly.end() );
    //clq::output( "Reshuffling done" );

    do {
        // re-initialise quality and keep track of movements
        did_nodes_move = false;
        old_quality = current_quality;

        // loop over all nodes in random order
        for( auto n1_it = nodes_ordered_randomly.begin(); n1_it != nodes_ordered_randomly.end(); ++n1_it ) {

            // get node id and comm id
            Node n1 = *n1_it;
            unsigned int node_id = graph.id( n1 );
            unsigned int comm_id = partition.find_set( node_id );
            //clq::output( "NodeID: ", node_id, "CommID: ", comm_id, "Crash here?" );
            isolate_and_update_internals( graph, weights, n1, internals, partition );

            unsigned int best_comm  = find_best_comm_move( compute_quality_diff, node_id, internals, comm_id );

            insert_and_update_internals( graph, weights, n1, internals, partition, best_comm );

            // if there has been any move node
            if( best_comm != comm_id ) {
                did_nodes_move = true;
                do_construct_new_graph = true;
            }
        }

        if( did_nodes_move ) {
            current_quality = compute_quality( internals );
            //clq::output( "current quality: ", current_quality );
        }


    } while( ( current_quality - old_quality ) > minimum_improve );

    //clq::output( "Optimization round finished; current_quality", current_quality );
    //clq::print_partition_list( partition );
    //clq::output( "partition above; now starting second phase" );

    ////////////////////////////////////////////////////////////
    // Start Second phase - create reduced graph with self loops
    ////////////////////////////////////////////////////////////

    // 1) Normalise partition IDs and store next level in optimal partitions found by Louvain
    std::map<int, int> new_comm_id_to_old_comm_id = partition.normalise_ids();

    // 2) If there has actually been some movement, then we need to assemble a new graph
    if( do_construct_new_graph == true ) {
        // Compile P into original partition size. If there has been a move, then we 
        // want to store the intermediate result of the Louvain method. If there has been no move, then the
        // partition in the previous level was optimal.
        P partition_original_nodes = fold_partition_into_orginal_graph_size( optimal_partitions, partition );
        optimal_partitions.push_back( partition_original_nodes );
        //clq::output("Optimal partition: ", optimal_partitions.size());
        //clq::print_partition_list( partition );
        //clq::output( "renormalized partition above; now starting second phase" );
        
        //clq::output( "Constructing new graph" );
        // Create graph from partition
        T reduced_graph;
        W reduced_weights( reduced_graph );
        create_reduced_graph_from_partition( reduced_graph, reduced_weights, graph, weights, partition,
                                             new_comm_id_to_old_comm_id, internals );

        // get number of nodes in new reduced graph and initialise partition + new null model vectors
        unsigned int num_nodes_reduced_graph = lemon::countNodes( reduced_graph );
        P reduced_partition( num_nodes_reduced_graph );
        reduced_partition.initialise_as_singletons();
        // initialise reduced null model_vec
        auto reduced_null_model_vec = create_reduced_null_model_vec( new_comm_id_to_old_comm_id,internals,null_model_vec.size(),
                                      num_nodes_reduced_graph );

        //clq::output( "reduced null vectors" );
        //clq::print_2d_vector( reduced_null_model_vec );

        return find_optimal_partition_louvain_gen( reduced_graph,reduced_weights,reduced_null_model_vec,compute_quality,
                compute_quality_diff,reduced_partition,optimal_partitions, minimum_improve );
    } else {
        //clq::output( "Reached bottom", current_quality );
        if (optimal_partitions.size() == 0){
            optimal_partitions.push_back(partition);
        }
        return current_quality;
    }    
}


/**
 @brief  standard Louvain method - greedy algorithm to find community structure of a network.
 In contrast to the generalised variant no null model vectors are provided but they are 
 created directly from the data.

 @param[in]  graph    --    graph to partition
 @param[in]  weights  --    map of edge weights
 @param[in]  compute_quality -- functor to compute the quality function
 @param[in]  compute_quality_diff -- functor to compute change in quality function
 @param[in]  initial_partition -- partition to start from
 @param[in]  optimal_partitions -- set of optimal_partitions at each level of the algorithm (initially empty)
 @param[in]  minimum_improve -- minimum_improvement necessary in each iteration / stopping criterion

 @param[out]  (double) quality of the optimal partition

 */
template<typename P, typename T, typename W, typename QF, typename QFDIFF>
double find_optimal_partition_louvain( T& graph, W& weights, 
        QF compute_quality, QFDIFF compute_quality_diff, P initial_partition,
        std::vector<P>& optimal_partitions, double minimum_improve )
{

    typedef typename T::Node Node;
    typedef typename T::NodeIt NodeIt;
    
    // set up initial partitions
    P partition( initial_partition );
    P partition_init( initial_partition );

    double current_quality, old_quality;
    bool did_nodes_move = false;
    bool do_construct_new_graph = false;

    //clq::output( "\n\nFirst part of Louvain, current_quality", current_quality );
    if( !optimal_partitions.empty() ) {
        partition_init = optimal_partitions.back();
    }

    auto internals = clq::gen_internals( compute_quality, graph, weights, partition, partition_init);

    current_quality = compute_quality( internals );
    //clq::output( "\n\nFirst part of Louvain after internals, current_quality", current_quality );
    //clq::print_partition_list( partition );
    //clq::output( "end partition list \n" );

    // Randomise the looping over nodes. You should nitialise random number generator
    // outside louvain when calling externally multiple times!
    // srand(std::time(0));
    std::vector<Node> nodes_ordered_randomly;

    for( NodeIt temp_node( graph ); temp_node != lemon::INVALID; ++temp_node ) {
        nodes_ordered_randomly.push_back( temp_node );
    }

    //clq::output( "Reshuffling ", lemon::countNodes( graph ), "Nodes" );
    std::random_shuffle( nodes_ordered_randomly.begin(), nodes_ordered_randomly.end() );
    //clq::output( "Reshuffling done" );

    do {
        // re-initialise quality and keep track of movements
        did_nodes_move = false;
        old_quality = current_quality;

        // loop over all nodes in random order
        for( auto n1_it = nodes_ordered_randomly.begin(); n1_it != nodes_ordered_randomly.end(); ++n1_it ) {

            // get node id and comm id
            Node n1 = *n1_it;
            unsigned int node_id = graph.id( n1 );
            unsigned int comm_id = partition.find_set( node_id );
            //clq::output( "NodeID: ", node_id, "CommID: ", comm_id, "Crash here?" );
            isolate_and_update_internals( graph, weights, n1, internals, partition );

            unsigned int best_comm  = find_best_comm_move( compute_quality_diff, node_id, internals, comm_id );

            insert_and_update_internals( graph, weights, n1, internals, partition, best_comm );

            // if there has been any move node
            if( best_comm != comm_id ) {
                did_nodes_move = true;
                do_construct_new_graph = true;
            }
        }

        if( did_nodes_move ) {
            current_quality = compute_quality( internals );
            //clq::output( "current quality: ", current_quality );
        }


    } while( ( current_quality - old_quality ) > minimum_improve );

    //clq::output( "Optimization round finished; current_quality", current_quality );
    //clq::print_partition_list( partition );
    //clq::output( "partition above; now starting second phase" );

    ////////////////////////////////////////////////////////////
    // Start Second phase - create reduced graph with self loops
    ////////////////////////////////////////////////////////////

    // 1) Normalise partition IDs and store next level in optimal partitions found by Louvain
    std::map<int, int> new_comm_id_to_old_comm_id = partition.normalise_ids();

    // 2) If there has actually been some movement, then we need to assemble a new graph
    if( do_construct_new_graph == true ) {
        // Compile P into original partition size. If there has been a move, then we 
        // want to store the intermediate result of the Louvain method. If there has been no move, then the
        // partition in the previous level was optimal.
        P partition_original_nodes = fold_partition_into_orginal_graph_size( optimal_partitions, partition );
        optimal_partitions.push_back( partition_original_nodes );
        //clq::output("Optimal partition: ", optimal_partitions.size());
        //clq::print_partition_list( partition );
        //clq::output( "renormalized partition above; now starting second phase" );
        
        //clq::output( "Constructing new graph" );
        // Create graph from partition
        T reduced_graph;
        W reduced_weights( reduced_graph );
        create_reduced_graph_from_partition( reduced_graph, reduced_weights, graph, weights, partition,
                                             new_comm_id_to_old_comm_id, internals );

        // get number of nodes in new reduced graph and initialise partition + new null model vectors
        unsigned int num_nodes_reduced_graph = lemon::countNodes( reduced_graph );
        P reduced_partition( num_nodes_reduced_graph );
        reduced_partition.initialise_as_singletons();

        //clq::output( "reduced null vectors" );
        //clq::print_2d_vector( reduced_null_model_vec );

        return find_optimal_partition_louvain( reduced_graph,reduced_weights,compute_quality,
                compute_quality_diff,reduced_partition,optimal_partitions, minimum_improve );
    } else {
        //clq::output( "Reached bottom", current_quality );
        if (optimal_partitions.size() == 0){
            optimal_partitions.push_back(partition);
        }
        return current_quality;
    }    
}


}// end namespace..
