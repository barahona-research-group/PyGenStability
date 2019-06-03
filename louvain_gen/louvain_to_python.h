#include "stability_gen.h"
#include "louvain_gen.h"
#include "vector_partition.h"
#include "io.h"

#include <lemon/smart_graph.h>
#include <vector>

std::pair<double, vector<int>> run_louvain(int* from, int* to, double* w, int n_edges, double* null_model_input, int num_null_vectors, double time) {


    // initialise markov time from input
    double current_markov_time = 1;
    current_markov_time = time;
    //clq::output("Markov time: ", time);

    //initialise graph from input
    lemon::SmartGraph input_graph;
    lemon::SmartGraph::EdgeMap<double> input_graph_weights(input_graph);
    

    int max_node_id_seen = -1;
    for (int i=0; i<n_edges;++i){
        int node1_id = from[i];
        int node2_id = to[i];
        double weight = w[i];
        
        //std::cout << "Read Line: " << node1_id << " " << node2_id << " " << weight << " " << std::endl;
        
        if (node1_id > max_node_id_seen) {
            int difference = node1_id - max_node_id_seen;
            for (int i=0; i<difference; ++i) {
                input_graph.addNode();
            }
            max_node_id_seen = node1_id;
        }

        if (node2_id > max_node_id_seen) {
            int difference = node2_id - max_node_id_seen;
            for (int i=0; i<difference; ++i) {
                input_graph.addNode();
            }
            max_node_id_seen = node2_id;
        }

        input_graph_weights.set(input_graph.addEdge(input_graph.nodeFromId(node1_id), 
                                                    input_graph.nodeFromId(node2_id)
                                                    ), 
                                                    weight);
    }
    
    
    int num_nodes = lemon::countNodes(input_graph);
    //std::cout << "Num nodes: " << num_nodes <<std::endl;
    
    //initialise null model
    std::vector<std::vector<double>> null_model (num_null_vectors, std::vector<double> (num_nodes, 0) );
    for (int i = 0; i < num_null_vectors; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            null_model[i][j] = null_model_input[i*(num_nodes)+j];
        }
    }
    //clq::print_2d_vector(null_model);

    // define type for vector partition initialise start partition, qualitiy functions etc.
    typedef clq::VectorPartition partition;

    partition start_partition(lemon::countNodes(input_graph));
    start_partition.initialise_as_singletons();
    //clq::print_partition_list(start_partition);

    clq::find_linearised_generic_stability quality(current_markov_time);
    clq::linearised_generic_stability_gain quality_gain(current_markov_time);
    
    std::vector<partition> optimal_partitions;

    // call Louvain
    //clq::output("Start Louvain");
    double stability = clq::find_optimal_partition_louvain_gen<partition>(
                           input_graph, input_graph_weights, null_model, quality, quality_gain,
                           start_partition, optimal_partitions, 1e-18);
    
    // display final result
    partition best_partition = optimal_partitions.back();
    //clq::print_partition_list(best_partition);
    //clq::output("Stability: ", stability);
    //std::cout << best_partition.find_set(0) << std::endl;
    
    //make a pair of stability and partitions
    std::pair<double, vector<int>> output;
    output = std::make_pair(stability,  best_partition.return_partition_vector());  
    
    //return it
    return output;
}
