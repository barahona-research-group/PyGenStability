#pragma once
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <set>
#include <lemon/core.h>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

namespace clq {
//Variadic template for generic output ala printf();
//as variadic templates are recursive, must have a base case
void output() {
    std::cout << '\n';
}

//Output function to output any type without type specifiers like printf() family
template <typename T, typename ...P>
void output(T t, P ...p)
{
    std::cout << t << ' ';
    if (sizeof...(p)) {
        output(p...);
    }
    else {
        std::cout << '\n';
    }
}


template<class T>
void print_collection(T collection) {
    for (class T::iterator itr = collection.begin(); itr != collection.end(); ++itr) {
        std::cout << *itr << ", ";
    }
    std::cout << std::endl;
}

template<class T>
void print_collection(T collection, int new_line) {
    int i =0;
    for (class T::iterator itr = collection.begin(); itr != collection.end(); ++itr) {
        if (i % new_line == 0) {
            std::cout << "\n";
        }
        std::cout << *itr << ", ";
        ++i;

    }
    std::cout << std::endl;
}

//TODO: OPEN ISSUE -- formatting of prints; how many digits etc -- also relevant for output files..
//============================================================================================
// PRINT_MATRIX
// Template to print out linear (array/vector) in "matrix" format assuming row first ordering,
// i.e., A(i,j) corresponds to matrix[i+j*N], where N is the dimension of the matrix.
//
// INPUTS:  matrix -- linear container (array/vector)
//          lda -- dimension of the matrix
//============================================================================================
template <typename T>
void print_matrix(T matrix, int lda) {
    for (int i=0; i<lda; ++i) {
        for (int j=0; j<lda; ++j) {
            std::cout << matrix[j+i*lda] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void print_2d_vector(std::vector<std::vector<T> > my_vector) {
    typename std::vector<std::vector<T> >::iterator itr;
    for (itr = my_vector.begin(); itr != my_vector.end(); ++itr) {
        typename std::vector<T>::iterator new_itr;
        for (new_itr = itr->begin(); new_itr != itr->end(); ++new_itr) {
            std::cout << (*new_itr) << " ";
        }
         std::cout << "\n";
    }
}


/**
  Write partitions from a container into a file. (Template)

 This functions iterates over a given container and writes the partitions into 
 a file. Each line stands for one partition, each column stands for one node 
 with its corresponding community Id.

 */
template <typename P>
void partitions_to_file(std::string filename,
        P & all_partitions) {
    // init streams
    std::ofstream partitions_file;
    partitions_file.open(filename);
    
    // iterate over container and write partitions in file
    for (auto itr = all_partitions.begin();
        itr != all_partitions.end();++itr) {

        int length = itr->element_count();
        for (int i = 0; i < length; i++) {
            partitions_file << itr->find_set(i) << " ";
        }

        partitions_file << std::endl;
    }
    partitions_file.close();
}

template <typename P>
void stability_to_file(std::string filename,
      P &  stability) {
    // init streams
    std::ofstream stability_file;
    stability_file.open(filename);
    
    stability_file << stability << std::endl;

    stability_file.close();
}


template<typename P>
void print_partition_list(P &partition) {
    int length = partition.element_count();
    for (int i = 0; i < length; i++) {
        std::cout << i << "->" << partition.find_set(i) << std::endl;
    }
}



//TODO WRITE DESCRIPTION
std::vector<double> read_edgelist_weighted(std::string filename) {
    // initialise input stream and strings for readout
    std::ifstream my_file(filename.c_str());
    std::string line;
    std::string mystring;

    // check if file is open
    if (!my_file.is_open()) {
        std::cout << "couldn't open file:" << filename << std::endl;
        exit(1);
    }

    // keep track of size of graph and create internal adjacency list
    // NB: Node numbering starts with 0!
    int max_node_id_seen = -1;
    std::vector<int> from, to;
    std::vector<double> weight;

    //readout contents from my_file into string, line by line
    while (std::getline(my_file, line)) {

        std::stringstream lineStream(line);
        //readout node id and weights
        std::getline(lineStream, mystring, ' ');
        int node1_id = atoi(mystring.c_str());
        from.push_back(node1_id);

        std::getline(lineStream, mystring, ' ');
        int node2_id = atoi(mystring.c_str());
        to.push_back(node2_id);

        std::getline(lineStream, mystring, ' ');
        double edge_weight = atof(mystring.c_str());
        weight.push_back(edge_weight);

        if (node1_id > max_node_id_seen) {
            max_node_id_seen = node1_id;
        }

        if (node2_id > max_node_id_seen) {
            max_node_id_seen = node2_id;
        }
    }
    // don't forget to close file after readout...
    my_file.close();

    // now write adjecency matrix in vector form (row first ordering)
    std::vector<double> Adj((max_node_id_seen+1)*(max_node_id_seen+1),0);
    for (unsigned int i =0; i<to.size(); i++) {
        int index = from[i]*(max_node_id_seen+1)+to[i];
        Adj[index] = weight[i];
    }

    return Adj;
}

//TODO WRITE DESCRIPTION
template<typename G, typename E>
bool read_edgelist_weighted_graph(std::string filename, G &graph, E &weights) {
    // initialise input stream and strings for readout
    std::ifstream my_file(filename.c_str());
    std::string line;
    std::string mystring;

    // check if file is open
    if (!my_file.is_open()) {
        std::cout << "couldn't open file:" << filename << std::endl;
        exit(1);
    }

    // define Node class for convenience
    typedef typename G::Node Node;

    // keep track of graph size
    // NB: numbering starts with 0!
    int max_node_id_seen = -1;

    //readout contents from my_file into string, line by line
    while (std::getline(my_file, line)) {
        std::stringstream lineStream(line);

        //readout node id and weights
        std::getline(lineStream, mystring, ' ');
        int node1_id = atoi(mystring.c_str());
        std::getline(lineStream, mystring, ' ');
        int node2_id = atoi(mystring.c_str());
        std::getline(lineStream, mystring, ' ');
        double weight = atof(mystring.c_str());

        //std::cout << "Read Line: " << node1_id << " " << node2_id << " " << weight << " " << std::endl;
        if (node1_id > max_node_id_seen) {
            int difference = node1_id - max_node_id_seen;
            for (int i=0; i<difference; ++i) {
                graph.addNode();
            }
            max_node_id_seen = node1_id;
        }

        if (node2_id > max_node_id_seen) {
            int difference = node2_id - max_node_id_seen;
            for (int i=0; i<difference; ++i) {
                graph.addNode();
            }
            max_node_id_seen = node2_id;
        }

        //std::cout << max_node_id_seen << std::endl;
        Node node1 = graph.nodeFromId(node1_id);
        Node node2 = graph.nodeFromId(node2_id);

        typename G::Edge edge = graph.addEdge(node1, node2);
        weights.set(edge, weight);
    }

    my_file.close();
    return true;
}
//TODO WRITE DESCRIPTION
std::vector<std::vector<double> > read_null_model (std::string filename, int num_nodes)
{
    // initialise input stream and strings for readout
    std::ifstream my_file (filename.c_str() );
    std::string line;
    std::string mystring;

    // check if file is open
    if (!my_file.is_open() ) {
        std::cout << "couldn't open file:" << filename << std::endl;
        exit (1);
    }

    // find number of input vectors and initialise empty null model vectors
    std::getline (my_file, line);
    // number of vectors is numbers of spaces + 1...
    int num_null_vectors = std::count (line.begin(), line.end(), ' ') + 1;

    if (* (line.end() - 1) == ' ') {
        clq::output ("You provided ", num_null_vectors-1, " vectors as null model terms.\n",
                    "There should be an even number!");
        exit (1);
    }

    std::vector<std::vector<double>> null_model (num_null_vectors, std::vector<double> (num_nodes, 0) );

    int j = 0;

    //readout contents from my_file into string, line by line
    do {
        // create stream object of line to loop over
        std::stringstream lineStream (line);

        for (int i = 0; i < num_null_vectors; ++i) {

            std::getline (lineStream, mystring, ' ');
            double null_model_entry = atof (mystring.c_str() );
            null_model[i][j] = null_model_entry;
        }

        j = j + 1;

    } while (std::getline (my_file, line) );

    // don't forget to close file after readout...
    my_file.close();


    return null_model;
}


//TODO WRITE DESCRIPTION
void write_adj_matrix(std::string filename, std::vector<double> matrix) {
    // initialise input stream and strings for readout
    std::ofstream my_file(filename.c_str());

    // check if file is open
    if (!my_file.is_open()) {
        std::cout << "couldn't open file" << std::endl;
        exit(1);
    }
    int lda = std::sqrt(matrix.size());

    for (int i=0; i<lda; ++i) {
        for (int j=0; j<lda; ++j) {
            my_file << matrix[j+i*lda] << "\t";
        }
        my_file << "\n";
    }
    my_file << "\n";
    my_file.close();

}

//TODO WRITE DESCRIPTION
template<typename G, typename E>
void write_edgelist_weighted_graph(std::string filename, G &graph, E &weights) {
    // initialise input stream and strings for readout
    std::ofstream my_file(filename.c_str());
    std::string mystring;

    // check if file is open
    if (!my_file.is_open()) {
        std::cout << "couldn't open file" << std::endl;
        exit(1);
    }

    for(typename G::EdgeIt e(graph); e!=lemon::INVALID; ++e) {
        int node1 = graph.id(graph.u(e));
        int node2 = graph.id(graph.v(e));
        double weight = weights[e];

        my_file << node1 << " " << node2 << " " << weight << "\n";
    }

    my_file.close();
}

}
