#pragma once

#include <vector>
#include <set>
#include <map>

namespace clq
{

/**
 @brief  A simple data structure for a partition, where the position in the
 vector denotes the node_id and its value determines its set id.

 Good when you need constant time assignment of a node to a set and removal
 of a node from a set.
 */

class VectorPartition
{
private:
    int num_nodes;
    std::vector<int> partition_vector;
    bool is_normalised;

public:

    //#################### CONSTRUCTORS ####################
    // construct empty partition
    explicit VectorPartition (int num_nodes) :
        num_nodes (num_nodes),
        partition_vector (std::vector<int> (num_nodes, -1) ),
        is_normalised (false)
    {
    }

    // construct partition with initial_set
    explicit VectorPartition (int num_nodes, int initial_set) :
        num_nodes (num_nodes), partition_vector (std::vector<int> (num_nodes,
                initial_set) ), is_normalised (false)
    {
    }

    // construct partition from vector
    explicit VectorPartition (std::vector<int> partition) :
        num_nodes (partition.size() ), partition_vector (partition),
        is_normalised (false)
    {
    }

    // initialise partition as singleton -- with each node in its own group
    void initialise_as_singletons()
    {
        int i = 0;

        for (auto itr = partition_vector.begin(); itr != partition_vector.end(); ++itr) {
            *itr = i++; 
        }

        is_normalised = true;
    }

    // initialise as the global "all in one" partition
    void initialise_as_global()
    {
        partition_vector = std::vector<int> (partition_vector.size(), 0);
    }

    //#################### PUBLIC METHODS ####################
    // return community ID
    int find_set (int node_id) const
    {
        return partition_vector[node_id];
    }
    
    // Use to temporarily assign node to "void" community with index -1
    void isolate_node (int node_id)
    {
        partition_vector[node_id] = -1;
        is_normalised = false;
    }

    // move node to community with id set_id
    void add_node_to_set (int node_id, int set_id)
    {
        partition_vector[node_id] = set_id;
        is_normalised = false;
    }

    // count number of nodes
    int element_count() const
    {
        return partition_vector.size();
    }
	
    // count number of distinct communities in partition vector
    int set_count()
    {
        std::set<int> seen_nodes;
    
        for (auto itr = partition_vector.begin(); itr != partition_vector.end(); ++itr) {
            if (*itr != -1) {
                seen_nodes.insert (*itr);
            }
        }
    
        return seen_nodes.size();
    }

    // return vector with all nodes that are part of community set_id
    std::vector<int> get_nodes_from_set (int set_id)
    {
        std::vector<int> nodes_in_set;

        for (int i = 0; i < num_nodes; ++i) {
            if (partition_vector[i] == set_id) {
                nodes_in_set.push_back (i);
            }
        }

        return nodes_in_set;
    }

    // return the internal vector of IDs
    std::vector<int> return_partition_vector()
    {
        return partition_vector;
    }

    // Normalise IDs in partition vector.
    // Modifies IDs such that they are contiguous and start at 0
    // e.g. 2,1,4,2 -> 0,1,2,0
    // returns map from new IDs to previous IDs.
    std::map<int, int> normalise_ids()
    {
        // Mapping from new set ids to old set ids
        std::map<int, int> set_new_to_old;

        // Check if already normalised, if yes contruct identity mapping
        // and return, if node normalise and construct mapping
        if (!is_normalised) {
            int start_num = 0;
            std::map<int, int> set_old_to_new;

            // For every element, make a map
            for (auto itr = partition_vector.begin(); itr != partition_vector.end(); ++itr) {

                // Find current node ID in old to new mapping
                std::map<int, int>::iterator old_set = set_old_to_new.find (*itr);
                
                // if not present in mapping (mind the order!) 
                // a) update mappings
                // b) assign node ID to new contiguous ID
                // c) update counter
                if (old_set == set_old_to_new.end() ) {
                    set_old_to_new[*itr] = start_num;
                    set_new_to_old[start_num] = *itr;

                    *itr = start_num;
                    start_num++;
                
                // already found in mapping -- just assign to new ID
                } else {
                    *itr = old_set->second;
                }
            }
            
            // set state variable and return..
            is_normalised = true;
            return set_new_to_old;

        } else {

            // Still need to reconstruct a new to old mapping even if it is 
            // the identity (since we don't store it)
            for (auto itr = partition_vector.begin(); itr != partition_vector.end(); ++itr) {
                set_new_to_old[*itr] = *itr;
            }

            return set_new_to_old;
        }
    }


    //#################### Operators ####################
    bool operator== (const VectorPartition& other) const
    {
        if (this->is_normalised && other.is_normalised) {
            return (other.partition_vector == this->partition_vector);

        } else {
            VectorPartition a (*this);
            VectorPartition b (other);

            a.normalise_ids();
            b.normalise_ids();
            return a.partition_vector == b.partition_vector;
        }
    }

    bool operator!= (const VectorPartition& other) const
    {
        return ! (partition_vector == other.partition_vector);
    }
};

}
