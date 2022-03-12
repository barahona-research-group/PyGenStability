#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <lemon/smart_graph.h>

#include "../generalizedLouvain/CPP/cliques/io.h"
#include "../generalizedLouvain/CPP/cliques/louvain_gen.h"
#include "../generalizedLouvain/CPP/cliques/stability_gen.h"
#include "../generalizedLouvain/CPP/cliques/vector_partition.h"

namespace py = pybind11;

namespace clq {

template <typename P, typename T, typename W, typename QF>
double find_stability(T &graph, W &weights, vec2 null_model_vec,
                      QF compute_quality, P initial_partition,
                      double minimum_improve) {

  typedef typename T::Node Node;
  typedef typename T::NodeIt NodeIt;

  // set up initial partitions
  P partition(initial_partition);

  double current_quality;

  LinearisedInternalsGeneric internals(graph, weights, partition,
                                       null_model_vec);

  current_quality = compute_quality(internals);

  return current_quality;
}
} // namespace clq

double evaluate_quality(py::array_t<int> from_arr, py::array_t<int> to_arr,
                        py::array_t<double> w_arr, int n_edges,
                        py::array_t<double> null_model_input_arr,
                        int num_null_vectors, double time,
                        py::array_t<int> partition_arr) {

  // convert input arrys tto pointers
  py::buffer_info info_from = from_arr.request();
  auto from = static_cast<int *>(info_from.ptr);

  py::buffer_info info_to = to_arr.request();
  auto to = static_cast<int *>(info_to.ptr);

  py::buffer_info info_w = w_arr.request();
  auto w = static_cast<double *>(info_w.ptr);

  py::buffer_info info_null_model_input = null_model_input_arr.request();
  auto null_model_input = static_cast<double *>(info_null_model_input.ptr);

  py::buffer_info info_partition = partition_arr.request();
  auto partition_vec = static_cast<int *>(info_partition.ptr);

  // initialise markov time from input
  double current_markov_time = 1;
  current_markov_time = time;

  // initialise graph from input
  lemon::SmartGraph input_graph;
  lemon::SmartGraph::EdgeMap<double> input_graph_weights(input_graph);

  int max_node_id_seen = -1;
  for (int i = 0; i < n_edges; ++i) {
    int node1_id = from[i];
    int node2_id = to[i];
    double weight = w[i];

    if (node1_id > max_node_id_seen) {
      int difference = node1_id - max_node_id_seen;
      for (int i = 0; i < difference; ++i) {
        input_graph.addNode();
      }
      max_node_id_seen = node1_id;
    }

    if (node2_id > max_node_id_seen) {
      int difference = node2_id - max_node_id_seen;
      for (int i = 0; i < difference; ++i) {
        input_graph.addNode();
      }
      max_node_id_seen = node2_id;
    }

    input_graph_weights.set(
        input_graph.addEdge(input_graph.nodeFromId(node1_id),
                            input_graph.nodeFromId(node2_id)),
        weight);
  }

  int num_nodes = lemon::countNodes(input_graph);

  // initialise null model
  std::vector<std::vector<double>> null_model(
      num_null_vectors, std::vector<double>(num_nodes, 0));
  for (int i = 0; i < num_null_vectors; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      null_model[i][j] = null_model_input[i * (num_nodes) + j];
    }
  }

  // define type for vector partition initialise start partition, qualitiy
  // functions etc.
  typedef clq::VectorPartition partition;

  std::vector<int> part_test(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    part_test[i] = partition_vec[i];
  }

  partition start_partition(part_test);

  clq::find_linearised_generic_stability quality(current_markov_time);

  double stability = clq::find_stability<partition>(
      input_graph, input_graph_weights, null_model, quality, start_partition,
      1e-18);

  return stability;
}

std::pair<double, std::vector<int>>
run_louvain(py::array_t<int> from_arr, py::array_t<int> to_arr,
            py::array_t<double> w_arr, int n_edges,
            py::array_t<double> null_model_input_arr, int num_null_vectors,
            double time) {

  // convert input arrays to pointers
  py::buffer_info info_from = from_arr.request();
  auto from = static_cast<int *>(info_from.ptr);

  py::buffer_info info_to = to_arr.request();
  auto to = static_cast<int *>(info_to.ptr);

  py::buffer_info info_w = w_arr.request();
  auto w = static_cast<double *>(info_w.ptr);

  py::buffer_info info_null_model_input = null_model_input_arr.request();
  auto null_model_input = static_cast<double *>(info_null_model_input.ptr);

  // initialise markov time from input
  double current_markov_time = 1;
  current_markov_time = time;

  // initialise graph from input
  lemon::SmartGraph input_graph;
  lemon::SmartGraph::EdgeMap<double> input_graph_weights(input_graph);

  int max_node_id_seen = -1;
  for (int i = 0; i < n_edges; ++i) {
    int node1_id = from[i];
    int node2_id = to[i];
    double weight = w[i];

    if (node1_id > max_node_id_seen) {
      int difference = node1_id - max_node_id_seen;
      for (int i = 0; i < difference; ++i) {
        input_graph.addNode();
      }
      max_node_id_seen = node1_id;
    }

    if (node2_id > max_node_id_seen) {
      int difference = node2_id - max_node_id_seen;
      for (int i = 0; i < difference; ++i) {
        input_graph.addNode();
      }
      max_node_id_seen = node2_id;
    }

    input_graph_weights.set(
        input_graph.addEdge(input_graph.nodeFromId(node1_id),
                            input_graph.nodeFromId(node2_id)),
        weight);
  }

  int num_nodes = lemon::countNodes(input_graph);

  // initialise null model
  std::vector<std::vector<double>> null_model(
      num_null_vectors, std::vector<double>(num_nodes, 0));
  for (int i = 0; i < num_null_vectors; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      null_model[i][j] = null_model_input[i * (num_nodes) + j];
    }
  }

  // define type for vector partition initialise start partition, qualitiy
  // functions etc.
  typedef clq::VectorPartition partition;

  partition start_partition(lemon::countNodes(input_graph));
  start_partition.initialise_as_singletons();

  clq::find_linearised_generic_stability quality(current_markov_time);
  clq::linearised_generic_stability_gain quality_gain(current_markov_time);

  std::vector<partition> optimal_partitions;

  // call Louvain
  double stability = clq::find_optimal_partition_louvain_gen<partition>(
      input_graph, input_graph_weights, null_model, quality, quality_gain,
      start_partition, optimal_partitions, 1e-18);

  partition best_partition = optimal_partitions.back();

  // make a pair of stability and partitions
  std::pair<double, std::vector<int>> output;
  output = std::make_pair(stability, best_partition.return_partition_vector());

  return output;
}

PYBIND11_MODULE(generalized_louvain, m) {
  m.doc() = R"pbdoc(
        Pybind11 binding of Generalized Louvain
        -----------------------------------------

        .. currentmodule:: pygenstability

        .. autosummary::
           :toctree: _generate

	run_louvain
    )pbdoc";

  m.def("run_louvain", &run_louvain, R"pbdoc(
	Run generalized Louvain once
    )pbdoc");

  m.def("evaluate_quality", &evaluate_quality, R"pbdoc(
    Evaluate the quality of a partition
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
