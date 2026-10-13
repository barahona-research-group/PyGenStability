#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>

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

namespace {

// C-contiguous arrays of the expected dtype (pybind11 converts/copies otherwise)
using int_array = py::array_t<int, py::array::c_style | py::array::forcecast>;
using double_array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Build the graph and null model from the python inputs.
//
// The number of nodes is taken from the null model (shape (num_null_vectors,
// num_nodes)), not from the largest node id in the edge list, so that nodes
// without any quality entry (e.g. the last node) are not silently dropped.
int build_graph_and_null_model(const int_array &from_arr,
                               const int_array &to_arr,
                               const double_array &w_arr, int n_edges,
                               const double_array &null_model_input_arr,
                               int num_null_vectors,
                               lemon::SmartGraph &input_graph,
                               lemon::SmartGraph::EdgeMap<double> &weights,
                               clq::vec2 &null_model) {
  if (null_model_input_arr.ndim() != 2 ||
      null_model_input_arr.shape(0) != num_null_vectors) {
    throw py::value_error(
        "null_model must have shape (num_null_vectors, num_nodes)");
  }
  if (num_null_vectors % 2 != 0) {
    throw py::value_error("null model vectors must be provided as pairs");
  }
  if (n_edges < 0 || from_arr.size() < n_edges || to_arr.size() < n_edges ||
      w_arr.size() < n_edges) {
    throw py::value_error("edge arrays are shorter than n_edges");
  }
  const int num_nodes = static_cast<int>(null_model_input_arr.shape(1));

  input_graph.reserveNode(num_nodes);
  input_graph.reserveEdge(n_edges);
  for (int i = 0; i < num_nodes; ++i) {
    input_graph.addNode();
  }

  const int *from = from_arr.data();
  const int *to = to_arr.data();
  const double *w = w_arr.data();
  for (int i = 0; i < n_edges; ++i) {
    if (from[i] < 0 || from[i] >= num_nodes || to[i] < 0 ||
        to[i] >= num_nodes) {
      throw py::value_error("edge node id out of range of the null model");
    }
    weights.set(input_graph.addEdge(input_graph.nodeFromId(from[i]),
                                    input_graph.nodeFromId(to[i])),
                w[i]);
  }

  const double *null_model_input = null_model_input_arr.data();
  null_model.assign(num_null_vectors, std::vector<double>(num_nodes, 0));
  for (int i = 0; i < num_null_vectors; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      null_model[i][j] = null_model_input[i * num_nodes + j];
    }
  }
  return num_nodes;
}

} // namespace

double evaluate_quality(int_array from_arr, int_array to_arr,
                        double_array w_arr, int n_edges,
                        double_array null_model_input_arr,
                        int num_null_vectors, double time,
                        int_array partition_arr) {
  lemon::SmartGraph input_graph;
  lemon::SmartGraph::EdgeMap<double> input_graph_weights(input_graph);
  clq::vec2 null_model;
  int num_nodes = build_graph_and_null_model(
      from_arr, to_arr, w_arr, n_edges, null_model_input_arr,
      num_null_vectors, input_graph, input_graph_weights, null_model);

  if (partition_arr.size() != num_nodes) {
    throw py::value_error("partition size does not match number of nodes");
  }
  const int *partition_vec = partition_arr.data();
  std::vector<int> part_test(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    if (partition_vec[i] < 0 || partition_vec[i] >= num_nodes) {
      throw py::value_error("partition ids must be in [0, num_nodes)");
    }
    part_test[i] = partition_vec[i];
  }

  typedef clq::VectorPartition partition;
  partition start_partition(part_test);

  clq::find_linearised_generic_stability quality(time);

  return clq::find_stability<partition>(input_graph, input_graph_weights,
                                        null_model, quality, start_partition,
                                        1e-18);
}

std::pair<double, std::vector<int>>
run_louvain(int_array from_arr, int_array to_arr, double_array w_arr,
            int n_edges, double_array null_model_input_arr,
            int num_null_vectors, double time, unsigned int seed) {

  std::mt19937 rng(seed);

  lemon::SmartGraph input_graph;
  lemon::SmartGraph::EdgeMap<double> input_graph_weights(input_graph);
  clq::vec2 null_model;
  int num_nodes = build_graph_and_null_model(
      from_arr, to_arr, w_arr, n_edges, null_model_input_arr,
      num_null_vectors, input_graph, input_graph_weights, null_model);

  typedef clq::VectorPartition partition;

  partition start_partition(num_nodes);
  start_partition.initialise_as_singletons();

  clq::find_linearised_generic_stability quality(time);
  clq::linearised_generic_stability_gain quality_gain(time);

  std::vector<partition> optimal_partitions;

  // call Louvain
  double stability = clq::find_optimal_partition_louvain_gen<partition>(
      input_graph, input_graph_weights, null_model, quality, quality_gain,
      start_partition, optimal_partitions, 1e-18, rng);

  partition best_partition = optimal_partitions.back();

  return std::make_pair(stability, best_partition.return_partition_vector());
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
