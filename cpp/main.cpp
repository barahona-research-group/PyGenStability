#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../generalizedLouvain/CPP/cliques/stability_gen.h"
#include "../generalizedLouvain/CPP/cliques/stability_gen.h"
#include "../generalizedLouvain/CPP/cliques/louvain_gen.h"
#include "../generalizedLouvain/CPP/cliques/vector_partition.h"
#include "../generalizedLouvain/CPP/cliques/io.h"


int add(int i, int j) {
    return i + j;
}

int mult(double x, double y) {
    return x*y;
}

namespace py = pybind11;

PYBIND11_MODULE(cpp, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("mult", py::vectorize(mult));

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
