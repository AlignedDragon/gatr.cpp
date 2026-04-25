#include <torch/extension.h>
#include "pga_ops.h"

PYBIND11_MODULE(_opt_ops, m) {
    m.doc() = "ezgatr.opt — C++ implementations of PGA primitives";

    m.def("geometric_product", &ezgatr::opt::geometric_product,
          py::arg("x"), py::arg("y"),
          "Geometric product of two multi-vectors of shape (..., 16).");

    m.def("equi_join", &ezgatr::opt::equi_join,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join of two multi-vectors of shape (..., 16).");
}
