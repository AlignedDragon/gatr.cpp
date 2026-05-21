#include <torch/extension.h>
#include "pga_ops.h"

PYBIND11_MODULE(_opt_ops, m) {
    m.doc() = "ezgatr.opt — C++ implementations of PGA primitives";

    m.def("geometric_product", &ezgatr::opt::geometric_product,
          py::arg("x"), py::arg("y"),
          "Geometric product of two multi-vectors of shape (..., 16).");

    m.def("geometric_product_dense", &ezgatr::opt::geometric_product_dense,
          py::arg("x"), py::arg("y"),
          "Dense triple-loop geometric product (pre-optimization baseline).");

    m.def("geometric_product_sparse_rt", &ezgatr::opt::geometric_product_sparse_rt,
          py::arg("x"), py::arg("y"),
          "Sparse-but-non-unrolled geometric product (zero-mul elimination only).");

    m.def("equi_join", &ezgatr::opt::equi_join,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join of two multi-vectors of shape (..., 16).");

    m.def("equi_join_dense", &ezgatr::opt::equi_join_dense,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Dense triple-loop equivariant join (pre-optimization baseline).");

    m.def("equi_join_sparse_rt", &ezgatr::opt::equi_join_sparse_rt,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Sparse-but-non-unrolled equivariant join (zero-mul elimination only).");

    m.def("geometric_product_ilp2", &ezgatr::opt::geometric_product_ilp2,
          py::arg("x"), py::arg("y"),
          "Geometric product with K=2 accumulators per blade in a single inlined block.");

    m.def("geometric_product_ilp4", &ezgatr::opt::geometric_product_ilp4,
          py::arg("x"), py::arg("y"),
          "Geometric product with K=4 accumulators per blade in a single inlined block.");

    m.def("equi_join_ilp2", &ezgatr::opt::equi_join_ilp2,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join with K=2 accumulators per blade in a single inlined block.");

    m.def("equi_join_ilp4", &ezgatr::opt::equi_join_ilp4,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join with K=4 accumulators per blade in a single inlined block.");
}
