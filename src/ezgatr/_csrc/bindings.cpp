#include <torch/extension.h>
#include "pga_ops.h"

PYBIND11_MODULE(_opt_ops, m) {
    m.doc() = "ezgatr.opt — C++ implementations of PGA primitives";

    m.def("geometric_product", &ezgatr::opt::geometric_product,
          py::arg("x"), py::arg("y"),
          "Geometric product of two multi-vectors of shape (..., 16).");

    m.def("geometric_product_v0", &ezgatr::opt::geometric_product_v0,
          py::arg("x"), py::arg("y"),
          "Dense triple-loop geometric product (pre-optimization baseline).");

    m.def("geometric_product_v1", &ezgatr::opt::geometric_product_v1,
          py::arg("x"), py::arg("y"),
          "Sparse-but-non-unrolled geometric product (zero-mul elimination only).");

    m.def("geometric_product_v2", &ezgatr::opt::geometric_product_v2,
          py::arg("x"), py::arg("y"),
          "Per-blade unrolled geometric product (base unrolled kernel).");

    m.def("equi_join", &ezgatr::opt::equi_join,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join of two multi-vectors of shape (..., 16).");

    m.def("equi_join_v0", &ezgatr::opt::equi_join_v0,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Dense triple-loop equivariant join (pre-optimization baseline).");

    m.def("equi_join_v1", &ezgatr::opt::equi_join_v1,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Sparse-but-non-unrolled equivariant join (zero-mul elimination only).");

    m.def("equi_join_v2", &ezgatr::opt::equi_join_v2,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Per-blade unrolled equivariant join (base unrolled kernel).");

    m.def("geometric_product_v2_3", &ezgatr::opt::geometric_product_v2_3,
          py::arg("x"), py::arg("y"),
          "Geometric product with K=2 accumulators per blade in a single inlined block.");

    m.def("geometric_product_v2_4", &ezgatr::opt::geometric_product_v2_4,
          py::arg("x"), py::arg("y"),
          "Geometric product with K=4 accumulators per blade in a single inlined block.");

    m.def("equi_join_v2_3", &ezgatr::opt::equi_join_v2_3,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join with K=2 accumulators per blade in a single inlined block.");

    m.def("equi_join_v2_4", &ezgatr::opt::equi_join_v2_4,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join with K=4 accumulators per blade in a single inlined block.");

    m.def("geometric_product_v2_1", &ezgatr::opt::geometric_product_v2_1,
          py::arg("x"), py::arg("y"),
          "Per-blade unrolled geometric product with K=2 internal accumulators.");

    m.def("geometric_product_v2_2", &ezgatr::opt::geometric_product_v2_2,
          py::arg("x"), py::arg("y"),
          "Per-blade unrolled geometric product with K=4 internal accumulators.");

    m.def("equi_join_v2_1", &ezgatr::opt::equi_join_v2_1,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Per-blade unrolled equivariant join with K=2 internal accumulators.");

    m.def("equi_join_v2_2", &ezgatr::opt::equi_join_v2_2,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Per-blade unrolled equivariant join with K=4 internal accumulators.");

    m.def("geometric_product_v2_5", &ezgatr::opt::geometric_product_v2_5,
          py::arg("x"), py::arg("y"),
          "Geometric product: multivector loop unrolled by K=2 (2 accumulators/blade/lane).");

    m.def("geometric_product_v2_6", &ezgatr::opt::geometric_product_v2_6,
          py::arg("x"), py::arg("y"),
          "Geometric product: multivector loop unrolled by K=4 (2 accumulators/blade/lane).");

    m.def("geometric_product_v2_7", &ezgatr::opt::geometric_product_v2_7,
          py::arg("x"), py::arg("y"),
          "Geometric product: cache-blocked (tiled) multivector loop.");

    m.def("geometric_product_v3", &ezgatr::opt::geometric_product_v3,
          py::arg("x"), py::arg("y"),
          "Geometric product: AVX2 SoA vectorization across multivectors (fp32).");

    m.def("equi_join_v2_5", &ezgatr::opt::equi_join_v2_5,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join: multivector loop unrolled by K=2 (2 accumulators/blade/lane).");

    m.def("equi_join_v2_6", &ezgatr::opt::equi_join_v2_6,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join: multivector loop unrolled by K=4 (2 accumulators/blade/lane).");

    m.def("equi_join_v2_7", &ezgatr::opt::equi_join_v2_7,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join: cache-blocked (tiled) multivector loop.");

    m.def("equi_join_v3", &ezgatr::opt::equi_join_v3,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join: AVX2 SoA vectorization across multivectors (fp32).");
}
