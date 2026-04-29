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

    m.def("inner_product", &ezgatr::opt::inner_product,
          py::arg("x"), py::arg("y"),
          "Inner Product of two multi-vectors of shape (..., 16).");

    m.def("equi_rms_norm", &ezgatr::opt::equi_rms_norm,
          py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
          "Equi rms norm of two multivectors of shape (...,16).");

    m.def("scaler_gated_gelu", &ezgatr::opt::scaler_gated_gelu,
          py::arg("x"),
          py::arg("approximate") = "tanh",
          "Scaler gated gelu for multivectors of shape (..., 16).");

}
