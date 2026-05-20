#include <torch/extension.h>
#include "attention_ops.h"
#include "pga_ops.h"
#include "rms_ops.h"

PYBIND11_MODULE(_opt_ops, m) {
    m.doc() = "ezgatr.opt — C++ implementations of PGA primitives";

//     m.def("inner_product_ver_0", &ezgatr::opt::inner_product_ver_0,
//           py::arg("x"), py::arg("y"),
//           "Inner Product of two multi-vectors of shape (..., 16).");
    
//     m.def("inner_product_ver_1", &ezgatr::opt::inner_product_ver_1,
//           py::arg("x"), py::arg("y"),
//           "Inner Product of two multi-vectors of shape (..., 16).");

//     m.def("inner_product_ver_2", &ezgatr::opt::inner_product_ver_2,
//           py::arg("x"), py::arg("y"),
//           "Inner Product of two multi-vectors of shape (..., 16).");

//     m.def("inner_product_ver_3", &ezgatr::opt::inner_product_ver_3,
//           py::arg("x"), py::arg("y"),
//           "Inner Product of two multi-vectors of shape (..., 16).");      

//     m.def("equi_rms_norm_ver_0", &ezgatr::opt::equi_rms_norm_ver_0,
//           py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
//           "Equi rms norm of two multivectors of shape (...,16).");

//     m.def("equi_rms_norm_ver_1", &ezgatr::opt::equi_rms_norm_ver_1,
//           py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
//           "Equi rms norm of two multivectors of shape (...,16).");

//     m.def("equi_rms_norm_ver_1", &ezgatr::opt::equi_rms_norm_ver_2,
//           py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
//           "Equi rms norm of two multivectors of shape (...,16).");

//     m.def("equi_rms_norm_ver_3", &ezgatr::opt::equi_rms_norm_ver_3,
//           py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
//           "Equi rms norm of two multivectors of shape (...,16).");

//     m.def("scaler_gated_gelu_ver_0", &ezgatr::opt::scaler_gated_gelu_ver_0,
//           py::arg("x"),
//           py::arg("approximate") = "tanh",
//           "Scaler gated gelu for multivectors of shape (..., 16).");

//     m.def("scaler_gated_gelu_ver_1", &ezgatr::opt::scaler_gated_gelu_ver_1,
//           py::arg("x"),
//           py::arg("approximate") = "tanh",
//           "Scaler gated gelu for multivectors of shape (..., 16).");

//     m.def("scaler_gated_gelu_ver_2", &ezgatr::opt::scaler_gated_gelu_ver_2,
//           py::arg("x"),
//           py::arg("approximate") = "tanh",
//           "Scaler gated gelu for multivectors of shape (..., 16).");

//     m.def("scaler_gated_gelu_ver_3", &ezgatr::opt::scaler_gated_gelu_ver_3,
//           py::arg("x"),
//           py::arg("approximate") = "tanh",
//           "Scaler gated gelu for multivectors of shape (..., 16).");
}
