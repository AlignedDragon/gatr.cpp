#include <torch/extension.h>
#include "attention_ops.h"
#include "pga_ops.h"
#include "rms_ops.h"

PYBIND11_MODULE(_opt_ops, m) {
    m.doc() = "ezgatr.opt — C++ implementations of PGA primitives";

//     m.def("geometric_product", &ezgatr::opt::geometric_product,
//           py::arg("x"), py::arg("y"),
//           "Geometric product of two multi-vectors of shape (..., 16).");

//     m.def("equi_join", &ezgatr::opt::equi_join,
//           py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
//           "Equivariant join of two multi-vectors of shape (..., 16).");

    m.def("inner_product_ver_0", &ezgatr::opt::inner_product_ver_0,
          py::arg("x"), py::arg("y"),
          "Inner Product of two multi-vectors of shape (..., 16).");
    
    m.def("inner_product_ver_1", &ezgatr::opt::inner_product_ver_1,
          py::arg("x"), py::arg("y"),
          "Inner Product of two multi-vectors of shape (..., 16).");

    m.def("inner_product_ver_2", &ezgatr::opt::inner_product_ver_2,
          py::arg("x"), py::arg("y"),
          "Inner Product of two multi-vectors of shape (..., 16).");

    m.def("inner_product_ver_3", &ezgatr::opt::inner_product_ver_3,
          py::arg("x"), py::arg("y"),
          "Inner Product of two multi-vectors of shape (..., 16).");      

    m.def("equi_rms_norm_ver_0", &ezgatr::opt::equi_rms_norm_ver_0,
          py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
          "Equi rms norm of two multivectors of shape (...,16).");

    m.def("equi_rms_norm_ver_1", &ezgatr::opt::equi_rms_norm_ver_1,
          py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
          "Equi rms norm of two multivectors of shape (...,16).");

    m.def("equi_rms_norm_ver_1", &ezgatr::opt::equi_rms_norm_ver_2,
          py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
          "Equi rms norm of two multivectors of shape (...,16).");

    m.def("equi_rms_norm_ver_3", &ezgatr::opt::equi_rms_norm_ver_3,
          py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
          "Equi rms norm of two multivectors of shape (...,16).");

    m.def("scaler_gated_gelu_ver_0", &ezgatr::opt::scaler_gated_gelu_ver_0,
          py::arg("x"),
          py::arg("approximate") = "tanh",
          "Scaler gated gelu for multivectors of shape (..., 16).");

    m.def("scaler_gated_gelu_ver_1", &ezgatr::opt::scaler_gated_gelu_ver_1,
          py::arg("x"),
          py::arg("approximate") = "tanh",
          "Scaler gated gelu for multivectors of shape (..., 16).");

    m.def("scaler_gated_gelu_ver_2", &ezgatr::opt::scaler_gated_gelu_ver_2,
          py::arg("x"),
          py::arg("approximate") = "tanh",
          "Scaler gated gelu for multivectors of shape (..., 16).");

    m.def("scaler_gated_gelu_ver_3", &ezgatr::opt::scaler_gated_gelu_ver_3,
          py::arg("x"),
          py::arg("approximate") = "tanh",
          "Scaler gated gelu for multivectors of shape (..., 16).");

    m.def("equi_geometric_attention_mv_only",
          &ezgatr::opt::equi_geometric_attention_mv_only,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("kinds"),
          py::arg("weight") = py::none(),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          "Equivariant geometric attention forward pass for mv-only inputs.");

    m.def("equi_linear", &ezgatr::opt::equi_linear,
          py::arg("x"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
          "Pin(3,0,1)-equivariant linear map. x: (..., C_in, 16), weight: (C_out, C_in, 9).");
}
