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
    m.def("equi_linear", &ezgatr::opt::equi_linear,
          py::arg("x"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
          py::arg("version") = 0,
          "Pin(3,0,1)-equivariant linear map. x: (..., C_in, 16), "
          "weight: (C_out, C_in, 9). version: 0=no-opt (default), "
          "1=math, 2=pre-SIMD, 3=SIMD placeholder.");

    m.def("equi_linear_v0", &ezgatr::opt::equi_linear_v0,
          py::arg("x"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
          "Pin(3,0,1)-equivariant linear map, dense baseline.");

    m.def("equi_linear_v1", &ezgatr::opt::equi_linear_v1,
          py::arg("x"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
          "Pin(3,0,1)-equivariant linear map, math-optimized kernel.");

    m.def("equi_linear_v2", &ezgatr::opt::equi_linear_v2,
          py::arg("x"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
          "Pin(3,0,1)-equivariant linear map, scalar pre-SIMD kernel.");

    m.def("equi_linear_v3", &ezgatr::opt::equi_linear_v3,
          py::arg("x"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
          "Pin(3,0,1)-equivariant linear map, SIMD placeholder.");
}
