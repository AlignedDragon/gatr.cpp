#include <torch/extension.h>
#include "attention_ops.h"
#include "pga_ops.h"
#include "rms_ops.h"

PYBIND11_MODULE(_opt_ops, m) {
    m.doc() = "ezgatr.opt — C++ implementations of PGA primitives";

    m.def("geometric_product", &ezgatr::opt::geometric_product,
          py::arg("x"), py::arg("y"),
          "Geometric product of two multi-vectors of shape (..., 16).");

    m.def("equi_join", &ezgatr::opt::equi_join,
          py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
          "Equivariant join of two multi-vectors of shape (..., 16).");

    m.def("equi_geometric_attention_mv_only_base",
          &ezgatr::opt::equi_geometric_attention_mv_only_base,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("kinds"),
          py::arg("weight") = py::none(),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          "Equivariant geometric attention forward pass for mv-only inputs, baseline version.");

    m.def("equi_geometric_attention_mv_only_opt1",
          &ezgatr::opt::equi_geometric_attention_mv_only_opt1,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("kinds"),
          py::arg("weight") = py::none(),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          "Equivariant geometric attention forward pass for mv-only inputs, cache-optimized version.");

    m.def("equi_geometric_attention_mv_only_opt2",
          &ezgatr::opt::equi_geometric_attention_mv_only_opt2,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("kinds"),
          py::arg("weight") = py::none(),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          "Equivariant geometric attention forward pass for mv-only inputs, explicit-DAA version.");

    m.def("equi_geometric_attention_mv_only_opt3",
          &ezgatr::opt::equi_geometric_attention_mv_only_opt3,
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("kinds"),
          py::arg("weight") = py::none(),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          "Equivariant geometric attention forward pass for mv-only inputs, fast-path assembly version.");

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
