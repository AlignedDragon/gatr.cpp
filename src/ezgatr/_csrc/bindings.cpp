#include <torch/extension.h>
#include "attention_ops.h"
#include "pga_ops.h"
#include "rms_ops.h"

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
      m.def("geometric_product_v2_1", &ezgatr::opt::geometric_product_v2_1,
            py::arg("x"), py::arg("y"),
            "Per-blade unrolled geometric product with K=2 internal accumulators.");
      m.def("geometric_product_v2_2", &ezgatr::opt::geometric_product_v2_2,
            py::arg("x"), py::arg("y"),
            "Per-blade unrolled geometric product with K=4 internal accumulators.");
      m.def("geometric_product_v2_3", &ezgatr::opt::geometric_product_v2_3,
            py::arg("x"), py::arg("y"),
            "Geometric product with K=2 accumulators per blade in a single inlined block.");
      m.def("geometric_product_v2_4", &ezgatr::opt::geometric_product_v2_4,
            py::arg("x"), py::arg("y"),
            "Geometric product with K=4 accumulators per blade in a single inlined block.");

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
      m.def("equi_join_v2_1", &ezgatr::opt::equi_join_v2_1,
            py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
            "Per-blade unrolled equivariant join with K=2 internal accumulators.");
      m.def("equi_join_v2_2", &ezgatr::opt::equi_join_v2_2,
            py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
            "Per-blade unrolled equivariant join with K=4 internal accumulators.");
      m.def("equi_join_v2_3", &ezgatr::opt::equi_join_v2_3,
            py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
            "Equivariant join with K=2 accumulators per blade in a single inlined block.");
      m.def("equi_join_v2_4", &ezgatr::opt::equi_join_v2_4,
            py::arg("x"), py::arg("y"), py::arg("reference") = py::none(),
            "Equivariant join with K=4 accumulators per blade in a single inlined block.");

      m.def("outer_product", &ezgatr::opt::outer_product,
            py::arg("x"), py::arg("y"),
            "Outer product of two multi-vectors of shape (..., 16).");
      // m.def("outer_product_ver_0", &ezgatr::opt::outer_product,
      //       py::arg("x"), py::arg("y"),
      //       "Outer product version 0 alias.");
      // m.def("outer_product_ver_1", &ezgatr::opt::outer_product,
      //       py::arg("x"), py::arg("y"),
      //       "Outer product version 1 alias.");
      // m.def("outer_product_ver_2", &ezgatr::opt::outer_product,
      //       py::arg("x"), py::arg("y"),
      //       "Outer product version 2 alias.");
      // m.def("outer_product_ver_3", &ezgatr::opt::outer_product,
      //       py::arg("x"), py::arg("y"),
      //       "Outer product version 3 alias.");

      m.def("equi_geometric_attention_mv_only_ver_0",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_0,
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
      m.def("equi_geometric_attention_ver_0",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_0,
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            py::arg("kinds"),
            py::arg("weight") = py::none(),
            py::arg("attn_mask") = py::none(),
            py::arg("dropout_p") = 0.0,
            py::arg("is_causal") = false,
            py::arg("scale") = py::none(),
            "Equivariant geometric attention version 0 alias.");

      m.def("equi_geometric_attention_mv_only_ver_1",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_1,
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
      m.def("equi_geometric_attention_ver_1",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_1,
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            py::arg("kinds"),
            py::arg("weight") = py::none(),
            py::arg("attn_mask") = py::none(),
            py::arg("dropout_p") = 0.0,
            py::arg("is_causal") = false,
            py::arg("scale") = py::none(),
            "Equivariant geometric attention version 1 alias.");

      m.def("equi_geometric_attention_mv_only_ver_2",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_2,
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
      m.def("equi_geometric_attention_ver_2",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_2,
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            py::arg("kinds"),
            py::arg("weight") = py::none(),
            py::arg("attn_mask") = py::none(),
            py::arg("dropout_p") = 0.0,
            py::arg("is_causal") = false,
            py::arg("scale") = py::none(),
            "Equivariant geometric attention version 2 alias.");

      m.def("equi_geometric_attention_mv_only_ver_3",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_3,
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
      m.def("equi_geometric_attention_ver_3",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_3,
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            py::arg("kinds"),
            py::arg("weight") = py::none(),
            py::arg("attn_mask") = py::none(),
            py::arg("dropout_p") = 0.0,
            py::arg("is_causal") = false,
            py::arg("scale") = py::none(),
            "Equivariant geometric attention version 3 alias.");
      m.def("equi_geometric_attention_mv_only_ver_4",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_4,
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            py::arg("kinds"),
            py::arg("weight") = py::none(),
            py::arg("attn_mask") = py::none(),
            py::arg("dropout_p") = 0.0,
            py::arg("is_causal") = false,
            py::arg("scale") = py::none(),
            "Experimental SIMD attention path kept for benchmarking.");
      m.def("equi_geometric_attention_ver_4",
            &ezgatr::opt::equi_geometric_attention_mv_only_ver_4,
            py::arg("query"),
            py::arg("key"),
            py::arg("value"),
            py::arg("kinds"),
            py::arg("weight") = py::none(),
            py::arg("attn_mask") = py::none(),
            py::arg("dropout_p") = 0.0,
            py::arg("is_causal") = false,
            py::arg("scale") = py::none(),
            "Experimental SIMD attention version 4 alias.");
      // m.def("inner_product", &ezgatr::opt::inner_product_ver_2,
      //       py::arg("x"), py::arg("y"),
      //       "Inner Product of two multi-vectors of shape (..., 16).");
      // m.def("inner_product_ver_0", &ezgatr::opt::inner_product_ver_0,
      //       py::arg("x"), py::arg("y"),
      //       "Inner product version 0.");
      // m.def("inner_product_ver_1", &ezgatr::opt::inner_product_ver_1,
      //       py::arg("x"), py::arg("y"),
      //       "Inner product version 1.");
      // m.def("inner_product_ver_2", &ezgatr::opt::inner_product_ver_2,
      //       py::arg("x"), py::arg("y"),
      //       "Inner product version 2.");
      // m.def("inner_product_ver_3", &ezgatr::opt::inner_product_ver_2,
      //       py::arg("x"), py::arg("y"),
      //       "Inner product version 3 fallback alias.");

      // m.def("equi_rms_norm", &ezgatr::opt::equi_rms_norm_ver_2,
      //       py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
      //       "Equi rms norm of two multivectors of shape (...,16).");
      m.def("equi_rms_norm_ver_0", &ezgatr::opt::equi_rms_norm_ver_0,
            py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
            "Equi RMS norm version 0.");
      m.def("equi_rms_norm_ver_1", &ezgatr::opt::equi_rms_norm_ver_1,
            py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
            "Equi RMS norm version 1.");
      m.def("equi_rms_norm_ver_2", &ezgatr::opt::equi_rms_norm_ver_2,
            py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
            "Equi RMS norm version 2.");
      m.def("equi_rms_norm_ver_3", &ezgatr::opt::equi_rms_norm_ver_3,
            py::arg("x"), py::arg("weight") = py::none(), py::arg("eps_opt") = py::none(),
            "Equi RMS norm version 3.");

      // m.def("scaler_gated_gelu", &ezgatr::opt::scaler_gated_gelu_ver_2,
      //       py::arg("x"),
      //       py::arg("approximate") = "tanh",
      //       "Scaler gated gelu for multivectors of shape (..., 16).");
      m.def("scaler_gated_gelu_ver_0", &ezgatr::opt::scaler_gated_gelu_ver_0,
            py::arg("x"),
            py::arg("approximate") = "tanh",
            "Scalar gated GELU version 0.");
      m.def("scaler_gated_gelu_ver_1", &ezgatr::opt::scaler_gated_gelu_ver_1,
            py::arg("x"),
            py::arg("approximate") = "tanh",
            "Scalar gated GELU version 1.");
      m.def("scaler_gated_gelu_ver_2", &ezgatr::opt::scaler_gated_gelu_ver_2,
            py::arg("x"),
            py::arg("approximate") = "tanh",
            "Scalar gated GELU version 2.");
      // m.def("scaler_gated_gelu_ver_3", &ezgatr::opt::scaler_gated_gelu_ver_2,
      //       py::arg("x"),
      //       py::arg("approximate") = "tanh",
      //       "Scalar gated GELU version 3 fallback alias.");

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
      m.def("equi_linear_ver_0", &ezgatr::opt::equi_linear_v0,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map version 0 alias.");

      m.def("equi_linear_v1", &ezgatr::opt::equi_linear_v1,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map, math-optimized kernel.");
      m.def("equi_linear_ver_1", &ezgatr::opt::equi_linear_v1,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map version 1 alias.");

      m.def("equi_linear_v2", &ezgatr::opt::equi_linear_v2,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map, scalar pre-SIMD kernel.");
      m.def("equi_linear_ver_2", &ezgatr::opt::equi_linear_v2,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map version 2 alias.");

      m.def("equi_linear_v3", &ezgatr::opt::equi_linear_v3,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map, SIMD placeholder.");
      m.def("equi_linear_ver_3", &ezgatr::opt::equi_linear_v3,
            py::arg("x"), py::arg("weight"),
            py::arg("bias") = py::none(), py::arg("normalize_basis") = true,
            "Pin(3,0,1)-equivariant linear map version 3 alias.");
}
