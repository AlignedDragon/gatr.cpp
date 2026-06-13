import torch as _torch  # noqa: F401  -- loads libtorch/libc10 into the process

from ezgatr.opt import _opt_ops

#vers?
equi_geometric_attention_mv_only = _opt_ops.equi_geometric_attention_mv_only
#start

geometric_product = _opt_ops.geometric_product
geometric_product_v0 = _opt_ops.geometric_product_v0
geometric_product_v1 = _opt_ops.geometric_product_v1
geometric_product_v2 = _opt_ops.geometric_product_v2
geometric_product_v2_1 = _opt_ops.geometric_product_v2_1
geometric_product_v2_2 = _opt_ops.geometric_product_v2_2
geometric_product_v2_3 = _opt_ops.geometric_product_v2_3
geometric_product_v2_4 = _opt_ops.geometric_product_v2_4
geometric_product_v2_5 = _opt_ops.geometric_product_v2_5
geometric_product_v2_6 = _opt_ops.geometric_product_v2_6
geometric_product_v2_7 = _opt_ops.geometric_product_v2_7
geometric_product_v3 = _opt_ops.geometric_product_v3
equi_join = _opt_ops.equi_join
equi_join_v0 = _opt_ops.equi_join_v0
equi_join_v1 = _opt_ops.equi_join_v1
equi_join_v2 = _opt_ops.equi_join_v2
equi_join_v2_1 = _opt_ops.equi_join_v2_1
equi_join_v2_2 = _opt_ops.equi_join_v2_2
equi_join_v2_3 = _opt_ops.equi_join_v2_3
equi_join_v2_4 = _opt_ops.equi_join_v2_4
equi_join_v2_5 = _opt_ops.equi_join_v2_5
equi_join_v2_6 = _opt_ops.equi_join_v2_6
equi_join_v2_7 = _opt_ops.equi_join_v2_7
equi_join_v3 = _opt_ops.equi_join_v3

geometric_bilinear_v3_1 = _opt_ops.geometric_bilinear_v3_1

outer_product = _opt_ops.outer_product
#inner_product = _opt_ops.inner_product
equi_linear = _opt_ops.equi_linear
equi_linear_v0 = _opt_ops.equi_linear_v0
equi_linear_v1 = _opt_ops.equi_linear_v1
equi_linear_v2 = _opt_ops.equi_linear_v2
equi_linear_v3 = _opt_ops.equi_linear_v3
#equi_rms_norm = _opt_ops.equi_rms_norm
#scaler_gated_gelu = _opt_ops.scaler_gated_gelu

#outer_product_ver_0 = _opt_ops.outer_product_ver_0
#inner_product_ver_0 = _opt_ops.inner_product_ver_0
equi_linear_ver_0 = _opt_ops.equi_linear_ver_0
equi_rms_norm_ver_0 = _opt_ops.equi_rms_norm_ver_0
scaler_gated_gelu_ver_0 = _opt_ops.scaler_gated_gelu_ver_0
equi_geometric_attention_ver_0 = _opt_ops.equi_geometric_attention_ver_0

#outer_product_ver_1 = _opt_ops.outer_product_ver_1
#inner_product_ver_1 = _opt_ops.inner_product_ver_1
equi_linear_ver_1 = _opt_ops.equi_linear_ver_1
equi_rms_norm_ver_1 = _opt_ops.equi_rms_norm_ver_1
scaler_gated_gelu_ver_1 = _opt_ops.scaler_gated_gelu_ver_1
equi_geometric_attention_ver_1 = _opt_ops.equi_geometric_attention_ver_1

#outer_product_ver_2 = _opt_ops.outer_product_ver_2
#inner_product_ver_2 = _opt_ops.inner_product_ver_2
equi_linear_ver_2 = _opt_ops.equi_linear_ver_2
equi_rms_norm_ver_2 = _opt_ops.equi_rms_norm_ver_2
scaler_gated_gelu_ver_2 = _opt_ops.scaler_gated_gelu_ver_2
equi_geometric_attention_ver_2 = _opt_ops.equi_geometric_attention_ver_2

#outer_product_ver_3 = _opt_ops.outer_product_ver_3
#inner_product_ver_3 = _opt_ops.inner_product_ver_3
equi_linear_ver_3 = _opt_ops.equi_linear_ver_3
equi_rms_norm_ver_3 = _opt_ops.equi_rms_norm_ver_3
scaler_gated_gelu_ver_3 = _opt_ops.scaler_gated_gelu_ver_3
equi_geometric_attention_ver_3 = _opt_ops.equi_geometric_attention_ver_3


__all__ = [
    "equi_geometric_attention_mv_only",

    "geometric_product",
    "geometric_product_v0",
    "geometric_product_v1",
    "geometric_product_v2",
    "geometric_product_v2_1",
    "geometric_product_v2_2",
    "geometric_product_v2_3",
    "geometric_product_v2_4",
    "geometric_product_v2_5",
    "geometric_product_v2_6",
    "geometric_product_v2_7",
    "geometric_product_v3",
    "equi_join",
    "equi_join_v0",
    "equi_join_v1",
    "equi_join_v2",
    "equi_join_v2_1",
    "equi_join_v2_2",
    "equi_join_v2_3",
    "equi_join_v2_4",
    "outer_product",
    #"inner_product",
    "equi_linear",
    "equi_linear_v0",
    "equi_linear_v1",
    "equi_linear_v2",
    "equi_linear_v3",
    #"equi_rms_norm",
    #"scaler_gated_gelu",
    #"outer_product_ver_0",
    # "inner_product_ver_0",
    "equi_linear_ver_0",
    "equi_rms_norm_ver_0",
    "scaler_gated_gelu_ver_0",
    "equi_geometric_attention_ver_0",
    #"outer_product_ver_1",
    #"inner_product_ver_1",
    "equi_linear_ver_1",
    "equi_rms_norm_ver_1",
    "scaler_gated_gelu_ver_1",
    "equi_geometric_attention_ver_1",
    #"outer_product_ver_2",
    #"inner_product_ver_2",
    "equi_linear_ver_2",
    "equi_rms_norm_ver_2",
    "scaler_gated_gelu_ver_2",
    "equi_geometric_attention_ver_2",
    #"outer_product_ver_3",
    #"inner_product_ver_3",
    "equi_linear_ver_3",
    "equi_rms_norm_ver_3",
    "scaler_gated_gelu_ver_3",
    "equi_geometric_attention_ver_3",

    "equi_join_v2_5",
    "equi_join_v2_6",
    "equi_join_v2_7",
    "equi_join_v3",
    "geometric_bilinear_v3_1",
]
