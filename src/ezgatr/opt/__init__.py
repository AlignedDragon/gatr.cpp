import torch as _torch  # noqa: F401  -- loads libtorch/libc10 into the process

from ezgatr.opt import _opt_ops

geometric_product = _opt_ops.geometric_product
geometric_product_v0 = _opt_ops.geometric_product_v0
geometric_product_v1 = _opt_ops.geometric_product_v1
geometric_product_v2 = _opt_ops.geometric_product_v2
geometric_product_v2_1 = _opt_ops.geometric_product_v2_1
geometric_product_v2_2 = _opt_ops.geometric_product_v2_2
geometric_product_v2_3 = _opt_ops.geometric_product_v2_3
geometric_product_v2_4 = _opt_ops.geometric_product_v2_4

equi_join = _opt_ops.equi_join
equi_join_v0 = _opt_ops.equi_join_v0
equi_join_v1 = _opt_ops.equi_join_v1
equi_join_v2 = _opt_ops.equi_join_v2
equi_join_v2_1 = _opt_ops.equi_join_v2_1
equi_join_v2_2 = _opt_ops.equi_join_v2_2
equi_join_v2_3 = _opt_ops.equi_join_v2_3
equi_join_v2_4 = _opt_ops.equi_join_v2_4

__all__ = [
    "geometric_product",
    "geometric_product_v0",
    "geometric_product_v1",
    "geometric_product_v2",
    "geometric_product_v2_1",
    "geometric_product_v2_2",
    "geometric_product_v2_3",
    "geometric_product_v2_4",
    "equi_join",
    "equi_join_v0",
    "equi_join_v1",
    "equi_join_v2",
    "equi_join_v2_1",
    "equi_join_v2_2",
    "equi_join_v2_3",
    "equi_join_v2_4",
]
