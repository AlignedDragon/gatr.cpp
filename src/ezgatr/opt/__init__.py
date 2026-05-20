import torch as _torch  # noqa: F401  -- loads libtorch/libc10 into the process

from ezgatr.opt import _opt_ops

geometric_product = _opt_ops.geometric_product
geometric_product_dense = _opt_ops.geometric_product_dense
geometric_product_sparse_rt = _opt_ops.geometric_product_sparse_rt

equi_join = _opt_ops.equi_join
equi_join_dense = _opt_ops.equi_join_dense
equi_join_sparse_rt = _opt_ops.equi_join_sparse_rt

__all__ = [
    "geometric_product",
    "geometric_product_dense",
    "geometric_product_sparse_rt",
    "equi_join",
    "equi_join_dense",
    "equi_join_sparse_rt",
]
