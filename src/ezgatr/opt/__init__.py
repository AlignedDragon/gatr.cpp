import torch as _torch  # noqa: F401  -- loads libtorch/libc10 into the process

from ezgatr.opt import _opt_ops

geometric_product = _opt_ops.geometric_product
geometric_product_dense = _opt_ops.geometric_product_dense
geometric_product_sparse_rt = _opt_ops.geometric_product_sparse_rt
geometric_product_ilp2 = _opt_ops.geometric_product_ilp2
geometric_product_ilp4 = _opt_ops.geometric_product_ilp4
geometric_product_acc2 = _opt_ops.geometric_product_acc2
geometric_product_acc4 = _opt_ops.geometric_product_acc4

equi_join = _opt_ops.equi_join
equi_join_dense = _opt_ops.equi_join_dense
equi_join_sparse_rt = _opt_ops.equi_join_sparse_rt
equi_join_ilp2 = _opt_ops.equi_join_ilp2
equi_join_ilp4 = _opt_ops.equi_join_ilp4
equi_join_acc2 = _opt_ops.equi_join_acc2
equi_join_acc4 = _opt_ops.equi_join_acc4

__all__ = [
    "geometric_product",
    "geometric_product_dense",
    "geometric_product_sparse_rt",
    "geometric_product_ilp2",
    "geometric_product_ilp4",
    "geometric_product_acc2",
    "geometric_product_acc4",
    "equi_join",
    "equi_join_dense",
    "equi_join_sparse_rt",
    "equi_join_ilp2",
    "equi_join_ilp4",
    "equi_join_acc2",
    "equi_join_acc4",
]
