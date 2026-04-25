import torch as _torch  # noqa: F401  -- loads libtorch/libc10 into the process

from ezgatr.opt import _opt_ops

geometric_product = _opt_ops.geometric_product
equi_join = _opt_ops.equi_join

__all__ = ["geometric_product", "equi_join"]
