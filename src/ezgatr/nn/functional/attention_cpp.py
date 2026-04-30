from __future__ import annotations

import functools

import torch

from .attention import GeometricAttnKindType, GeometricQKVType
from ezgatr.opt import _opt_ops


@functools.lru_cache(maxsize=1)
def _load_attention_cpp_extension():
    return _opt_ops


def equi_geometric_attention_cpp(
    query: GeometricQKVType,
    key: GeometricQKVType,
    value: GeometricQKVType,
    kinds: dict[GeometricAttnKindType, dict[str, object] | None],
    weight: list[torch.Tensor | float] | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> GeometricQKVType:
    r"""C++ port of the multivector-only geometric attention forward pass.

    This mirrors the path used by ``MVOnlyGATrModel``. Scalar side channels are
    intentionally left to the Python implementation for now.
    """

    if isinstance(query, tuple) or isinstance(key, tuple) or isinstance(value, tuple):
        raise NotImplementedError(
            "The C++ port currently supports the multivector-only attention path."
        )

    ext = _load_attention_cpp_extension()
    ret = ext.equi_geometric_attention_mv_only(
        query,
        key,
        value,
        kinds,
        weight,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    )
    return ret, None
