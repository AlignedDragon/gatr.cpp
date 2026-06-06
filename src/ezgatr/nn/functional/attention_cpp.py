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
    return equi_geometric_attention_cpp_ver_3(
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


def equi_geometric_attention_cpp_ver_0(
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
    r"""Baseline mv-only geometric attention (v0).

    einsum Q/K assembly and a plain scalar SDPA (score a row, softmax, P @ V),
    no PyTorch ``scaled_dot_product_attention``. Scalar side channels are left
    to the Python path for now.
    """

    if isinstance(query, tuple) or isinstance(key, tuple) or isinstance(value, tuple):
        raise NotImplementedError(
            "The C++ port currently supports the multivector-only attention path."
        )

    ext = _load_attention_cpp_extension()
    ret = ext.equi_geometric_attention_mv_only_ver_0(
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


def equi_geometric_attention_cpp_ver_1(
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
    r"""v1 math: explicit DAA formula instead of einsum + cached constants. SDPA stays naive scalar."""

    if isinstance(query, tuple) or isinstance(key, tuple) or isinstance(value, tuple):
        raise NotImplementedError(
            "The C++ port currently supports the multivector-only attention path."
        )

    ext = _load_attention_cpp_extension()
    ret = ext.equi_geometric_attention_mv_only_ver_1(
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


def equi_geometric_attention_cpp_ver_2(
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
    r"""v2 scalar/memory: single-pass Q/K assembly (no torch::cat), unrolled for channels=4,8, and a scalar flash SDPA (tiling + online softmax, no T x T matrix)."""

    if isinstance(query, tuple) or isinstance(key, tuple) or isinstance(value, tuple):
        raise NotImplementedError(
            "The C++ port currently supports the multivector-only attention path."
        )

    ext = _load_attention_cpp_extension()
    ret = ext.equi_geometric_attention_mv_only_ver_2(
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


def equi_geometric_attention_cpp_ver_3(
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
    r"""v3 SIMD: fused per-head Q/K assembly + AVX2 flash SDPA (cache-resident scratch so query/key never hit DRAM, packed-K register-blocked QK^T micro-kernel, register-blocked P@V, vectorized exp)."""

    if isinstance(query, tuple) or isinstance(key, tuple) or isinstance(value, tuple):
        raise NotImplementedError(
            "The C++ port currently supports the multivector-only attention path."
        )

    ext = _load_attention_cpp_extension()
    ret = ext.equi_geometric_attention_mv_only_ver_3(
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
