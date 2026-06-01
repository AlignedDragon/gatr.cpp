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
    r"""Baseline C++ port of the multivector-only geometric attention forward pass.

    This mirrors the path used by ``MVOnlyGATrModel``. Scalar side channels are
    intentionally left to the Python implementation for now.
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
    r"""Cache-optimized C++ port of the multivector-only geometric attention."""

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
    r"""Pre-SIMD C++ port using explicit DAA formulas and compact Q/K assembly."""

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
    r"""SIMD C++ port with vectorized compact Q/K assembly."""

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


def equi_geometric_attention_cpp_ver_4(
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
    r"""Experimental SIMD path kept for benchmarking as a failed optimization."""

    if isinstance(query, tuple) or isinstance(key, tuple) or isinstance(value, tuple):
        raise NotImplementedError(
            "The C++ port currently supports the multivector-only attention path."
        )

    ext = _load_attention_cpp_extension()
    ret = ext.equi_geometric_attention_mv_only_ver_4(
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
