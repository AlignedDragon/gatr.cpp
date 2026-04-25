"""Unit tests of ezgatr.nn.functional primitives.

Adapted from `tests/gatr/primitives/{test_bilinear,test_linear,test_dual,test_invariants,
test_normalization,test_nonlinearities}.py` in geometric-algebra-transformer
(Qualcomm Technologies, Inc., 2023).
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.nn.functional import (
    equi_dual,
    equi_join,
    equi_linear,
    equi_rms_norm,
    geometric_product,
    inner_product,
    outer_product,
    scaler_gated_gelu,
)
from tests.helpers import (
    BATCH_DIMS,
    TOLERANCES,
    check_consistence_with_dual,
    check_consistence_with_geometric_product,
    check_consistence_with_outer_product,
    check_pin_equivariance,
    check_pin_invariance,
)


# ---------------------------------------------------------------------------
# Bilinear products: geometric_product, outer_product
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_correctness(batch_dims):
    """Geometric product matches the clifford library reference."""
    check_consistence_with_geometric_product(geometric_product, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_outer_product_correctness(batch_dims):
    """Outer product matches the clifford library reference."""
    check_consistence_with_outer_product(outer_product, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_equivariance(batch_dims):
    check_pin_equivariance(geometric_product, 2, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_outer_product_equivariance(batch_dims):
    check_pin_equivariance(outer_product, 2, batch_dims=batch_dims, **TOLERANCES)


# ---------------------------------------------------------------------------
# Linear: equi_linear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_identity_equivariance(batch_dims):
    """Identity map is Pin-equivariant (sanity check on the test harness itself)."""
    check_pin_equivariance(lambda x: x, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize(
    "input_batch_dims,in_channels,out_channels",
    [
        ((7,), 3, 5),
        ((3, 7), 3, 5),
        ((), 2, 4),
    ],
)
def test_equi_linear_equivariance(input_batch_dims, in_channels, out_channels):
    """`equi_linear` is Pin-equivariant for arbitrary weights."""
    weight = torch.randn(out_channels, in_channels, 9)
    fn_kwargs = dict(weight=weight)
    check_pin_equivariance(
        equi_linear,
        num_multivector_args=1,
        fn_kwargs=fn_kwargs,
        batch_dims=(*input_batch_dims, in_channels),
        **TOLERANCES,
    )


# ---------------------------------------------------------------------------
# Dual and join: equi_dual, equi_join
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_equi_dual_correctness(batch_dims):
    """`equi_dual` matches the clifford library reference.

    Note: the PGA dual is *not* Pin-equivariant in the strict sense — under a
    reflection it picks up an extra grade-involution. GATr's own test suite only
    checks correctness here and relies on `equi_join` (which sandwiches three
    duals) for the Pin-equivariant downstream behavior.
    """
    check_consistence_with_dual(equi_dual, batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_equi_join_equivariance(batch_dims):
    """`equi_join(x, y, reference)` is Pin-equivariant when a reference is supplied."""
    def _join(x, y, reference):
        return equi_join(x, y, reference=reference)

    check_pin_equivariance(_join, 3, batch_dims=batch_dims, **TOLERANCES, num_checks=10)


# ---------------------------------------------------------------------------
# Inner product invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_inner_product_invariance(batch_dims):
    """`inner_product` is Pin-invariant."""
    check_pin_invariance(inner_product, 2, batch_dims=batch_dims, **TOLERANCES)


# ---------------------------------------------------------------------------
# RMS normalization: equi_rms_norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dims", [(7, 9), ()])
@pytest.mark.parametrize("scale", [0.1, 1.0, 50.0])
def test_equi_rms_norm_correctness(batch_dims, scale):
    """After `equi_rms_norm`, mean inner-product over channels is one."""
    n_channels = 4
    inputs = scale * torch.randn(*batch_dims, n_channels, 16)
    normalized = equi_rms_norm(inputs, eps=1e-9)
    rms = torch.mean(inner_product(normalized, normalized), dim=-2)
    torch.testing.assert_close(rms, torch.ones_like(rms), **TOLERANCES)


@pytest.mark.parametrize("input_batch_dims", [(7, 9), ()])
def test_equi_rms_norm_equivariance(input_batch_dims):
    """`equi_rms_norm` is Pin-equivariant."""
    check_pin_equivariance(
        equi_rms_norm,
        num_multivector_args=1,
        batch_dims=(*input_batch_dims, 4),
        **TOLERANCES,
    )


# ---------------------------------------------------------------------------
# Activation: scaler_gated_gelu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_scaler_gated_gelu_equivariance(batch_dims):
    """`scaler_gated_gelu` is Pin-equivariant."""
    check_pin_equivariance(
        scaler_gated_gelu, 1, batch_dims=batch_dims, **TOLERANCES
    )
