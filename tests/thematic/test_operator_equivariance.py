"""Sandwich-product equivariance checks across the public primitive set.

These are higher-level integration checks: the corresponding low-level
equivariance tests live in `tests/nn/test_functional.py`. This module verifies
that compositions of primitives remain Pin-equivariant.
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.nn.functional import geometric_product, inner_product, outer_product
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance, check_pin_invariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_geometric_product_equivariance(batch_dims):
    check_pin_equivariance(geometric_product, 2, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_outer_product_equivariance(batch_dims):
    check_pin_equivariance(outer_product, 2, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_double_geometric_product_equivariance(batch_dims):
    """`gp(gp(x, y), z)` is Pin-equivariant in `(x, y, z)`."""

    def fn(x, y, z):
        return geometric_product(geometric_product(x, y), z)

    check_pin_equivariance(fn, 3, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_inner_product_invariance(batch_dims):
    """`inner_product` is Pin-invariant — a Pin transform of both inputs leaves it fixed."""
    check_pin_invariance(inner_product, 2, batch_dims=batch_dims, **TOLERANCES)
