"""Adapted from `tests/gatr/layers/test_linear.py` and `test_normalization.py`.

ezgatr's `EquiLinear` has no scalar-channel paths and no `initialization` kwarg —
only the linearity and Pin-equivariance tests transfer. ezgatr's `EquiRMSNorm`
replaces GATr's `EquiLayerNorm`; the variance-equals-one assertion is reframed
as RMS-equals-one over the channel axis (ezgatr divides by `mean(<x,x>)` over
`dim=-2`).
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.nn.functional import inner_product
from ezgatr.nn.modules.linear import EquiLinear
from ezgatr.nn.modules.norm import EquiRMSNorm
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("rescaling", [0.0, -2.0, 100.0])
@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_channels", [9, 1])
@pytest.mark.parametrize("out_channels", [7, 1])
def test_equi_linear_linearity(batch_dims, in_channels, out_channels, rescaling):
    """`f(x + a * y) == f(x) + a * f(y)` for the bias-free EquiLinear."""
    layer = EquiLinear(in_channels, out_channels, bias=False)

    x = torch.randn(*batch_dims, in_channels, 16)
    y = torch.randn(*batch_dims, in_channels, 16)
    xy = x + rescaling * y

    o_xy = layer(xy)
    o_x = layer(x)
    o_y = layer(y)

    torch.testing.assert_close(o_xy, o_x + rescaling * o_y, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("in_channels", [9, 1])
@pytest.mark.parametrize("out_channels", [7, 1])
@pytest.mark.parametrize("bias", [False, True])
def test_equi_linear_equivariance(batch_dims, in_channels, out_channels, bias):
    """EquiLinear is Pin(3,0,1)-equivariant."""
    layer = EquiLinear(in_channels, out_channels, bias=bias)
    data_dims = tuple(list(batch_dims) + [in_channels])
    check_pin_equivariance(layer, 1, batch_dims=data_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("in_channels", [9])
def test_equi_rms_norm_correctness(batch_dims, in_channels):
    """Output mean inner product over the channel axis equals 1.

    ezgatr's `equi_rms_norm` divides by `sqrt(mean_c <x_c, x_c>)`, so the
    same statistic on the output should be ~1.
    """
    layer = EquiRMSNorm(in_channels=in_channels, channelwise_rescale=False)
    inputs = torch.randn(*batch_dims, in_channels, 16)
    outputs = layer(inputs)

    rms_sq = inner_product(outputs, outputs).squeeze(-1).mean(dim=-1)
    torch.testing.assert_close(rms_sq, torch.ones_like(rms_sq), **TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(7, 9)])
@pytest.mark.parametrize("in_channels", [9])
def test_equi_rms_norm_equivariance(batch_dims, in_channels):
    """EquiRMSNorm is Pin(3,0,1)-equivariant."""
    layer = EquiRMSNorm(in_channels=in_channels)
    data_dims = tuple(list(batch_dims) + [in_channels])
    check_pin_equivariance(layer, 1, batch_dims=data_dims, **TOLERANCES)
