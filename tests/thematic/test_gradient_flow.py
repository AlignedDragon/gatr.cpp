"""Gradient-flow checks for ezgatr modules and cached basis tensors."""

from __future__ import annotations

import pathlib

import torch

from ezgatr.nn.functional.linear import _compute_pin_equi_linear_basis
from ezgatr.nn.modules.linear import EquiLinear
from ezgatr.nn.modules.norm import EquiRMSNorm


def test_equi_linear_weights_receive_gradients():
    """`EquiLinear`'s weight and bias accumulate gradients on a scalar loss."""
    layer = EquiLinear(in_channels=3, out_channels=5)
    x = torch.randn(2, 3, 16, requires_grad=False)
    loss = layer(x).pow(2).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.any(layer.weight.grad != 0)
    assert layer.bias is not None
    assert layer.bias.grad is not None


def test_equi_rms_norm_weight_receives_gradient():
    layer = EquiRMSNorm(in_channels=3)
    x = torch.randn(2, 3, 16)
    loss = layer(x).pow(2).sum()
    loss.backward()
    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert torch.any(layer.weight.grad != 0)


def test_cached_basis_tensors_have_no_grad():
    """The on-disk PGA basis tensors must not require gradients.

    Loading a cached `.pt` with `requires_grad=True` would silently inject
    parameters into every linear forward — these are constants, not parameters.
    """
    basis_dir = pathlib.Path(_compute_pin_equi_linear_basis.__module__).parent
    # Load actual cached files from the package
    import ezgatr.nn.functional.linear as linear_mod

    pkg_dir = pathlib.Path(linear_mod.__file__).parent / "basis"
    files = sorted(pkg_dir.glob("*.pt"))
    assert files, f"expected cached basis tensors under {pkg_dir}"

    for f in files:
        tensor = torch.load(f, weights_only=True)
        assert isinstance(tensor, torch.Tensor)
        assert not tensor.requires_grad, f"{f.name} has requires_grad=True"
