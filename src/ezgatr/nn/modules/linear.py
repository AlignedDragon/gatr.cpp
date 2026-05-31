from __future__ import annotations

import math

import torch
import torch.nn as nn

from ezgatr.nn.functional.linear import equi_linear
from ezgatr.opt import equi_linear_ver_0,equi_linear_ver_1,equi_linear_ver_2,equi_linear_ver_3


class EquiLinear(nn.Module):
    r"""Pin(3, 0, 1)-equivariant linear map.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    normalize_basis : bool
        Whether to normalize the basis elements according to the number of
        "inflow paths" for each blade.
    """

    __constants__ = ["in_channels", "out_channels", "normalize_basis"]

    in_channels: int
    out_channels: int
    normalize_basis: bool
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        normalize_basis: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_basis = normalize_basis
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, 9), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear(x, self.weight, self.bias, self.normalize_basis)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"bias={self.bias is not None}, normalize_basis={self.normalize_basis}"
        )


class EquiLinearASL_ver_0(EquiLinear):
    r"""C++-backed Pin(3, 0, 1)-equivariant linear map.

    Drop-in replacement for :class:`EquiLinear` that calls the optimized
    unrolled C++ kernel instead of the Python einsum baseline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear_ver_0(x, self.weight, self.bias, self.normalize_basis)
    
class EquiLinearASL_ver_1(EquiLinear):
    r"""C++-backed Pin(3, 0, 1)-equivariant linear map.

    Drop-in replacement for :class:`EquiLinear` that calls the optimized
    unrolled C++ kernel instead of the Python einsum baseline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear_ver_1(x, self.weight, self.bias, self.normalize_basis)
    
class EquiLinearASL_ver_2(EquiLinear):
    r"""C++-backed Pin(3, 0, 1)-equivariant linear map.

    Drop-in replacement for :class:`EquiLinear` that calls the optimized
    unrolled C++ kernel instead of the Python einsum baseline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear_ver_2(x, self.weight, self.bias, self.normalize_basis)
    
class EquiLinearASL_ver_3(EquiLinear):
    r"""C++-backed Pin(3, 0, 1)-equivariant linear map.

    Drop-in replacement for :class:`EquiLinear` that calls the optimized
    unrolled C++ kernel instead of the Python einsum baseline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return equi_linear_ver_3(x, self.weight, self.bias, self.normalize_basis)


# Backward-compatible alias used throughout the model code.
#ASLEquiLinear = EquiLinearASL_ver_0
