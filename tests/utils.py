"""Test-only random data and clifford-tensor conversion utilities."""

from __future__ import annotations

import clifford
import numpy as np
import torch
from clifford import pga as PGA
from hypothesis import strategies as st

from tests.helpers.clifford import mv_list_to_tensor


def make_random_3d_vectors(batch_dims: tuple[int, ...]) -> torch.Tensor:
    """Sample a tensor of standard-normal 3D vectors with the given batch shape."""
    return torch.randn(*batch_dims, 3)


def make_random_pga_mvs(batch_dims: tuple[int, ...]) -> list:
    """Sample a list of clifford PGA multivectors of length ``prod(batch_dims)``."""
    total = 1 if not batch_dims else int(np.prod(batch_dims))
    xs = clifford.randomMV(layout=PGA.layout, n=total)
    if total == 1:
        xs = [xs]
    return xs


def mv_to_tensor(mv_list, batch_dims: tuple[int, ...]) -> torch.Tensor:
    """Pack a list of clifford multivectors into a torch.Tensor of shape ``(*batch_dims, 16)``."""
    return mv_list_to_tensor(mv_list, batch_shape=batch_dims)


def strategy_batch_dims(max_size: int = 8):
    """Hypothesis strategy producing tuples of integer batch dimensions."""
    return st.lists(
        st.integers(min_value=1, max_value=max_size),
        min_size=0,
        max_size=3,
    ).map(tuple)


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Apply the PGA grade involution to a multi-vector tensor of shape ``(..., 16)``.

    Sign-flips the odd-grade blades (indices 1-4 and 11-14) of the standard 16-dim
    PGA basis. Used in tests for reflection-style operators.
    """
    sign = torch.tensor(
        [1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1],
        device=x.device,
        dtype=x.dtype,
    )
    return x * sign


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Apply the PGA reversal to a multi-vector tensor of shape ``(..., 16)``.

    Sign-flips grades 2 and 3 (blade indices 5-14). Used to compute the inverse of
    unit rotors / translators in test sandwich products.
    """
    sign = torch.tensor(
        [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
        device=x.device,
        dtype=x.dtype,
    )
    return x * sign


def random_quaternion(batch_dims: tuple[int, ...]) -> torch.Tensor:
    """Sample unit quaternions ``(*batch_dims, 4)`` in (x, y, z, w) order."""
    q = torch.randn(*batch_dims, 4)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``(*batch_dims, 4)`` (x, y, z, w) to rotation matrices ``(*batch_dims, 3, 3)``."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_xx = 2 * x * x
    two_yy = 2 * y * y
    two_zz = 2 * z * z
    two_xy = 2 * x * y
    two_xz = 2 * x * z
    two_yz = 2 * y * z
    two_wx = 2 * w * x
    two_wy = 2 * w * y
    two_wz = 2 * w * z

    rot = torch.stack(
        [
            torch.stack([1 - two_yy - two_zz, two_xy - two_wz, two_xz + two_wy], dim=-1),
            torch.stack([two_xy + two_wz, 1 - two_xx - two_zz, two_yz - two_wx], dim=-1),
            torch.stack([two_xz - two_wy, two_yz + two_wx, 1 - two_xx - two_yy], dim=-1),
        ],
        dim=-2,
    )
    return rot
