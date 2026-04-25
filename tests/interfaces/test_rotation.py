"""Adapted from `tests/gatr/interface/test_rotation.py` in geometric-algebra-transformer."""

from __future__ import annotations

import pytest
import torch

from ezgatr.interfaces import plane, point, pseudoscalar, rotation, scalar
from ezgatr.nn.functional import geometric_product
from tests.helpers import BATCH_DIMS, TOLERANCES
from tests.utils import quaternion_to_rotation_matrix, random_quaternion, reverse


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_embedding_consistency(batch_dims):
    """`encode`/`decode` round-trip on quaternions."""
    q = random_quaternion(batch_dims)
    mv = rotation.encode(q)
    q_back = rotation.decode(mv)
    torch.testing.assert_close(q, q_back, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_point(batch_dims):
    """Sandwich product of a rotation rotor on a point matches matrix-vector rotation."""
    q = random_quaternion(batch_dims)
    x = torch.randn(*batch_dims, 3)

    r = quaternion_to_rotation_matrix(q)
    rx = torch.einsum("...ij,...j->...i", r, x)

    m = rotation.encode(q)
    p = point.encode(x)
    mpm = geometric_product(geometric_product(m, p), reverse(m))
    rxp = point.encode(rx)
    extracted = point.decode(mpm)

    torch.testing.assert_close(mpm, rxp, **TOLERANCES)
    torch.testing.assert_close(extracted, rx, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_plane(batch_dims):
    """Sandwich product of a rotation rotor on a plane through the origin matches matrix-vector rotation of its normal."""
    q = random_quaternion(batch_dims)
    normals = torch.randn(*batch_dims, 3)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)

    r = quaternion_to_rotation_matrix(q)
    rotated_normals = torch.einsum("...ij,...j->...i", r, normals)

    m = rotation.encode(q)
    p = plane.encode(normals, torch.zeros_like(normals))
    mpm = geometric_product(geometric_product(m, p), reverse(m))
    rxp = plane.encode(rotated_normals, torch.zeros_like(normals))
    extracted_normals, _ = plane.decode(mpm)

    torch.testing.assert_close(mpm, rxp, **TOLERANCES)
    torch.testing.assert_close(extracted_normals, rotated_normals, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_scalar(batch_dims):
    """Rotation acts trivially on scalars."""
    s = torch.randn(*batch_dims, 1)
    x = scalar.encode(s)

    q = random_quaternion(batch_dims)
    u = rotation.encode(q)

    rx = geometric_product(geometric_product(u, x), reverse(u))
    rs = scalar.decode(rx)

    torch.testing.assert_close(rs, s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_rotation_on_pseudoscalar(batch_dims):
    """Rotation acts trivially on pseudoscalars."""
    ps = torch.randn(*batch_dims, 1)
    x = pseudoscalar.encode(ps)

    q = random_quaternion(batch_dims)
    u = rotation.encode(q)

    rx = geometric_product(geometric_product(u, x), reverse(u))
    rps = pseudoscalar.decode(rx)

    torch.testing.assert_close(rps, ps, **TOLERANCES)
