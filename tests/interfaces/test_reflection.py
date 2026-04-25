"""Adapted from `tests/gatr/interface/test_reflection.py` in geometric-algebra-transformer.

ezgatr does not export a public `grade_involute` primitive, so the test uses
the local helper ``tests.utils.grade_involute``.
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.interfaces import plane, point, pseudoscalar, reflection, scalar
from ezgatr.nn.functional import geometric_product
from tests.helpers import BATCH_DIMS, TOLERANCES
from tests.utils import grade_involute


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reflection_embedding_consistency(batch_dims):
    """Encoded reflection planes round-trip through `decode`."""
    normals = torch.randn(*batch_dims, 3)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
    pos = torch.randn(*batch_dims, 3)
    mv = reflection.encode(normals, pos)
    extracted_normals, _ = reflection.decode(mv)

    extracted_unit = extracted_normals / torch.linalg.norm(
        extracted_normals, dim=-1, keepdim=True
    )
    torch.testing.assert_close(extracted_unit, normals, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflection_on_point(batch_dims, axis):
    """Reflecting a point along a coordinate axis flips that coordinate."""
    points = torch.randn(*batch_dims, 3)
    n = torch.zeros(3)
    n[axis] = 1.0

    expected = points.clone()
    expected[..., axis] = -points[..., axis]

    p = point.encode(points)
    r = reflection.encode(n, torch.zeros_like(n))
    sandwich = grade_involute(geometric_product(geometric_product(r, p), r))
    reflected = point.decode(sandwich)

    torch.testing.assert_close(reflected, expected, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflection_on_plane(batch_dims, axis):
    """Reflecting a plane through-the-origin along a coordinate axis flips its normal component."""
    normals = torch.randn(*batch_dims, 3)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)

    expected = normals.clone()
    expected[..., axis] = -normals[..., axis]

    p = plane.encode(normals, torch.zeros_like(normals))
    n = torch.zeros(3)
    n[axis] = 1.0
    r = reflection.encode(n, torch.zeros_like(n))

    sandwich = grade_involute(geometric_product(geometric_product(r, p), r))
    reflected_normals, _ = plane.decode(sandwich)

    torch.testing.assert_close(reflected_normals, expected, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reflection_on_scalar(batch_dims):
    """Reflection acts trivially on scalars."""
    inputs = torch.randn(*batch_dims, 1)
    normals = torch.randn(*batch_dims, 3)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
    pos = torch.zeros_like(normals)

    x = scalar.encode(inputs)
    r = reflection.encode(normals, pos)
    sandwich = grade_involute(geometric_product(geometric_product(r, x), r))

    torch.testing.assert_close(scalar.decode(sandwich), inputs, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_reflection_on_pseudoscalar(batch_dims):
    """Reflection negates pseudoscalars (orientation flip)."""
    inputs = torch.randn(*batch_dims, 1)
    normals = torch.randn(*batch_dims, 3)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
    pos = torch.zeros_like(normals)

    x = pseudoscalar.encode(inputs)
    r = reflection.encode(normals, pos)
    sandwich = grade_involute(geometric_product(geometric_product(r, x), r))

    torch.testing.assert_close(pseudoscalar.decode(sandwich), -inputs, **TOLERANCES)
