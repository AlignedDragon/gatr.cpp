"""Adapted from `tests/gatr/interface/test_translation.py` in geometric-algebra-transformer.

Note: ezgatr's `translation.decode` raises ``NotImplementedError``, so the
cycle-consistency test from GATr is intentionally omitted. The four
sandwich-product tests below only need ``translation.encode``.
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.interfaces import plane, point, pseudoscalar, scalar, translation
from ezgatr.nn.functional import geometric_product
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_point(batch_dims):
    """Sandwich product of a translation rotor on a point yields the translated point."""
    points = torch.randn(*batch_dims, 3)
    translation_vectors = torch.randn(*batch_dims, 3)

    points_embedding = point.encode(points)
    u = translation.encode(translation_vectors)
    u_inv = translation.encode(-translation_vectors)

    sandwich = geometric_product(geometric_product(u, points_embedding), u_inv)
    translated = point.decode(sandwich)

    torch.testing.assert_close(translated, points + translation_vectors, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_plane(batch_dims):
    """Translation acts on planes through the origin without changing the normal."""
    normals = torch.randn(*batch_dims, 3)
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
    translation_vectors = torch.randn(*batch_dims, 3)
    pos = torch.zeros_like(normals)

    plane_embedding = plane.encode(normals, pos)
    u = translation.encode(translation_vectors)
    u_inv = translation.encode(-translation_vectors)

    sandwich = geometric_product(geometric_product(u, plane_embedding), u_inv)
    translated_normals, _ = plane.decode(sandwich)

    torch.testing.assert_close(translated_normals, normals, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_scalar(batch_dims):
    """Translation acts trivially on scalars."""
    s = torch.randn(*batch_dims, 1)
    x = scalar.encode(s)

    t = torch.randn(*batch_dims, 3)
    u = translation.encode(t)
    u_inv = translation.encode(-t)

    rx = geometric_product(geometric_product(u, x), u_inv)
    rs = scalar.decode(rx)

    torch.testing.assert_close(rs, s, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_translation_on_pseudoscalar(batch_dims):
    """Translation acts trivially on pseudoscalars."""
    ps = torch.randn(*batch_dims, 1)
    x = pseudoscalar.encode(ps)

    t = torch.randn(*batch_dims, 3)
    u = translation.encode(t)
    u_inv = translation.encode(-t)

    rx = geometric_product(geometric_product(u, x), u_inv)
    rps = pseudoscalar.decode(rx)

    torch.testing.assert_close(rps, ps, **TOLERANCES)
