from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.strategies import floats

from ezgatr.interfaces.point import decode, encode
from tests.helpers import BATCH_DIMS, TOLERANCES
from tests.utils import make_random_3d_vectors


def test_loss_less_encode_decode():
    pcs = make_random_3d_vectors((8, 1024))
    torch.testing.assert_close(pcs, decode(encode(pcs)))


@given(
    scale=floats(
        min_value=1e-4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_mv_scale_equivariant(scale: float):
    """Test ``pts = Decode(a * Encode(pts))``.

    The multi-vector representation should be scale equivariant because
    of the existence of the homogeneous coordinate.
    """
    pcs = make_random_3d_vectors((8, 1024))
    enc = encode(pcs)

    torch.testing.assert_close(pcs, decode(enc * scale))
    torch.testing.assert_close(pcs, decode(enc * scale * -1.0))


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("rescaling", [2.0, -10.0, None])
def test_point_scaling_invariance(batch_dims, rescaling):
    """Adapted from GATr: extracted points are invariant under multivector rescaling."""
    if rescaling is None:
        while rescaling is None or np.abs(rescaling) < 0.1:
            rescaling = np.random.randn()

    multivectors = torch.randn(*batch_dims, 16)
    multivectors[..., 14] = torch.where(
        abs(multivectors[..., 14]) < 0.1, 0.1, multivectors[..., 14]
    )

    rescaled = rescaling * multivectors
    torch.testing.assert_close(decode(multivectors), decode(rescaled), **TOLERANCES)
