"""Adapted from `tests/gatr/interface/test_scalar.py` in geometric-algebra-transformer."""

from __future__ import annotations

import pytest
import torch

from ezgatr.interfaces import scalar
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_scalar_embedding_consistency(batch_dims):
    """`encode` puts the scalar in blade 0 and zeros elsewhere; `decode` recovers it."""
    s = torch.randn(*batch_dims, 1)
    mv = scalar.encode(s)

    torch.testing.assert_close(mv[..., [0]], s, **TOLERANCES)
    torch.testing.assert_close(mv[..., 1:], torch.zeros_like(mv[..., 1:]), **TOLERANCES)
    torch.testing.assert_close(scalar.decode(mv), s, **TOLERANCES)
