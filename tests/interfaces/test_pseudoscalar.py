"""Test for ezgatr.interfaces.pseudoscalar (mirror of scalar test)."""

from __future__ import annotations

import pytest
import torch

from ezgatr.interfaces import pseudoscalar
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_pseudoscalar_embedding_consistency(batch_dims):
    """`encode` puts the value in blade 15 and zeros elsewhere; `decode` recovers it."""
    ps = torch.randn(*batch_dims, 1)
    mv = pseudoscalar.encode(ps)

    torch.testing.assert_close(mv[..., [15]], ps, **TOLERANCES)
    torch.testing.assert_close(mv[..., :15], torch.zeros_like(mv[..., :15]), **TOLERANCES)
    torch.testing.assert_close(pseudoscalar.decode(mv), ps, **TOLERANCES)
