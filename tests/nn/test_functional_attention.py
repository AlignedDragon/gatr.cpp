"""Adapted from `tests/gatr/primitives/test_attention.py`.

The ezgatr `equi_geometric_attention` API differs significantly from GATr's
`geometric_attention`/`pga_attention`/`sdp_attention`:

- Inputs are pure multi-vectors `(B, H, T, channels, 16)` (or `(mv, scalar)`
  tuples). Tests here only exercise the multi-vector path.
- Similarity kinds are passed as a `kinds` dict like `{"ipa": None, "daa": None}`.

Drop the xformers/BlockDiagonalMask cross-attention case from GATr.
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.interfaces import point, scalar
from ezgatr.nn.functional.attention import equi_geometric_attention
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_mv_channels_out", [3, 1])
@pytest.mark.parametrize("num_mv_channels_in", [2, 1])
@pytest.mark.parametrize("num_tokens_out", [5, 1])
@pytest.mark.parametrize("num_tokens_in", [7, 1])
def test_equi_geometric_attention_shape(
    batch_dims,
    num_tokens_in,
    num_tokens_out,
    num_mv_channels_in,
    num_mv_channels_out,
):
    """Outputs of `equi_geometric_attention` have correct shape (mv-only path)."""
    q_mv = torch.randn(*batch_dims, num_tokens_out, num_mv_channels_in, 16)
    k_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_in, 16)
    v_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_out, 16)

    out_mv, out_s = equi_geometric_attention(
        q_mv, k_mv, v_mv, kinds={"ipa": None, "daa": None}
    )

    assert out_mv.shape == (*batch_dims, num_tokens_out, num_mv_channels_out, 16)
    assert out_s is None


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("key_dim", [2])
@pytest.mark.parametrize("item_dim", [3])
def test_equi_geometric_attention_equivariance(batch_dims, key_dim, item_dim):
    """`equi_geometric_attention` is Pin(3,0,1)-equivariant on its mv inputs."""
    data_dims = tuple(list(batch_dims) + [item_dim, key_dim])
    fn_kwargs = dict(kinds={"ipa": None, "daa": None})
    check_pin_equivariance(
        equi_geometric_attention, 3, batch_dims=data_dims, fn_kwargs=fn_kwargs, **TOLERANCES
    )


def test_equi_geometric_attention_proximity():
    """With distance-aware attention only, attends to the closer point."""
    q_mv = point.encode(torch.zeros(1, 1, 3))  # (1, 1, 16) point at origin
    k_mv = point.encode(torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 100.0, 0.0]]]))  # (2, 1, 16)
    v_mv = scalar.encode(torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]]))  # (2, 2, 16)

    out_mv, out_s = equi_geometric_attention(
        q_mv, k_mv, v_mv, kinds={"daa": None}
    )

    torch.testing.assert_close(out_mv, v_mv[[0]], **TOLERANCES)
    assert out_s is None
