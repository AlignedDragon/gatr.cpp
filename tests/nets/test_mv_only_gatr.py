"""Adapted from `tests/gatr/nets/test_gatr.py`.

ezgatr's `MVOnlyGATrModel` takes an `MVOnlyGATrConfig` dataclass instead of
keyword args. Equivariance only holds when `attn_is_causal=False` (a causal
mask breaks Pin equivariance under permutation), so the equivariance test
constructs a non-causal config explicitly. ezgatr has no scalar channels,
positional encoding, multi-query attention, or checkpointing — those paths
are dropped.
"""

from __future__ import annotations

import pytest
import torch

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from tests.helpers import MILD_TOLERANCES, TOLERANCES, check_pin_equivariance


def _make_config(
    in_channels: int = 3,
    out_channels: int = 4,
    hidden_channels: int = 6,
    num_layers: int = 1,
    num_heads: int = 4,
    is_causal: bool = False,
) -> MVOnlyGATrConfig:
    return MVOnlyGATrConfig(
        num_layers=num_layers,
        size_channels_in=in_channels,
        size_channels_out=out_channels,
        size_channels_hidden=hidden_channels,
        size_channels_intermediate=hidden_channels,
        attn_num_heads=num_heads,
        attn_is_causal=is_causal,
    )


@pytest.mark.parametrize("batch_dims", [(2,), (1,)])
@pytest.mark.parametrize(
    "num_items, in_channels, out_channels, hidden_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads, num_layers", [(4, 1)])
def test_mv_only_gatr_shape(
    batch_dims, num_items, in_channels, out_channels, hidden_channels, num_heads, num_layers
):
    """Output shape matches `(*batch_dims, num_items, out_channels, 16)`."""
    config = _make_config(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    net = MVOnlyGATrModel(config)
    inputs = torch.randn(*batch_dims, num_items, in_channels, 16)
    outputs = net(inputs)
    assert outputs.shape == (*batch_dims, num_items, out_channels, 16)


@pytest.mark.parametrize("batch_dims", [(2,)])
@pytest.mark.parametrize(
    "num_items, in_channels, out_channels, hidden_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads, num_layers", [(4, 1)])
def test_mv_only_gatr_equivariance(
    batch_dims, num_items, in_channels, out_channels, hidden_channels, num_heads, num_layers
):
    """Pin-equivariance of the full model with a non-causal attention mask."""
    config = _make_config(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        is_causal=False,
    )
    net = MVOnlyGATrModel(config)
    net.eval()
    data_dims = tuple(list(batch_dims) + [num_items, in_channels])
    check_pin_equivariance(net, 1, batch_dims=data_dims, **MILD_TOLERANCES)


@pytest.mark.parametrize("batch_dims", [(2,), (1,)])
@pytest.mark.parametrize(
    "num_items, in_channels, out_channels, hidden_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_heads, num_layers", [(4, 1)])
def test_mv_only_gatr_state_dict(
    batch_dims, num_items, in_channels, out_channels, hidden_channels, num_heads, num_layers
):
    """Saving and reloading a state dict reproduces the same forward output."""
    config = _make_config(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    inputs = torch.randn(*batch_dims, num_items, in_channels, 16)

    net = MVOnlyGATrModel(config)
    net.eval()
    out1 = net(inputs)

    state = net.state_dict()
    net2 = MVOnlyGATrModel(config)
    net2.load_state_dict(state)
    net2.eval()
    out2 = net2(inputs)

    torch.testing.assert_close(out1, out2, **TOLERANCES)
