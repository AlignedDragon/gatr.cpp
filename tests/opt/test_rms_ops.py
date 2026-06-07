import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings

from ezgatr.nn.functional import inner_product as inner_py
from ezgatr.nn.functional import equi_rms_norm as rms_py
from ezgatr.nn.functional import scaler_gated_gelu as gelu_py
from ezgatr.opt import (
    equi_rms_norm_ver_0 as rms_v0,
    equi_rms_norm_ver_1 as rms_v1,
    equi_rms_norm_ver_2 as rms_v2,
    equi_rms_norm_ver_3 as rms_v3,
)
from ezgatr.opt import scaler_gated_gelu_ver_2 as gelu_cpp

ALL_RMS_VERSIONS = [rms_v0, rms_v1, rms_v2, rms_v3]

batch_shape = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=3)
batch_shape_rms = st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=3)


def _check_rms_version(fn, batch):
    """Check a single rms_norm implementation against the Python reference."""
    x = torch.randn(*batch, 16, dtype=torch.float64)
    weight = torch.randn(batch[-1], dtype=torch.float64)

    torch.testing.assert_close(fn(x),              rms_py(x),              rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fn(x, weight),      rms_py(x, weight),      rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fn(x, weight, 1e-6),rms_py(x, weight, 1e-6),rtol=1e-10, atol=1e-12)

    x_zeros = torch.zeros(2, 16, dtype=torch.float64)
    torch.testing.assert_close(fn(x_zeros), rms_py(x_zeros), rtol=1e-10, atol=1e-12)

    # float32 path
    x32 = torch.randn(*batch, 16, dtype=torch.float32)
    w32 = torch.randn(batch[-1], dtype=torch.float32)
    ref32 = rms_py(x32.double()).float()
    torch.testing.assert_close(fn(x32), ref32, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(fn(x32, w32), rms_py(x32, w32), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fn", ALL_RMS_VERSIONS, ids=["v0","v1","v2","v3"])
@given(batch_shape_rms)
@settings(deadline=None, max_examples=15)
def test_equi_rms_norm_all_versions(fn, batch):
    _check_rms_version(fn, batch)


@given(batch_shape_rms)
@settings(deadline=None, max_examples=20)
def test_equi_rms_norm_matches_python(batch):
    """Legacy test kept for backwards compat — now runs v3 via parametrize above."""
    _check_rms_version(rms_v3, batch)

@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_scaler_gated_gelu_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)

    # default (tanh)
    torch.testing.assert_close(
        gelu_cpp(x),
        gelu_py(x),
        rtol=1e-10,
        atol=1e-12,
    )

    # explicit tanh
    torch.testing.assert_close(
        gelu_cpp(x, "tanh"),
        gelu_py(x, "tanh"),
        rtol=1e-10,
        atol=1e-12,
    )

    # none case (important!)
    torch.testing.assert_close(
        gelu_cpp(x, "none"),
        gelu_py(x, "none"),
        rtol=1e-10,
        atol=1e-12,
    )


