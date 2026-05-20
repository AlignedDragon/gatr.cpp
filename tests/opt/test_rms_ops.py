import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from ezgatr.nn.functional import inner_product as inner_py
from ezgatr.nn.functional import equi_rms_norm as rms_py
from ezgatr.nn.functional import scaler_gated_gelu as gelu_py
from ezgatr.opt import inner_product_ver_1 as inner_cpp
#from ezgatr.opt import equi_rms_norm as rms_cpp
from ezgatr.opt import equi_rms_norm_ver_2 as rms_cpp
from ezgatr.opt import scaler_gated_gelu_ver_1 as gelu_cpp


batch_shape = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=3)
batch_shape_rms = st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=3)



@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_inner_product_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(inner_cpp(x, y), inner_py(x, y), rtol=1e-10, atol=1e-12)

@given(batch_shape_rms)
@settings(deadline=None, max_examples=20)
def test_equi_rms_norm_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)

    #no weight
    torch.testing.assert_close(
        rms_cpp(x),
        rms_py(x),
        rtol=1e-10,
        atol=1e-12,
    )

    weight = torch.randn(batch[-1],dtype=torch.float64)

    #weight
    torch.testing.assert_close(
        rms_cpp(x, weight),
        rms_py(x, weight),
        rtol=1e-10,
        atol=1e-12,
    )

    # custom eps
    torch.testing.assert_close(
        rms_cpp(x, weight, 1e-6),
        rms_py(x, weight, 1e-6),
        rtol=1e-10,
        atol=1e-12,
    )

    # test clamp

    x = torch.zeros(2, 16, dtype=torch.float64)

    torch.testing.assert_close(
        rms_cpp(x), rms_py(x),
        rtol=1e-10,
        atol=1e-12,
    )

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


