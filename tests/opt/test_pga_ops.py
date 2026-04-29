import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from ezgatr.nn.functional import equi_join as join_py
from ezgatr.nn.functional import geometric_product as gp_py
from ezgatr.nn.functional import inner_product as inner_py
from ezgatr.nn.functional import equi_rms_norm as rms_py
from ezgatr.nn.functional import scaler_gated_gelu as gelu_py
from ezgatr.opt import equi_join as join_cpp
from ezgatr.opt import geometric_product as gp_cpp
from ezgatr.opt import inner_product as inner_cpp
from ezgatr.opt import equi_rms_norm as rms_cpp
from ezgatr.opt import scaler_gated_gelu as gelu_cpp


batch_shape = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=3)
batch_shape_rms = st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=3)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(gp_cpp(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_no_reference_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(join_cpp(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_with_reference_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(join_cpp(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)

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


def test_dtype_float32():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    y = torch.randn(2, 3, 16)
    ref = torch.randn(2, 3, 16)
    torch.testing.assert_close(gp_cpp(x, y), gp_py(x, y), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(join_cpp(x, y), join_py(x, y), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(join_cpp(x, y, ref), join_py(x, y, ref), rtol=1e-5, atol=1e-6)


def test_caches_warm_up_idempotently():
    x = torch.randn(4, 16, dtype=torch.float64)
    y = torch.randn(4, 16, dtype=torch.float64)
    a = gp_cpp(x, y)
    b = gp_cpp(x, y)
    torch.testing.assert_close(a, b)
    a2 = join_cpp(x, y)
    b2 = join_cpp(x, y)
    torch.testing.assert_close(a2, b2)
