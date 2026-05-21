import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from ezgatr.nn.functional import equi_join as join_py
from ezgatr.nn.functional import geometric_product as gp_py
from ezgatr.opt import equi_join as join_cpp
from ezgatr.opt import equi_join_acc2, equi_join_acc4
from ezgatr.opt import equi_join_ilp2, equi_join_ilp4
from ezgatr.opt import geometric_product as gp_cpp
from ezgatr.opt import geometric_product_acc2, geometric_product_acc4
from ezgatr.opt import geometric_product_ilp2, geometric_product_ilp4


batch_shape = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=3)


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


def test_dtype_float32():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    y = torch.randn(2, 3, 16)
    ref = torch.randn(2, 3, 16)
    torch.testing.assert_close(gp_cpp(x, y), gp_py(x, y), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(join_cpp(x, y), join_py(x, y), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(join_cpp(x, y, ref), join_py(x, y, ref), rtol=1e-5, atol=1e-6)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_ilp2_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_ilp2(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_ilp4_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_ilp4(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_ilp2_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_ilp2(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_ilp2(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_ilp4_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_ilp4(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_ilp4(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_acc2_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_acc2(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_acc4_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_acc4(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_acc2_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_acc2(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_acc2(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_acc4_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_acc4(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_acc4(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


def test_caches_warm_up_idempotently():
    x = torch.randn(4, 16, dtype=torch.float64)
    y = torch.randn(4, 16, dtype=torch.float64)
    a = gp_cpp(x, y)
    b = gp_cpp(x, y)
    torch.testing.assert_close(a, b)
    a2 = join_cpp(x, y)
    b2 = join_cpp(x, y)
    torch.testing.assert_close(a2, b2)
