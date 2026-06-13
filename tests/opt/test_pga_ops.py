import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings

from ezgatr.nn.functional import equi_join as join_py
from ezgatr.nn.functional import geometric_product as gp_py
from ezgatr.nn.functional import equi_linear as el_py
from ezgatr.nn.functional import scaler_gated_gelu as gelu_py
from ezgatr.opt import equi_join as join_cpp
from ezgatr.opt import (
    equi_join_v0, equi_join_v1, equi_join_v2, equi_join_v3,
    equi_join_v2_1, equi_join_v2_2, equi_join_v2_3, equi_join_v2_4,
    equi_join_v2_5, equi_join_v2_6, equi_join_v2_7,
)
from ezgatr.opt import geometric_bilinear_v3_1
from ezgatr.opt import geometric_product as gp_cpp
from ezgatr.opt import (
    geometric_product_v0, geometric_product_v1, geometric_product_v2, geometric_product_v3,
    geometric_product_v2_1, geometric_product_v2_2, geometric_product_v2_3, geometric_product_v2_4,
    geometric_product_v2_5, geometric_product_v2_6, geometric_product_v2_7,
)
from ezgatr.opt import equi_linear as el_cpp
from ezgatr.opt import equi_linear_v0 as el_cpp_v0
from ezgatr.opt import equi_linear_v1 as el_cpp_v1
from ezgatr.opt import equi_linear_v2 as el_cpp_v2
from ezgatr.opt import equi_linear_v3 as el_cpp_v3
from ezgatr.opt import (
    scaler_gated_gelu_ver_0 as gelu_v0,
    scaler_gated_gelu_ver_1 as gelu_v1,
    scaler_gated_gelu_ver_2 as gelu_v2,
    scaler_gated_gelu_ver_3 as gelu_v3,
)


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


batch_shape_bil = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=2)


@given(batch_shape_bil, st.integers(min_value=1, max_value=5))
@settings(deadline=None, max_examples=20)
def test_geometric_bilinear_v3_1_matches_separate(batch, inter):
    """Fused bilinear == cat([gp(lg,rg), join(lj,rj,ref)]) (fp32, AVX2 path)."""
    torch.manual_seed(0)
    p = torch.randn(*batch, 4 * inter, 16, dtype=torch.float32)
    ref = torch.randn(*batch, 1, 16, dtype=torch.float32)
    lg, rg, lj, rj = torch.split(p, inter, dim=-2)

    # with reference
    out = geometric_bilinear_v3_1(p, ref)
    exp = torch.cat([gp_py(lg, rg), join_py(lj, rj, ref)], dim=-2)
    torch.testing.assert_close(out, exp, rtol=1e-5, atol=1e-6)

    # without reference
    out_nr = geometric_bilinear_v3_1(p, None)
    exp_nr = torch.cat([gp_py(lg, rg), join_py(lj, rj)], dim=-2)
    torch.testing.assert_close(out_nr, exp_nr, rtol=1e-5, atol=1e-6)


def test_geometric_bilinear_v3_1_matches_separate_cpp_ops():
    """Fused bilinear is bit-identical to the separate v3 gp/join ops + cat."""
    torch.manual_seed(0)
    inter = 6
    p = torch.randn(5, 4 * inter, 16, dtype=torch.float32)
    ref = torch.randn(5, 1, 16, dtype=torch.float32)
    lg, rg, lj, rj = torch.split(p, inter, dim=-2)
    out = geometric_bilinear_v3_1(p, ref)
    exp = torch.cat([geometric_product_v3(lg, rg), equi_join_v3(lj, rj, ref)], dim=-2)
    torch.testing.assert_close(out, exp, rtol=0, atol=0)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_linear_matches_python_normalized(batch):
    in_ch, out_ch = 3, 5
    x = torch.randn(*batch, in_ch, 16, dtype=torch.float64)
    w = torch.randn(out_ch, in_ch, 9, dtype=torch.float64)
    torch.testing.assert_close(el_cpp(x, w), el_py(x, w), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_linear_matches_python_unnormalized(batch):
    in_ch, out_ch = 3, 5
    x = torch.randn(*batch, in_ch, 16, dtype=torch.float64)
    w = torch.randn(out_ch, in_ch, 9, dtype=torch.float64)
    torch.testing.assert_close(
        el_cpp(x, w, normalize_basis=False),
        el_py(x, w, normalize_basis=False),
        rtol=1e-10, atol=1e-12,
    )


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_linear_matches_python_with_bias(batch):
    in_ch, out_ch = 3, 5
    x = torch.randn(*batch, in_ch, 16, dtype=torch.float64)
    w = torch.randn(out_ch, in_ch, 9, dtype=torch.float64)
    b = torch.randn(out_ch, dtype=torch.float64)
    torch.testing.assert_close(el_cpp(x, w, b), el_py(x, w, b), rtol=1e-10, atol=1e-12)


def test_equi_linear_float32():
    torch.manual_seed(0)
    in_ch, out_ch = 4, 8
    x = torch.randn(2, in_ch, 16)
    w = torch.randn(out_ch, in_ch, 9)
    b = torch.randn(out_ch)
    torch.testing.assert_close(el_cpp(x, w, b), el_py(x, w, b), rtol=1e-5, atol=1e-6)


def test_equi_linear_version_exports_and_dispatch_match_python():
    torch.manual_seed(0)
    in_ch, out_ch = 3, 5
    x = torch.randn(2, in_ch, 16, dtype=torch.float64)
    w = torch.randn(out_ch, in_ch, 9, dtype=torch.float64)
    b = torch.randn(out_ch, dtype=torch.float64)

    versioned = (el_cpp_v0, el_cpp_v1, el_cpp_v2, el_cpp_v3)
    for version, fn in enumerate(versioned):
        for normalize_basis in (False, True):
            expected = el_py(x, w, b, normalize_basis=normalize_basis)
            named = fn(x, w, b, normalize_basis=normalize_basis)
            dispatched = el_cpp(x, w, b, normalize_basis=normalize_basis, version=version)

            torch.testing.assert_close(named, expected, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(dispatched, expected, rtol=1e-10, atol=1e-12)

    torch.testing.assert_close(el_cpp(x, w, b), el_cpp_v0(x, w, b), rtol=1e-10, atol=1e-12)
    with pytest.raises(RuntimeError, match="unknown version"):
        el_cpp(x, w, b, version=4)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_v2_3_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_v2_3(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_v2_4_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_v2_4(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_v2_3_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_v2_3(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_v2_3(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_v2_4_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_v2_4(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_v2_4(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_v2_1_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_v2_1(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_geometric_product_v2_2_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(geometric_product_v2_2(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_v2_1_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_v2_1(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_v2_1(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=20)
def test_equi_join_v2_2_matches_python(batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(equi_join_v2_2(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(equi_join_v2_2(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


def test_caches_warm_up_idempotently():
    x = torch.randn(4, 16, dtype=torch.float64)
    y = torch.randn(4, 16, dtype=torch.float64)
    a = gp_cpp(x, y)
    b = gp_cpp(x, y)
    torch.testing.assert_close(a, b)
    a2 = join_cpp(x, y)
    b2 = join_cpp(x, y)
    torch.testing.assert_close(a2, b2)


# ---------------------------------------------------------------------------
# Parametrized correctness tests — every exported version against Python ref
# ---------------------------------------------------------------------------

# v0–v2.x support float64; v3 is float32-only (AVX2/FMA SIMD path).
ALL_GP_F64 = [
    geometric_product_v0, geometric_product_v1,
    geometric_product_v2, geometric_product_v2_1, geometric_product_v2_2,
    geometric_product_v2_3, geometric_product_v2_4, geometric_product_v2_5,
    geometric_product_v2_6, geometric_product_v2_7,
]
ALL_GP_F64_IDS = ["v0","v1","v2","v2_1","v2_2","v2_3","v2_4","v2_5","v2_6","v2_7"]

ALL_JOIN_F64 = [
    equi_join_v0, equi_join_v1,
    equi_join_v2, equi_join_v2_1, equi_join_v2_2, equi_join_v2_3,
    equi_join_v2_4, equi_join_v2_5, equi_join_v2_6, equi_join_v2_7,
]
ALL_JOIN_F64_IDS = ["v0","v1","v2","v2_1","v2_2","v2_3","v2_4","v2_5","v2_6","v2_7"]

# v3 on float64 / "none" delegates to v2 (exact), so it joins the tight-tolerance
# float64 sweep; its AVX2 float32 "tanh" path is checked separately below.
ALL_GELU_VERSIONS = [gelu_v0, gelu_v1, gelu_v2, gelu_v3]
ALL_GELU_IDS = ["v0", "v1", "v2", "v3"]


@pytest.mark.parametrize("fn", ALL_GP_F64, ids=ALL_GP_F64_IDS)
@given(batch_shape)
@settings(deadline=None, max_examples=15)
def test_geometric_product_all_versions(fn, batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(fn(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=15)
def test_geometric_product_v3_float32(batch):
    x = torch.randn(*batch, 16, dtype=torch.float32)
    y = torch.randn(*batch, 16, dtype=torch.float32)
    torch.testing.assert_close(geometric_product_v3(x, y), gp_py(x, y), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fn", ALL_JOIN_F64, ids=ALL_JOIN_F64_IDS)
@given(batch_shape)
@settings(deadline=None, max_examples=15)
def test_equi_join_all_versions(fn, batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    y = torch.randn(*batch, 16, dtype=torch.float64)
    ref = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(fn(x, y),      join_py(x, y),      rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fn(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@given(batch_shape)
@settings(deadline=None, max_examples=15)
def test_equi_join_v3_float32(batch):
    x = torch.randn(*batch, 16, dtype=torch.float32)
    y = torch.randn(*batch, 16, dtype=torch.float32)
    ref = torch.randn(*batch, 16, dtype=torch.float32)
    torch.testing.assert_close(equi_join_v3(x, y),      join_py(x, y),      rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(equi_join_v3(x, y, ref), join_py(x, y, ref), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fn", ALL_GELU_VERSIONS, ids=ALL_GELU_IDS)
@given(batch_shape)
@settings(deadline=None, max_examples=15)
def test_scaler_gated_gelu_all_versions(fn, batch):
    x = torch.randn(*batch, 16, dtype=torch.float64)
    torch.testing.assert_close(fn(x),         gelu_py(x),         rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fn(x, "tanh"), gelu_py(x, "tanh"), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fn(x, "none"), gelu_py(x, "none"), rtol=1e-10, atol=1e-12)


# Sizes spanning the 8-wide AVX2 block boundary (incl. < 8 scalar tail, and a
# non-multiple-of-8 that mixes vector body + tail).
@pytest.mark.parametrize("n", [1, 4, 7, 8, 13, 64, 257])
def test_scaler_gated_gelu_v3_float32(n):
    x = torch.randn(n, 16, dtype=torch.float32)
    # AVX2 tanh fast path: vectorized expf approximation -> float32 tolerance.
    torch.testing.assert_close(gelu_v3(x, "tanh"), gelu_py(x, "tanh"), rtol=2e-4, atol=2e-6)
    # "none" (erf) path delegates to the exact ver_2 implementation.
    torch.testing.assert_close(gelu_v3(x, "none"), gelu_py(x, "none"), rtol=1e-5, atol=1e-6)
