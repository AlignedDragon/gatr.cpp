"""Correctness tests for the loop-unroll (v2_5/v2_6), tiling (v2_7) and
vectorized (v3) geometric_product / equi_join variants.

Kept separate from test_pga_ops.py (which imports pre-rename friendly aliases)
so these run independently. Each variant must match the reference Python op to
bit tolerance; batch sizes are chosen to exercise both the unrolled/SIMD main
loop and the N % K remainder tail (K in {2, 4} for unroll, 4 for the AVX block).
"""
import pytest
import torch

from ezgatr.nn.functional import equi_join as join_py
from ezgatr.nn.functional import geometric_product as gp_py
from ezgatr import opt

VARIANTS = ["v2_5", "v2_6", "v2_7", "v3", "v3_1"]
# 1..3 hit the v3 tail only; 5,7,9,65 hit a tail after the main loop.
SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 64, 65, 100, 257]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n", SIZES)
def test_variants_match_python_fp64(variant, n):
    torch.manual_seed(n)
    x = torch.randn(n, 16, dtype=torch.float64)
    y = torch.randn(n, 16, dtype=torch.float64)
    ref = torch.randn(n, 16, dtype=torch.float64)

    gp = getattr(opt, f"geometric_product_{variant}")
    jo = getattr(opt, f"equi_join_{variant}")

    torch.testing.assert_close(gp(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(jo(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(jo(x, y, ref), join_py(x, y, ref), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_match_python_fp32(variant):
    """fp32 takes the scalar fallback for v3; the others are dtype-generic."""
    torch.manual_seed(0)
    x = torch.randn(257, 16, dtype=torch.float32)
    y = torch.randn(257, 16, dtype=torch.float32)
    ref = torch.randn(257, 16, dtype=torch.float32)

    gp = getattr(opt, f"geometric_product_{variant}")
    jo = getattr(opt, f"equi_join_{variant}")

    torch.testing.assert_close(gp(x, y), gp_py(x, y), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(jo(x, y, ref), join_py(x, y, ref), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_match_python_batched_shape(variant):
    """Multi-dim leading shape (flattened to N internally)."""
    torch.manual_seed(1)
    x = torch.randn(3, 17, 16, dtype=torch.float64)
    y = torch.randn(3, 17, 16, dtype=torch.float64)
    gp = getattr(opt, f"geometric_product_{variant}")
    jo = getattr(opt, f"equi_join_{variant}")
    torch.testing.assert_close(gp(x, y), gp_py(x, y), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(jo(x, y), join_py(x, y), rtol=1e-10, atol=1e-12)
