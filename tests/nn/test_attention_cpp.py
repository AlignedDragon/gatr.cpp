from __future__ import annotations

import pytest
import torch

from ezgatr.nn.functional.attention import equi_geometric_attention
from ezgatr.nn.functional.attention_cpp import equi_geometric_attention_cpp


def _make_inputs():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 4, 16)
    k = torch.randn(2, 3, 5, 4, 16)
    v = torch.randn(2, 3, 5, 4, 16)
    return q, k, v


@pytest.mark.parametrize(
    ("kinds", "weight"),
    [
        ({"ipa": None}, None),
        ({"daa": None}, None),
        ({"ipa": None, "daa": {"eps": 1e-3}}, [0.5, 1.25]),
    ],
)
def test_equi_geometric_attention_cpp_matches_python(kinds, weight):
    q, k, v = _make_inputs()

    expected, expected_scalar = equi_geometric_attention(
        q,
        k,
        v,
        kinds=kinds,
        weight=weight,
        dropout_p=0.0,
        is_causal=False,
    )
    actual, actual_scalar = equi_geometric_attention_cpp(
        q,
        k,
        v,
        kinds=kinds,
        weight=weight,
        dropout_p=0.0,
        is_causal=False,
    )

    assert expected_scalar is None
    assert actual_scalar is None
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_equi_geometric_attention_cpp_supports_causal_attention():
    q, k, v = _make_inputs()

    expected, _ = equi_geometric_attention(
        q,
        k,
        v,
        kinds={"ipa": None, "daa": None},
        is_causal=True,
        dropout_p=0.0,
    )
    actual, _ = equi_geometric_attention_cpp(
        q,
        k,
        v,
        kinds={"ipa": None, "daa": None},
        is_causal=True,
        dropout_p=0.0,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
