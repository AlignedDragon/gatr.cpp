"""Utility functions to test callables for consistency with the clifford library.

Adapted from `tests/helpers/geometric_algebra.py` in the geometric-algebra-transformer
project (Qualcomm Technologies, Inc., 2023).
"""

import clifford
import numpy as np
import torch
from clifford import pga as PGA

from tests.helpers.clifford import mv_list_to_tensor


def _sample_list_of_mv(batch_dims, rng):
    """Utility function that samples a list of multivectors."""
    total_batchsize = 1 if not batch_dims else int(np.prod(batch_dims))
    xs = clifford.randomMV(layout=PGA.layout, n=total_batchsize, rng=rng)
    if total_batchsize == 1:
        xs = [xs]
    return xs


def check_consistence_with_geometric_product(function, batch_dims=(1,), rng=None, **kwargs):
    """Checks whether a callable computes the geometric product."""
    xs = _sample_list_of_mv(batch_dims, rng)
    ys = _sample_list_of_mv(batch_dims, rng)

    xy_true = [(x * y) for x, y in zip(xs, ys)]
    xy_true = mv_list_to_tensor(xy_true, batch_dims)

    x_tensor = mv_list_to_tensor(xs, batch_dims)
    y_tensor = mv_list_to_tensor(ys, batch_dims)
    xy = function(x_tensor, y_tensor)

    torch.testing.assert_close(xy, xy_true, **kwargs)


def check_consistence_with_outer_product(function, batch_dims=(1,), rng=None, **kwargs):
    """Checks whether a callable computes the outer product."""
    xs = _sample_list_of_mv(batch_dims, rng)
    ys = _sample_list_of_mv(batch_dims, rng)

    xy_true = [(x ^ y) for x, y in zip(xs, ys)]
    xy_true = mv_list_to_tensor(xy_true, batch_dims)

    x_tensor = mv_list_to_tensor(xs, batch_dims)
    y_tensor = mv_list_to_tensor(ys, batch_dims)
    xy = function(x_tensor, y_tensor)

    torch.testing.assert_close(xy, xy_true, **kwargs)


def check_consistence_with_reversal(function, batch_dims=(1,), rng=None, **kwargs):
    """Checks whether a callable computes the reversal of a multivector."""
    xs = _sample_list_of_mv(batch_dims, rng)

    reversed_x_true = [~x for x in xs]
    reversed_x_true = mv_list_to_tensor(reversed_x_true, batch_dims)

    x_tensor = mv_list_to_tensor(xs, batch_dims)
    reversed_x = function(x_tensor)

    torch.testing.assert_close(reversed_x, reversed_x_true, **kwargs)


def check_consistence_with_grade_involution(function, batch_dims=(1,), rng=None, **kwargs):
    """Checks whether a callable computes the grade involution of a multivector."""
    xs = _sample_list_of_mv(batch_dims, rng)

    invol_x_true = [x.gradeInvol() for x in xs]
    invol_x_true = mv_list_to_tensor(invol_x_true, batch_dims)

    x_tensor = mv_list_to_tensor(xs, batch_dims)
    invol_x = function(x_tensor)

    torch.testing.assert_close(invol_x, invol_x_true, **kwargs)


def check_consistence_with_dual(function, batch_dims=(1,), rng=None, **kwargs):
    """Checks whether a callable computes the dual of a multivector."""
    xs = _sample_list_of_mv(batch_dims, rng)

    dual_x_true = [x.dual() for x in xs]
    dual_x_true = mv_list_to_tensor(dual_x_true, batch_dims)

    x_tensor = mv_list_to_tensor(xs, batch_dims)
    dual_x = function(x_tensor)

    torch.testing.assert_close(dual_x, dual_x_true, **kwargs)
