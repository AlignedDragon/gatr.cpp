"""Utility functions to test callables for equivariance with respect to Pin(3,0,1).

Adapted from `tests/helpers/equivariance.py` in the geometric-algebra-transformer
project (Qualcomm Technologies, Inc., 2023).
"""

import torch

from tests.helpers.clifford import SlowRandomPinTransform


def get_first_output(outputs):
    """Extracts the first output of a tuple of multiple outputs.

    If only one output is present, returns that.
    """
    if isinstance(outputs, tuple):
        return outputs[0]

    return outputs


def check_pin_equivariance(
    function,
    num_multivector_args=1,
    fn_kwargs=None,
    batch_dims=(1,),
    spin=False,
    rng=None,
    num_checks=3,
    **kwargs,
):
    """Checks whether a callable is Pin(3,0,1)- (or Spin(3,0,1)-) equivariant."""
    if fn_kwargs is None:
        fn_kwargs = {}

    if rng is not None:
        torch.manual_seed(rng.integers(100000))

    for _ in range(num_checks):
        inputs = torch.randn(num_multivector_args, *batch_dims, 16)
        transform = SlowRandomPinTransform(rng=rng, spin=spin)

        outputs = get_first_output(function(*inputs, **fn_kwargs))
        transformed_outputs = transform(outputs)

        transformed_inputs = transform(inputs)
        outputs_of_transformed = get_first_output(function(*transformed_inputs, **fn_kwargs))

        torch.testing.assert_close(transformed_outputs, outputs_of_transformed, **kwargs)


def check_pin_invariance(
    function,
    num_multivector_args=1,
    fn_kwargs=None,
    batch_dims=(1,),
    spin=False,
    rng=None,
    num_checks=3,
    **kwargs,
):
    """Checks whether a callable is Pin(3,0,1)- (or Spin(3,0,1)-) invariant."""
    if fn_kwargs is None:
        fn_kwargs = {}

    if rng is not None:
        torch.manual_seed(rng.integers(100000))

    for _ in range(num_checks):
        inputs = torch.randn(num_multivector_args, *batch_dims, 16)

        transform = SlowRandomPinTransform(rng=rng, spin=spin)
        transformed_inputs = transform(inputs)

        outputs = get_first_output(function(*inputs, **fn_kwargs))
        outputs_of_transformed = get_first_output(function(*transformed_inputs, **fn_kwargs))

        torch.testing.assert_close(outputs, outputs_of_transformed, **kwargs)
