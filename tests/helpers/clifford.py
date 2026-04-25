"""Geometric algebra operations based on the clifford library.

Adapted from `gatr.utils.clifford` in the geometric-algebra-transformer project
(Qualcomm Technologies, Inc., 2023). Test-only utilities — slow PGA reference
implementations for cross-checking ezgatr's primitives.
"""

from typing import Optional

import clifford
import numpy as np
import torch
from clifford import pga as PGA


def np_to_mv(array):
    """Shorthand to transform a numpy array to a PGA multivector."""
    return clifford.MultiVector(PGA.layout, value=array)


def tensor_to_mv(tensor):
    """Shorthand to transform a torch tensor to a PGA multivector."""
    return np_to_mv(tensor.detach().cpu().numpy())


def tensor_to_mv_list(tensor):
    """Transforms a torch.Tensor to a list of multivector objects."""

    tensor = tensor.reshape((-1, 16))
    mv_list = [tensor_to_mv(x) for x in tensor]

    return mv_list


def mv_list_to_tensor(multivectors, batch_shape=None):
    """Transforms a list of multivector objects to a torch.Tensor."""

    tensor = torch.from_numpy(np.array([mv.value for mv in multivectors])).to(torch.float32)
    if batch_shape is not None:
        tensor = tensor.reshape(*batch_shape, 16)

    return tensor


def sample_pin_multivector(spin: bool = False, rng: Optional[np.random.Generator] = None):
    """Samples from the Pin(3,0,1) group as a product of reflections."""

    if rng is None:
        rng = np.random.default_rng()

    if spin:
        i = np.random.randint(3) * 2
    else:
        i = np.random.randint(5)

    if i == 0:
        return PGA.blades[""]

    multivector = 1.0
    for _ in range(i):
        vector = np.zeros(16)
        vector[1:5] = rng.normal(size=4)
        vector_mv = np_to_mv(vector)
        vector_mv = vector_mv / vector_mv.mag2() ** 0.5

        multivector = multivector * vector_mv

    return multivector


def get_parity(mv):
    """Gets parity of a clifford multivector.

    Returns True if pure-odd-grade, False if pure-even-grade,
    raises a RuntimeError if mixed.
    """
    if mv == mv.even:
        return False
    if mv == mv.odd:
        return True
    raise RuntimeError(f"Mixed-grade multivector: {mv}")


def sandwich(u, x):
    """Sandwich product of two clifford multivectors.

    Given a Pin element u and a PGA element x, computes
    ``sandwich(x, u) = (-1)^(grade(u) * grade(x)) u x u^{-1}``.
    """

    if get_parity(u):
        return u * x.gradeInvol() * u.shirokov_inverse()

    return u * x * u.shirokov_inverse()


class SlowRandomPinTransform:
    """Random Pin transform on a multivector torch.Tensor.

    Slow, only used for testing purposes. Breaks computational graph.
    """

    def __init__(self, spin=False, rng=None):
        super().__init__()
        self._u = sample_pin_multivector(spin, rng)
        self._u_inverse = self._u.shirokov_inverse()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply Pin transformation to multivector inputs."""
        assert inputs.shape[-1] == 16
        batch_dims = inputs.shape[:-1]

        inputs_mv = tensor_to_mv_list(inputs)
        outputs_mv = [sandwich(self._u, x) for x in inputs_mv]
        outputs = mv_list_to_tensor(outputs_mv, batch_shape=batch_dims)

        return outputs.to(device=inputs.device, dtype=inputs.dtype)
