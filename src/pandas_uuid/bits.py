# SPDX-License-Identifier: MPL-2.0
"""Helper functions for bit manipulation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def n_lsb(x: int) -> int:
    """Get number of least significant bits in x."""
    return (1 << x) - 1


def n_to_m(n: int, m: int) -> int:
    """Get bit mask from n to m."""
    return n_lsb(n) & ~n_lsb(m)


def set_bits[T: NDArray | int](
    i: T, n: int, m: int, *, pat: int, out: T | None = None
) -> T:
    """Set bits `n` to `m` in `i` to `pat`."""
    c = partial(as_scalar, dtype=i.dtype)
    # zero the bits we want to set
    i_with_gap = np.bitwise_and(i, c(~n_to_m(n, m)), out=out)
    # set them to `pat`
    return np.bitwise_or(i_with_gap, c(pat << m), out=out)


def as_scalar[T: np.generic](x: object, dtype: np.dtype[T]) -> NDArray[T]:
    """Convert `x` to a scalar of type `dtype`."""
    return np.array([x]).astype(dtype).item()
