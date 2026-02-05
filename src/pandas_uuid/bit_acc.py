# SPDX-License-Identifier: MPL-2.0
"""Helper functions for bit manipulation."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


__all__ = ["bits"]


@dataclass(frozen=True)
class bits:  # noqa: N801
    """Element-wise bit accessor.

    Works on any dtype that accepts arbitrary byte patterns,
    i.e. void, bytes, and unsigned integers.

    Examples
    --------
    >>> arr = np.array([0b00000000, 0b11111111], dtype=np.uint8)
    >>> bits(arr)[2:6] = 0b0011
    >>> [f"{e:08b}" for e in arr]
    ['00001100', '11001111']

    """

    arr: NDArray
    _: KW_ONLY
    force: bool = False

    def __post_init__(self) -> None:  # noqa: D105
        if self.arr.ndim != 1:
            raise NotImplementedError
        if not self.force and (
            self.arr.dtype.kind not in {"u", "V", "S"}
            or self.arr.dtype.fields is not None
        ):
            msg = f"dtype must be  {self.arr.dtype}"
            raise TypeError(msg)

    def __setitem__(self, s: slice[int, int, Literal[1] | None], pat: int, /) -> None:
        """Set each element’s bits `s.start:s.stop` to `pat`."""
        if not isinstance(s, slice) or s.step not in {1, None}:
            msg = f"index must be a range with step 1/None, not {s!r}"
            raise TypeError(msg)
        # create a mutable view into the to-be-modified bytes
        v = self.arr.view(np.uint8).reshape(-1, self.arr.itemsize)
        end = self.arr.itemsize * 8
        # zero out the bits we want to replace
        v &= ~self._to_scalar_array((2 ** (s.stop - s.start) - 1) << (end - s.stop))
        # set the new bits
        v |= self._to_scalar_array(pat << (end - s.stop))

    def _to_scalar_array(self, val: int) -> NDArray[np.uint8]:
        n = self.arr.itemsize
        return np.array([val.to_bytes(n)], dtype=f"V{n}").view(np.uint8).reshape(-1, n)
