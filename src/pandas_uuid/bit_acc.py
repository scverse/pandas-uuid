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
        if (i_byte := s.start // 8) != (s.stop - 1) // 8:
            msg = "n and m must be in the same byte"
            raise ValueError(msg)
        # create a mutable view into the to-be-modified bytes
        v = self.arr.view(np.uint8)[i_byte :: self.arr.dtype.itemsize]
        # adjust the slice to fit into the byte
        s = slice(*(np.uint8(x % 8) for x in (s.start, s.stop)))
        # zero out the bits we want to replace
        v &= ~np.uint8((2 ** (s.stop - s.start) - 1) << (8 - s.stop))
        # set the new bits
        v |= pat << (8 - s.stop)
