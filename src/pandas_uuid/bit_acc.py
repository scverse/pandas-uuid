# SPDX-License-Identifier: MPL-2.0
"""Helper functions for bit manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


__all__ = ["bits"]


@dataclass(frozen=True)
class bits:  # noqa: N801
    """Element-wise bit accessor: `bac(arr)[start:stop] = 0b…`."""

    arr: NDArray

    def __setitem__(self, s: slice[int, int, Literal[1] | None], pat: int, /) -> None:
        """Set bits `start:stop` to `pat`."""
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
