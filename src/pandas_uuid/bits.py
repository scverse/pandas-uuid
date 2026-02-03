# SPDX-License-Identifier: MPL-2.0
"""Helper functions for bit manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


__all__ = ["bac"]


@dataclass(frozen=True)
class bac[T: NDArray]:  # noqa: D101, N801
    arr: T

    def __setitem__(
        self, index: slice[int, int, Literal[1] | None], pat: int, /
    ) -> None:
        """Set bits `start:stop` to `pat`."""
        if not isinstance(index, slice) or index.step not in {1, None}:
            msg = f"index must be a range with step 1/None, not {index!r}"
            raise TypeError(msg)
        start, stop = index.start, index.stop
        if (i_byte := stop // 8) != start // 8:
            msg = "n and m must be in the same byte"
            raise ValueError(msg)
        stop, start = (np.uint8(x % 8) for x in (stop, start))
        v = self.arr.view(np.uint8)[i_byte :: self.arr.dtype.itemsize]
        v &= ~np.uint8((2 ** (stop - start) - 1) << (8 - stop))
        v |= pat << (8 - stop)
