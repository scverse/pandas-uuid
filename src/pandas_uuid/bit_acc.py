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
    >>> bits(arr)[2:6]
    array([3, 3], dtype=uint64)

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
            msg = (
                "dtype must be an unsigned int, void, or bytes dtype without fields, "
                f"not {self.arr.dtype}"
            )
            raise TypeError(msg)

    def __setitem__(self, s: slice[int, int, Literal[1] | None], pat: int, /) -> None:
        """Set each element’s bits `s.start:s.stop` to `pat`."""
        start, stop = self._bounds(s)
        # create a mutable view into the to-be-modified bytes
        v = self.arr.view(np.uint8).reshape(-1, self.arr.itemsize)
        end = self.arr.itemsize * 8
        # zero out the bits we want to replace
        v &= ~self._to_scalar_array((2 ** (stop - start) - 1) << (end - stop))
        # set the new bits
        v |= self._to_scalar_array(pat << (end - stop))

    def __getitem__(
        self, s: slice[int, int, Literal[1] | None], /
    ) -> NDArray[np.uint64]:
        """Read each element’s bits `s.start:s.stop` as an unsigned integer."""
        start, stop = self._bounds(s)
        # bytes the field lives in, and how far it sits from their right edge
        lo, hi = start // 8, -(-stop // 8)
        if (n := hi - lo) > 8:  # noqa: PLR2004
            msg = f"can only read up to 64 bits at once, not bits {start}:{stop}"
            raise NotImplementedError(msg)
        v = np.ascontiguousarray(self.arr).view(np.uint8).reshape(-1, self.arr.itemsize)
        # right-align the field’s bytes in a big-endian uint64, then shift it down
        padded = np.zeros((len(v), 8), dtype=np.uint8)
        padded[:, -n:] = v[:, lo:hi]
        ints = padded.view(">u8").reshape(-1)
        return (ints >> (hi * 8 - stop)) & (2 ** (stop - start) - 1)

    def _bounds(self, s: slice[int, int, Literal[1] | None]) -> tuple[int, int]:
        if (
            not isinstance(s, slice)
            or s.step not in {1, None}
            or not isinstance(s.start, int)
            or not isinstance(s.stop, int)
        ):
            msg = f"index must be a range with step 1/None, not {s!r}"
            raise TypeError(msg)
        end = self.arr.itemsize * 8
        if not 0 <= s.start < s.stop <= end:
            msg = f"index must be a range within 0:{end}, not {s.start}:{s.stop}"
            raise IndexError(msg)
        return s.start, s.stop

    def _to_scalar_array(self, val: int) -> NDArray[np.uint8]:
        n = self.arr.itemsize
        return np.array([val.to_bytes(n)], dtype=f"V{n}").view(np.uint8).reshape(-1, n)
