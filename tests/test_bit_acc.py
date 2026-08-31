# SPDX-License-Identifier: MPL-2.0
"""Test the element-wise bit accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas_uuid.bit_acc import bits

if TYPE_CHECKING:
    import numpy.typing as npt


@pytest.mark.parametrize(
    ("start", "stop", "pat", "expected"),
    [
        pytest.param(2, 6, 0b0011, ["00001100", "11001111"], id="within-byte"),
        pytest.param(0, 8, 0b10101010, ["10101010", "10101010"], id="whole-byte"),
        pytest.param(0, 1, 0b1, ["10000000", "11111111"], id="single-bit"),
    ],
)
def test_setitem(start: int, stop: int, pat: int, expected: list[str]) -> None:
    arr = np.array([0b00000000, 0b11111111], dtype=np.uint8)
    bits(arr)[start:stop] = pat
    assert [f"{e:08b}" for e in arr] == expected


def test_setitem_across_byte_boundary() -> None:
    """A field spanning two bytes must not disturb its neighbors."""
    arr = np.array([b"\x00\x00", b"\xff\xff"], dtype="V2")
    bits(arr)[6:10] = 0b1010
    assert [f"{int.from_bytes(e.tobytes()):016b}" for e in arr] == [
        "0000001010000000",
        "1111111010111111",
    ]


def test_setitem_wide_field() -> None:
    """A 48-bit field in a 16-byte record, as UUIDv7 needs."""
    arr = np.zeros(2, dtype="V16")
    bits(arr)[0:48] = 0xDEADBEEFCAFE
    assert arr[0].tobytes().hex() == "deadbeefcafe" + "00" * 10


def test_getitem_roundtrip() -> None:
    """Whatever `__setitem__` writes, `__getitem__` must read back."""
    arr = np.zeros(3, dtype="V16")
    for s, pat in [
        (slice(0, 48), 0xDEADBEEFCAFE),
        (slice(48, 52), 7),
        (slice(64, 66), 2),
    ]:
        bits(arr)[s] = pat
        assert list(bits(arr)[s]) == [pat] * 3


def test_getitem_neighbors() -> None:
    """Reading a field must ignore the bits around it."""
    arr = np.array([b"\xff\xff"], dtype="V2")
    bits(arr)[6:10] = 0b0000
    assert bits(arr)[0:6][0] == 0b111111
    assert bits(arr)[6:10][0] == 0b0000
    assert bits(arr)[10:16][0] == 0b111111


def test_getitem_too_wide() -> None:
    arr = np.zeros(2, dtype="V16")
    with pytest.raises(NotImplementedError, match=r"up to 64 bits"):
        bits(arr)[0:72]


@pytest.mark.parametrize("s", [slice(0, 0), slice(4, 2), slice(-1, 4), slice(0, 9)])
def test_out_of_bounds(s: slice) -> None:
    arr = np.zeros(2, dtype=np.uint8)
    with pytest.raises(IndexError, match=r"must be a range within 0:8"):
        bits(arr)[s]


@pytest.mark.parametrize("dtype", ["f8", "i4", np.dtype([("a", "u1")])])
def test_bad_dtype(dtype: npt.DTypeLike) -> None:
    with pytest.raises(TypeError, match=r"dtype must be an unsigned int"):
        bits(np.zeros(2, dtype=dtype))


def test_bad_ndim() -> None:
    with pytest.raises(NotImplementedError):
        bits(np.zeros((2, 2), dtype=np.uint8))


@pytest.mark.parametrize("index", [slice(0, 4, 2), 3])
def test_bad_index(index: object) -> None:
    arr = np.zeros(2, dtype=np.uint8)
    with pytest.raises(TypeError, match=r"index must be a range"):
        bits(arr)[index] = 0b1  # ty:ignore[invalid-assignment]
