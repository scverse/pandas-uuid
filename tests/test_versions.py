# SPDX-License-Identifier: MPL-2.0
"""Test UUID version handling."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid1, uuid4

import pandas as pd
import pytest

from pandas_uuid import UuidDtype
from pandas_uuid._pyarrow import HAS_PYARROW

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas.api.extensions import ExtensionArray

    from pandas_uuid import ArrowUuidArray, UuidArray, UuidStorage


# dtype


@pytest.mark.parametrize(("version", "name"), [(None, "uuid"), (4, "uuid[4]")])
def test_dtype_name(version: int | None, name: str) -> None:
    assert UuidDtype(version=version).name == name


@pytest.mark.parametrize("string", ["uuid", "uuid[1]", "uuid[8]"])
def test_dtype_from_string_roundtrip(string: str) -> None:
    assert UuidDtype.construct_from_string(string).name == string


@pytest.mark.parametrize("string", ["uuid[]", "uuid[0]", "uuid[9]", "int64", "uuid[x]"])
def test_dtype_from_string_error(string: str) -> None:
    with pytest.raises(TypeError, match=r"Cannot construct a 'UuidDtype'"):
        UuidDtype.construct_from_string(string)


def test_dtype_from_string_type_error() -> None:
    with pytest.raises(TypeError, match=r"expects a string"):
        UuidDtype.construct_from_string(1)  # ty:ignore[invalid-argument-type]


@pytest.mark.parametrize("version", [0, 9, -1])
def test_dtype_bad_version(version: int) -> None:
    with pytest.raises(ValueError, match=r"version must be None or in 1–8"):
        UuidDtype(version=version)


def test_dtype_version_affects_identity() -> None:
    """Versions have different field layouts, so they’re different dtypes."""
    assert UuidDtype("numpy") != UuidDtype("numpy", 4)
    assert UuidDtype("numpy", 4) != UuidDtype("numpy", 7)
    assert UuidDtype("numpy", 7) == UuidDtype("numpy", 7)


# generation


@pytest.mark.parametrize("version", [None, 4, 7])
def test_random_version(storage: UuidStorage, version: int | None) -> None:
    dtype = UuidDtype(storage, version)
    arr = dtype.construct_array_type().random(4, rng=0, dtype=dtype)
    assert arr.dtype == dtype
    for uuid in arr:
        assert uuid.version == (4 if version is None else version)
        assert uuid.variant == "specified in RFC 4122"


@pytest.mark.parametrize("version", [1, 3, 5, 6, 8])
def test_random_unsupported_version(storage: UuidStorage, version: int) -> None:
    dtype = UuidDtype(storage, version)
    with pytest.raises(NotImplementedError, match=r"version 4 or 7, not"):
        dtype.construct_array_type().random(2, rng=0, dtype=dtype)


def test_random_v7_timestamp(storage: UuidStorage) -> None:
    """The leading 48 bits must be a Unix timestamp in ms, per RFC 9562 §5.7."""
    dtype = UuidDtype(storage, 7)
    before = time.time_ns() // 1_000_000
    arr = dtype.construct_array_type().random(2, rng=0, dtype=dtype)
    after = time.time_ns() // 1_000_000
    for uuid in arr:
        assert before <= uuid.int >> 80 <= after


def test_random_reproducible(storage: UuidStorage) -> None:
    dtype = UuidDtype(storage, 4)
    cls = dtype.construct_array_type()
    assert list(cls.random(4, rng=42, dtype=dtype)) == list(
        cls.random(4, rng=42, dtype=dtype)
    )


# construction validation


def test_construct_version_mismatch(storage: UuidStorage) -> None:
    cls = UuidDtype(storage).construct_array_type()
    with pytest.raises(ValueError, match=r"index 1 has version 1.*specifies version 4"):
        cls([uuid4(), uuid1(), uuid4()], dtype=UuidDtype(storage, 4))


@pytest.mark.parametrize("length", [0, 2])
def test_construct_version_match(storage: UuidStorage, length: int) -> None:
    dtype = UuidDtype(storage, 4)
    arr = dtype.construct_array_type()([uuid4() for _ in range(length)], dtype=dtype)
    assert arr.dtype.version == 4


def test_construct_version_unspecified(storage: UuidStorage) -> None:
    """Without a version in the dtype, mixed versions are allowed."""
    dtype = UuidDtype(storage)
    arr = dtype.construct_array_type()([uuid4(), uuid1()], dtype=dtype)
    assert arr.dtype.version is None


# operations


def _random(
    storage: UuidStorage, version: int | None, rng: int
) -> UuidArray | ArrowUuidArray:
    dtype = UuidDtype(storage, version)
    return dtype.construct_array_type().random(2, rng=rng, dtype=dtype)


@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        pytest.param((4, 4), 4, id="same"),
        pytest.param((4, 7), None, id="mismatch"),
        pytest.param((4, None), None, id="unversioned"),
    ],
)
def test_concat_version(
    storage: UuidStorage, versions: tuple[int | None, ...], expected: int | None
) -> None:
    """Concatenation keeps a shared version and drops mismatched ones."""
    arrays = [_random(storage, version, i) for i, version in enumerate(versions)]

    concat = type(arrays[0])._concat_same_type(arrays)  # ty:ignore[invalid-argument-type]  # noqa: SLF001
    assert concat.dtype == UuidDtype(storage, expected)
    assert concat.tolist() == [uuid for arr in arrays for uuid in arr]

    series = pd.concat([pd.Series(arr) for arr in arrays])
    assert series.dtype == UuidDtype(storage, expected)
    assert series.tolist() == concat.tolist()


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow is not installed")
def test_concat_mixed_storage() -> None:
    """Storage kinds can’t be mixed, so pandas falls back to object."""
    series = [pd.Series(_random(storage, 4, 0)) for storage in ("numpy", "pyarrow")]
    assert pd.concat(series).dtype == object


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda arr: arr.copy(), id="copy"),
        pytest.param(lambda arr: arr[:1], id="slice"),
        pytest.param(lambda arr: arr.take([0]), id="take"),
        pytest.param(lambda arr: pd.Series(arr).array, id="series"),
    ],
)
def test_subset_keeps_version(
    storage: UuidStorage, op: Callable[[UuidArray | ArrowUuidArray], ExtensionArray]
) -> None:
    assert op(_random(storage, 7, 0)).dtype == UuidDtype(storage, 7)
