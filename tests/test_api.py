# SPDX-License-Identifier: MPL-2.0
"""Test non-construction APIs."""

from __future__ import annotations

from itertools import batched, product
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pandas_uuid import ArrowUuidArray, UuidArray, UuidDtype
from pandas_uuid._pyarrow import HAS_PYARROW

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal
    from uuid import UUID

    from pandas._typing import ScalarIndexer, SequenceIndexer, TakeIndexer

    from pandas_uuid import UuidStorage


skipif_no_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow is not installed"
)


def test_isna(storage: UuidStorage, xfail_if_numpy_and_na: Callable[..., None]) -> None:
    xfail_if_numpy_and_na()
    arr = pd.array([uuid4(), uuid4(), None], dtype=UuidDtype(storage))
    assert arr.isna().tolist() == [False, False, True]


def test_isna_numpy() -> None:
    arr = pd.array([uuid4(), uuid4()], dtype=UuidDtype("numpy"))
    assert arr.isna().tolist() == [False, False]


@pytest.mark.parametrize(
    ("values", "index", "expected"),
    [
        pytest.param([u0 := uuid4()], 0, u0, id="only-py"),
        pytest.param([u1 := uuid4()], np.int64(0), u1, id="only-np"),
        pytest.param([pd.NA], 0, pd.NA, id="only-na"),
        pytest.param([u0, u1], 1, u1, id="second"),
        pytest.param([u0, u1], slice(None), [u0, u1], id="all-slice"),
        pytest.param([pd.NA, u1], slice(None), [pd.NA, u1], id="all-slice-na"),
        pytest.param([u0, u1], slice(None, None, -1), [u1, u0], id="inv-slice"),
        pytest.param([u0, u1], [0, 1], [u0, u1], id="all-list"),
        pytest.param([u0, u1], [1, 0], [u1, u0], id="all-list-reorder"),
    ],
)
def test_getitem(
    storage: UuidStorage,
    xfail_if_numpy_and_na: Callable[..., None],
    values: list[UUID | None],
    index: ScalarIndexer | SequenceIndexer,
    expected: UUID | list[UUID],
) -> None:
    xfail_if_numpy_and_na(values)
    arr = pd.array(values, dtype=UuidDtype(storage))
    match expected:
        case list():
            assert arr[index].tolist() == expected
        case _ if expected is pd.NA:
            assert pd.isna(arr[index])
        case _:
            assert arr[index] == expected


@pytest.mark.parametrize(
    ("values", "index_list", "expected"),
    [
        pytest.param([u0 := uuid4()], [0], [u0], id="only"),
        pytest.param([None], [0], [pd.NA], id="only-na"),
        pytest.param([u0, u1], [1], [u1], id="second"),
        pytest.param([u0, u1], [0, 1], [u0, u1], id="all-list"),
        pytest.param([u0, u1], [1, 0], [u1, u0], id="all-list-reorder"),
    ],
)
@pytest.mark.parametrize(
    "index_type",
    [
        pytest.param(np.array, id="array-numpy"),
        pytest.param(pd.array, id="array-pandas"),
        pd.Index,
        pd.Series,
    ],
)
def test_take(
    storage: UuidStorage,
    xfail_if_numpy_and_na: Callable[..., None],
    values: list[UUID | None],
    index_list: list[int | np.integer],
    index_type: Callable[..., TakeIndexer],
    expected: UUID | list[UUID],
) -> None:
    xfail_if_numpy_and_na(values)
    index = index_type(index_list)
    arr = pd.array(values, dtype=UuidDtype(storage))
    assert arr.take(index).tolist() == expected


def test_take_fill(request: pytest.FixtureRequest, storage: UuidStorage) -> None:
    if storage == "numpy":
        request.applymarker(pytest.mark.xfail(raises=ValueError))
    arr = pd.array([uuid4(), uuid4()], dtype=UuidDtype(storage))
    result = arr.take([1, -1], allow_fill=True).tolist()
    assert result == [arr[1], pd.NA]


@pytest.mark.parametrize(
    "index", [pytest.param(999, id="oob"), pytest.param("", id="type")]
)
def test_getitem_error(index: Any) -> None:  # noqa: ANN401
    arr = pd.array([uuid4(), uuid4()], dtype=UuidDtype("numpy"))
    with pytest.raises(IndexError):
        arr[index]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param([u0 := uuid4(), u1 := uuid4()], [u0, u1], [True] * 2, id="same"),
        pytest.param([u0, u1], [u0, u0], [True, False], id="different"),
        pytest.param([u0, pd.NA], [u0, u1], [True, pd.NA], id="na-left"),
        pytest.param([u0, u1], [u0, pd.NA], [True, pd.NA], id="na-right"),
    ],
)
@pytest.mark.parametrize(
    ("storage_left", "storage_right"),
    [
        pytest.param(
            left,
            right,
            id=f"{left}-{right}",
            marks=skipif_no_pyarrow if "pyarrow" in (left, right) else [],
        )
        for left, right in product(["numpy", "pyarrow", "list"], repeat=2)
        if left != "list" or right != "list"
    ],
)
def test_eq(
    request: pytest.FixtureRequest,
    storage_left: UuidStorage | Literal["list"],
    storage_right: UuidStorage | Literal["list"],
    left: list[UUID | None],
    right: list[UUID | None],
    expected: list[bool | None],
) -> None:
    combs: list[tuple[list[UUID | None], UuidStorage | Literal["list"]]] = [
        (left, storage_left),
        (right, storage_right),
    ]
    # xfail when numpy would store NAs
    if any(storage == "numpy" and pd.isna(values).any() for values, storage in combs):
        request.applymarker(pytest.mark.xfail(raises=TypeError))
    # xfail when numpy would compare to a list containing NAs
    if all(storage != "pyarrow" for _, storage in combs) and any(
        pd.isna(values).any() for values, _ in combs
    ):
        request.applymarker(pytest.mark.xfail(raises=TypeError))
    left_conv, right_conv = (
        values
        if storage == "list"
        else cast(
            "UuidArray | ArrowUuidArray",
            pd.array(values, dtype=UuidDtype(storage)),
        )
        for values, storage in combs
    )

    cmp = left_conv == right_conv

    arr_exp = pd.array(expected, dtype="boolean")
    pd.testing.assert_extension_array_equal(cmp, arr_exp, check_dtype=False)


def test_shape(storage: UuidStorage) -> None:
    arr = pd.array([uuid4(), uuid4()], dtype=UuidDtype(storage))
    assert arr.shape == (2,)


@pytest.mark.parametrize("length", [1, 2, 5, 13])
def test_nbytes(storage: UuidStorage, length: int) -> None:
    arr = pd.array([uuid4() for _ in range(length)], dtype=UuidDtype(storage))
    assert arr.nbytes == 16 * length


def test_copy(subtests: pytest.Subtests, storage: UuidStorage) -> None:
    data = [uuid4(), uuid4()]
    arr = pd.array(data, dtype=UuidDtype(storage))
    copy = arr.copy()

    with subtests.test("copies"):
        assert isinstance(copy, type(arr))
        assert copy.tolist() == data

    if isinstance(copy, UuidArray):
        with subtests.test("original array is not modified"):
            copy._ndarray[0] = uuid4().bytes  # noqa: SLF001
            assert copy.tolist() != data
            assert arr.tolist() == data


def test_concat(subtests: pytest.Subtests, storage: UuidStorage) -> None:
    dtype = UuidDtype(storage)
    batch_len = 4
    arrays = [
        pd.array([uuid4() for _ in range(batch_len)], dtype=dtype) for _ in range(4)
    ]
    concat = dtype.construct_array_type()._concat_same_type(arrays)  # noqa: SLF001
    for i, (expected, batch) in enumerate(
        zip(arrays, batched(concat, batch_len), strict=True)
    ):
        with subtests.test(i=i):
            assert list(batch) == expected.tolist()


def test_concat_empty(storage: UuidStorage) -> None:
    cls = UuidDtype(storage).construct_array_type()
    concat = cls._concat_same_type([])
    assert concat.tolist() == []


@pytest.mark.parametrize("container", [pd.array, pd.Index, pd.Series])
def test_repr(
    storage: UuidStorage,
    container: Callable[..., UuidArray | ArrowUuidArray | pd.Series | pd.Index],
) -> None:
    data = [uuid4(), uuid4()] if storage == "numpy" else [uuid4(), pd.NA]
    arr = container(data, dtype=UuidDtype(storage))
    match arr:
        case UuidArray() | ArrowUuidArray():
            name = type(arr).__name__
            expected = f"<{name}>\n[{data[0]}, {data[1]}]\nLength: 2, dtype: uuid"
        case pd.Index():
            expected = f"Index([{data[0]}, {data[1]}], dtype='uuid')"
        case pd.Series():
            expected = f"0    {data[0]}\n1    {data[1]!s:>36}\ndtype: uuid"
    assert repr(arr) == expected


@pytest.mark.parametrize("n", [1, 5, 1_000])
def test_random(storage: UuidStorage, n: int) -> None:
    cls = UuidDtype(storage).construct_array_type()
    arr = cls.random(n)
    assert isinstance(arr, cls)
    assert len(arr) == n
    assert arr.isna().sum() == 0
