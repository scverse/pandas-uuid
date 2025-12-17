# SPDX-License-Identifier: MPL-2.0
"""Test constructing a UuidExtensionArray in various ways."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pandas_uuid import UuidDtype, UuidExtensionArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas_uuid import UuidLike, UuidStorage


HAS_PYARROW = find_spec("pyarrow")

skipif_no_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow is not installed"
)


@pytest.fixture(scope="session", params=[pd.array, UuidExtensionArray])
def api(request: pytest.FixtureRequest) -> Callable[..., UuidExtensionArray]:
    return request.param


@pytest.fixture(
    scope="session", params=["numpy", pytest.param("pyarrow", marks=skipif_no_pyarrow)]
)
def storage(request: pytest.FixtureRequest) -> UuidStorage:
    return request.param


def test_default_storage() -> None:
    dtype = UuidDtype()
    assert dtype.storage == ("pyarrow" if HAS_PYARROW else "numpy")
    assert dtype.na_value is pd.NA


@pytest.mark.parametrize("arg", ["numpy", "pyarrow"])
def test_construct_array(
    request: pytest.FixtureRequest,
    api: Callable[..., UuidExtensionArray],
    arg: UuidStorage,
    storage: UuidStorage,
) -> None:
    if arg != storage:
        request.applymarker(pytest.mark.xfail(raises=NotImplementedError))

    values = [uuid4(), uuid4()]
    match arg:
        case "numpy":
            store = np.array([v.bytes for v in values], dtype=np.void(16))
        case "pyarrow":
            from pyarrow import uuid

            store = pa.array([v.bytes for v in values], type=uuid())
        case _:
            pytest.fail(f"Unknown storage: {storage}")

    arr = api(store, dtype=UuidDtype(storage))
    assert arr.dtype == UuidDtype(storage)
    assert arr.tolist() == values


try:
    import pyarrow as pa
except ImportError:
    pa_uuid = None
else:
    pa_uuid = pa.scalar(uuid4().bytes, type=pa.uuid())


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(uuid4(), id="uuid"),
        pytest.param(pa_uuid, marks=skipif_no_pyarrow, id="pyarrow"),
        pytest.param(uuid4().bytes, id="bytes"),
        pytest.param(5, id="int"),
        pytest.param(str(uuid4()), id="str"),
    ],
)
def test_construct_elem(
    api: Callable[..., UuidExtensionArray], storage: UuidStorage, value: UuidLike
) -> None:
    arr = api([value], dtype=UuidDtype(storage))
    assert arr.dtype == UuidDtype(storage)


def test_construct_elem_error(storage: UuidStorage) -> None:
    with pytest.raises(TypeError):
        UuidExtensionArray([()], dtype=UuidDtype(storage))  # pyright: ignore[reportArgumentType]


def test_isna(request: pytest.FixtureRequest, storage: UuidStorage) -> None:
    if storage == "numpy":
        request.applymarker(pytest.mark.xfail(raises=TypeError))
    arr = UuidExtensionArray([uuid4(), uuid4(), None], dtype=UuidDtype(storage))
    assert arr.isna().tolist() == [False, False, True]
