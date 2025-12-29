# SPDX-License-Identifier: MPL-2.0
"""Test constructing an {Arrow,}Uuidrray in various ways."""

from __future__ import annotations

from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pandas_uuid import ArrowUuidArray, UuidArray, UuidDtype
from pandas_uuid._pyarrow import HAS_PYARROW

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import Dtype as PdDtype

    from pandas_uuid import UuidLike, UuidStorage


skipif_no_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow is not installed"
)


@pytest.fixture(scope="session", params=["pd.array", "type"])
def api(
    request: pytest.FixtureRequest, storage: UuidStorage
) -> Callable[..., UuidArray | ArrowUuidArray]:
    match request.param, storage:
        case "pd.array", _:
            return partial(pd.array, dtype=UuidDtype(storage))  # pyright: ignore[reportReturnType]
        case "type", "numpy":
            return UuidArray
        case "type", "pyarrow":
            return ArrowUuidArray
        case _:
            pytest.fail(f"unknown (api, storage): {request.param} {storage}")


def test_default_storage() -> None:
    dtype = UuidDtype()
    assert dtype.storage == ("pyarrow" if HAS_PYARROW else "numpy")
    assert dtype.na_value is pd.NA


def test_storage_error() -> None:
    with pytest.raises(ValueError, match=r"storage.*not.*python"):
        UuidDtype("python")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "arg", ["numpy", pytest.param("pyarrow", marks=skipif_no_pyarrow)]
)
def test_construct_array(
    request: pytest.FixtureRequest,
    api: Callable[..., UuidArray | ArrowUuidArray],
    arg: UuidStorage,
    storage: UuidStorage,
) -> None:
    if storage == "pyarrow" and arg == "numpy":
        request.applymarker(pytest.mark.xfail(raises=NotImplementedError))

    values = [uuid4(), uuid4()]
    match arg:
        case "numpy":
            store = np.array([v.bytes for v in values], dtype=np.void(16))
        case "pyarrow":
            from pyarrow import uuid

            store = pa.array([v.bytes for v in values], type=uuid())

    arr = api(store)
    assert arr.dtype == UuidDtype(storage)
    assert arr.tolist() == values


@pytest.mark.parametrize(
    ("storage", "mk_arr", "dtype", "exc_cls"),
    [
        pytest.param(
            "numpy",
            lambda: np.array([[uuid4().bytes] * 2] * 2, dtype=np.void(16)),
            None,
            ValueError,
            id="numpy-2d",
        ),
        pytest.param(
            "numpy",
            lambda: np.array([datetime.now()]),  # noqa: DTZ005
            None,
            TypeError,
            id="numpy-dtype",
        ),
        pytest.param(
            "pyarrow",
            lambda pa: pa.array([datetime.now()], type=pa.timestamp("s")),  # noqa: DTZ005
            None,
            ValueError,
            id="pyarrow-dtype",
        ),
        pytest.param(
            "numpy",
            lambda: np.array([], dtype="V16"),
            UuidDtype("pyarrow"),
            ValueError,
            id="numpy-dtype-arg",
        ),
    ],
)
def test_construct_array_error(
    storage: UuidStorage,
    mk_arr: Callable[..., np.ndarray | pa.Array],
    dtype: PdDtype | None,
    exc_cls: type[Exception] | tuple[type[Exception], ...],
) -> None:
    if storage == "pyarrow":
        if not HAS_PYARROW:
            pytest.skip("pyarrow is not installed")
        import pyarrow as pa

        mk_arr = partial(mk_arr, pa)

    api = dict(numpy=UuidArray, pyarrow=ArrowUuidArray)[storage]
    with pytest.raises(exc_cls):
        api(mk_arr(), dtype=dtype)


def test_simple_new_error() -> None:
    with pytest.raises(ValueError, match=r"support.*UuidDtype"):
        UuidArray._simple_new(np.ndarray([]), dtype=object)  # pyright: ignore[reportArgumentType]  # noqa: SLF001


def test_from_backing_data_error() -> None:
    with pytest.raises(ValueError, match=r"values.dtype.*V16"):
        UuidArray([])._from_backing_data(np.ndarray([], dtype=object))  # pyright: ignore[reportArgumentType]  # noqa: SLF001


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
        pytest.param(uuid4().bytes, id="bytes"),
        pytest.param(5, id="int"),
        pytest.param(str(uuid4()), id="str"),
        pytest.param(pa_uuid, marks=skipif_no_pyarrow, id="pyarrow"),
    ],
)
def test_construct_elem(
    api: Callable[..., UuidArray | ArrowUuidArray],
    storage: UuidStorage,
    value: UuidLike,
) -> None:
    arr = api([value])
    assert arr.dtype == UuidDtype(storage)


def test_construct_elem_error(api: Callable[..., UuidArray | ArrowUuidArray]) -> None:
    with pytest.raises(TypeError):
        api([()])  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("api", [UuidArray, ArrowUuidArray])
def test_construct_dtype_error(api: type[UuidArray | ArrowUuidArray]) -> None:
    with pytest.raises(ValueError, match=r"support.*UuidDtype"):
        api([], dtype=object)  # pyright: ignore[reportArgumentType]
