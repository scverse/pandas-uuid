# SPDX-License-Identifier: MPL-2.0
"""Test pyarrow integration."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd
import pytest

from pandas_uuid import UuidDtype

if TYPE_CHECKING:
    from typing import Literal

    from pandas_uuid import UuidStorage


if TYPE_CHECKING or find_spec("pyarrow"):
    import pyarrow as pa


pytestmark = pytest.mark.skipif(
    not find_spec("pyarrow"), reason="pyarrow not installed"
)


@pytest.mark.parametrize("api", ["array", "from_pandas"])
def test_pandas_to_pyarrow(
    storage: UuidStorage, api: Literal["array", "from_pandas"]
) -> None:
    """`pa.array(ext_array)` should call `ext_array.__arrow_array__()`."""
    pd_arr = pd.array([uuid4(), uuid4()], dtype=UuidDtype(storage))

    to_pa = dict(array=pa.array, from_pandas=pa.Array.from_pandas)[api]
    pa_arr = to_pa(pd_arr)  # pyright: ignore[reportArgumentType, reportCallIssue]

    assert pa_arr.type == pa.uuid()
    assert [e.as_py() for e in pa_arr] == list(pd_arr)


@pytest.mark.parametrize("api", ["pyarrow", "pandas"])
def test_pyarrow_to_pandas(api: Literal["pyarrow", "pandas"]) -> None:
    """`pd.array(arr, dtype=UuidDtype())` should call `dtype.__from_arrow__(arr)`."""
    data = [b"\x01" * 16, b"\x02" * 16]
    pa_arr = pa.array(data, type=pa.uuid())

    match api:
        case "pandas":
            pd_ser = pd.Series(pa_arr, dtype=UuidDtype())  # pyright: ignore[reportArgumentType, reportCallIssue]
        case "pyarrow":
            mapper = {pa.uuid(): UuidDtype()}.get
            pd_ser = pa_arr.to_pandas(types_mapper=mapper)  # pyright: ignore[reportArgumentType]

    expected = pd.array(data, dtype=UuidDtype("pyarrow"))
    pd.testing.assert_extension_array_equal(pd_ser.array, expected)
