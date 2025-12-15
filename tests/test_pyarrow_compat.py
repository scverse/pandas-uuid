# SPDX-License-Identifier: MPL-2.0
"""Test pyarrow integration."""

from __future__ import annotations

from importlib.util import find_spec
from uuid import uuid4

import pandas as pd
import pytest

from pandas_uuid import UuidDtype, UuidExtensionArray

pytestmark = pytest.mark.skipif(
    not find_spec("pyarrow"), reason="pyarrow not installed"
)


def test_pandas_to_pyarrow() -> None:
    """`pa.array(ext_array)` should call `ext_array.__arrow_array__()`."""
    import pyarrow as pa

    pd_arr = pd.array([uuid4(), uuid4()], dtype=UuidDtype("pyarrow"))
    assert isinstance(pd_arr, UuidExtensionArray)
    assert isinstance(pd_arr._data, pa.Array)  # noqa: SLF001

    pa_arr = pa.array(pd_arr)  # pyright: ignore[reportArgumentType, reportCallIssue]

    assert pa_arr.type == pa.uuid()
    assert [e.as_py() for e in pa_arr] == list(pd_arr)


def test_pyarrow_to_pandas() -> None:
    """`pd.array(arr, dtype=UuidDtype())` should call `dtype.__from_arrow__(arr)`."""
    import pyarrow as pa

    data = [b"\x01" * 16, b"\x02" * 16]

    pa_arr = pa.array(data, type=pa.uuid())

    pd_ser = pd.Series(pa_arr, dtype=UuidDtype())
    expected = pd.array(data, dtype=UuidDtype("pyarrow"))
    pd.testing.assert_extension_array_equal(pd_ser.array, expected)
