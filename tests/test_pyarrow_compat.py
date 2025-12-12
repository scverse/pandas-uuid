# SPDX-License-Identifier: MPL-2.0
"""Test pyarrow compatibility."""

from __future__ import annotations

from importlib.util import find_spec
from uuid import uuid4

import pandas as pd
import pytest

from pandas_uuid import UuidDtype, UuidExtensionArray

pytestmark = pytest.mark.skipif(
    not find_spec("pyarrow"), reason="pyarrow not installed"
)


def test_roundtrip() -> None:
    import pyarrow as pa

    pd_arr = pd.array([uuid4()], dtype=UuidDtype())
    assert isinstance(pd_arr, UuidExtensionArray)

    # without `type=`, this would convert to a generic pandas ArrowExtensionArray
    pa_arr = pa.Array.from_pandas(pd_arr, type=pa.uuid())
    assert pa_arr.type == pa.uuid()

    pd_ser = pa_arr.to_pandas()
    assert pd.testing.assert_extension_array_equal(pd_arr, pd_ser.array)
