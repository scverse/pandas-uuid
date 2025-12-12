"""Test constructing a UuidExtensionArray in various ways."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from pandas_uuid import UuidDtype, UuidExtensionArray

if TYPE_CHECKING:
    from pandas_uuid import UuidLike, UuidStorageKind

skipif_no_pyarrow = pytest.mark.skipif(
    not find_spec("pyarrow"), reason="pyarrow is not installed"
)


@pytest.fixture(
    scope="session", params=["numpy", pytest.param("pyarrow", marks=skipif_no_pyarrow)]
)
def storage(request: pytest.FixtureRequest) -> UuidStorageKind:
    return request.param


try:
    import pyarrow as pa
except ImportError:
    pa_uuid = None
else:
    pa_uuid = pa.scalar(b"\x01" * 16, type=pa.uuid())


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(uuid4(), id="uuid"),
        pytest.param(pa_uuid, marks=skipif_no_pyarrow, id="pyarrow"),
        pytest.param(b"\x03" * 16, id="bytes"),
        pytest.param(5, id="int"),
        pytest.param("00010203-0405-0607-0809-0a0b0c0d0e0f", id="str"),
    ],
)
def test_construct(storage: UuidStorageKind, value: UuidLike) -> None:
    UuidExtensionArray([value], dtype=UuidDtype(storage))
