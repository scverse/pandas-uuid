# SPDX-License-Identifier: MPL-2.0
"""Stub for pyarrow classes for instance checks if pyarrow is not installed."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

__all__ = [
    "Array",
    "ChunkedArray",
    "DataType",
    "ExtensionArray",
    "Scalar",
    "UuidArray",
    "UuidScalar",
    "UuidType",
]


if TYPE_CHECKING or find_spec("pyarrow"):
    from pyarrow import (
        Array,
        ChunkedArray,
        DataType,
        ExtensionArray,
        Scalar,
        UuidArray,
        UuidScalar,
        UuidType,
    )

    HAS_PYARROW = True
else:  # pragma: no cover
    Array = type("Array", (), dict(__module__="pyarrow"))
    ChunkedArray = type("ChunkedArray", (), dict(__module__="pyarrow"))
    DataType = type("DataType", (), dict(__module__="pyarrow"))
    ExtensionArray = type("ExtensionArray", (), dict(__module__="pyarrow"))
    Scalar = type("Scalar", (), dict(__module__="pyarrow"))
    UuidArray = type("UuidArray", (), dict(__module__="pyarrow"))
    UuidScalar = type("UuidScalar", (), dict(__module__="pyarrow"))
    UuidType = type("UuidType", (), dict(__module__="pyarrow"))

    HAS_PYARROW = False
