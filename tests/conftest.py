# SPDX-License-Identifier: MPL-2.0
"""Configuration for pytest."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Final

    from pandas_uuid import UuidStorage


HAS_PYARROW: Final = bool(find_spec("pyarrow"))

skipif_no_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow is not installed"
)


@pytest.fixture(scope="session")
def has_pyarrow() -> bool:
    return HAS_PYARROW


@pytest.fixture(
    scope="session", params=["numpy", pytest.param("pyarrow", marks=skipif_no_pyarrow)]
)
def storage(request: pytest.FixtureRequest) -> UuidStorage:
    return request.param
