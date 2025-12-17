# SPDX-License-Identifier: MPL-2.0
"""Configuration for pytest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from pandas_uuid._pyarrow import HAS_PYARROW

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pandas_uuid import UuidStorage


skipif_no_pyarrow = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow is not installed"
)


@pytest.fixture(
    scope="session", params=["numpy", pytest.param("pyarrow", marks=skipif_no_pyarrow)]
)
def storage(request: pytest.FixtureRequest) -> UuidStorage:
    return request.param


@pytest.fixture
def xfail_if_numpy_and_na(
    request: pytest.FixtureRequest, storage: UuidStorage
) -> Callable[..., None]:
    """Xfail if `storage=="numpy"` and any of the sequences contain missing values."""

    def xf(*seqs: list[Any]) -> None:
        if storage == "numpy" and (not seqs or any(pd.isna(seq).any() for seq in seqs)):
            request.applymarker(pytest.mark.xfail(raises=TypeError))

    return xf
