# SPDX-License-Identifier: MPL-2.0
"""Test non-construction APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pandas_uuid import UuidDtype

if TYPE_CHECKING:
    from uuid import UUID

    from pandas._typing import ScalarIndexer, SequenceIndexer

    from pandas_uuid import UuidStorage


def test_isna(request: pytest.FixtureRequest, storage: UuidStorage) -> None:
    if storage == "numpy":
        request.applymarker(pytest.mark.xfail(raises=TypeError))
    arr = pd.array([uuid4(), uuid4(), None], dtype=UuidDtype(storage))
    assert arr.isna().tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("values", "index", "expected"),
    [
        pytest.param([u0 := uuid4()], 0, u0, id="only-py"),
        pytest.param([u0 := uuid4()], np.int64(0), u0, id="only-np"),
        pytest.param([None], 0, None, id="only-na"),
        pytest.param([u0, u1 := uuid4()], 1, u1, id="second"),
        pytest.param([u0, u1], slice(None), [u0, u1], id="all-slice"),
        pytest.param([None, u1], slice(None), [None, u1], id="all-slice-na"),
        pytest.param([u0, u1], slice(None, None, -1), [u1, u0], id="inv-slice"),
        pytest.param([u0, u1], [0, 1], [u0, u1], id="all-list"),
        pytest.param([u0, u1], [1, 0], [u1, u0], id="all-list-reorder"),
    ],
)
def test_getitem(
    request: pytest.FixtureRequest,
    storage: UuidStorage,
    values: list[UUID | None],
    index: ScalarIndexer | SequenceIndexer,
    expected: UUID | list[UUID],
) -> None:
    if None in values and storage == "numpy":
        request.applymarker(pytest.mark.xfail(raises=TypeError))
    arr = pd.array(values, dtype=UuidDtype(storage))
    match expected:
        case list():
            assert arr[index].tolist() == expected
        case _:
            assert arr[index] == expected
