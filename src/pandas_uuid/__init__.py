# SPDX-License-Identifier: MPL-2.0
"""Pandas ExtensionArray / ExtensionDType for UUIDs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, ClassVar, Literal, cast, get_args, overload, override
from uuid import UUID

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.api.indexers import check_array_indexer
from pandas.core.algorithms import take
from pandas.core.ops.common import unpack_zerodim_and_defer

from . import _pyarrow as pa

if TYPE_CHECKING:
    import builtins
    from collections.abc import Iterable
    from typing import Self

    import numpy.typing as npt

    npt._ArrayLikeInt_co = None  # type: ignore  # noqa: PGH003, SLF001

    from pandas._libs.missing import NAType
    from pandas._typing import ScalarIndexer, SequenceIndexer, TakeIndexer
    from pandas.core.arrays import BooleanArray


__all__ = ["UuidDtype", "UuidExtensionArray", "UuidLike", "UuidStorage"]


type UuidStorage = Literal["numpy", "pyarrow"]
"""Supported storage backend for :class:`~pandas_uuid.UuidDtype`."""
type UuidLike = UUID | pa.UuidScalar | bytes | int | str
"""Supported element types when creating a :class:`~pandas_uuid.UuidExtensionArray` \
from a sequence.
"""
type _UuidStorageArray = NDArray[np.void] | pa.UuidArray


# 16 void bytes: 128 bit, every pattern valid, no funky behavior like 0 stripping.
_UUID_NP_STORAGE_DTYPE: np.dtype[np.void] = np.dtype("V16")


@cache
def default_storage_kind() -> UuidStorage:
    if find_spec("pyarrow"):
        return "pyarrow"
    return "numpy"


def _to_uuid_numpy(v: UuidLike) -> UUID:
    match v:
        case UUID():
            return v
        case pa.UuidScalar():
            return v.as_py()
        case bytes():
            return UUID(bytes=v)
        case int():
            return UUID(int=v)
        case str():
            return UUID(v)
    msg = f"Unknown type for Uuid: {type(v)} is not {get_args(UuidLike.__value__)}"
    raise TypeError(msg)


def _to_uuid_pyarrow(v: UuidLike) -> pa.UuidScalar:
    from pyarrow import scalar, uuid

    match v:
        case pa.UuidScalar():
            return v
        case UUID():
            return scalar(v.bytes, type=uuid())
        case bytes():
            # raises a pa.ArrowInvalid error if not 16 bytes
            return scalar(v, type=uuid())
        case int():
            # raises an OverflowError if v has >128 bits
            return scalar(v.to_bytes(16), type=uuid())
        case str():
            return _to_uuid_pyarrow(UUID(v))
    msg = f"Unknown type for Uuid: {type(v)} is not {get_args(UuidLike.__value__)}"
    raise TypeError(msg)


@dataclass(frozen=True)
class UuidDtype(ExtensionDtype):
    """Pandas extension dtype for UUIDs."""

    # Custom

    storage: UuidStorage = field(default_factory=default_storage_kind)
    """Storage kind, either `"numpy"` or `"pyarrow"`."""

    # ExtensionDtype essential API (3 class attrs and methods)

    name: ClassVar[str] = "uuid"
    type: ClassVar[builtins.type[UUID]] = UUID

    @classmethod
    @override
    def construct_array_type(cls) -> type[UuidExtensionArray]:
        return UuidExtensionArray

    # ExtensionDtype overrides

    @property
    @override
    def kind(self) -> Literal["O", "V"]:
        """Return the dtype’s kind.

        Should be `"V"`, but `"O"` is used because of `pandas-dev/pandas#54810`_.

        .. _pandas-dev/pandas#54810: https://github.com/pandas-dev/pandas/issues/54810
        """
        return "O"

    @property
    @override
    def na_value(self) -> NAType:
        """Returns :attr:`pandas.NA`, i.e. this dtype has missing value semantics."""
        return pd.NA

    # IO

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> UuidExtensionArray:
        """PyArrow extension API for :meth:`pyarrow.Array.from_pandas`.

        See :ref:`pyarrow-integration` for an example.
        """
        return UuidExtensionArray(array)


class UuidExtensionArray(ExtensionArray):
    """Pandas extension array for UUIDs."""

    # Implementation details and convenience

    _data: _UuidStorageArray

    def __init__(
        self,
        values: Iterable[UuidLike | NAType | None],
        *,
        copy: bool = False,
        dtype: UuidDtype | None = None,
    ) -> None:
        """Initialize the array from an iterable of UUIDs.

        Constructing from a :type:`UuidStorage` is fast.
        """
        if isinstance(values, np.ndarray):
            if dtype is not None and dtype.storage != "numpy":
                raise NotImplementedError
            self._data = values.astype(_UUID_NP_STORAGE_DTYPE, copy=copy)
        elif isinstance(values, pa.Array):
            if dtype is not None and dtype.storage != "pyarrow":
                raise NotImplementedError
            from pyarrow import uuid

            self._data = values.cast(uuid())
        elif (dtype is None and find_spec("pyarrow")) or (
            dtype is not None and dtype.storage == "pyarrow"
        ):
            # TODO: make this and the next branch more efficient
            # https://github.com/scverse/pandas-uuid/issues/2
            from pyarrow import array, binary, uuid

            # cast because of https://github.com/apache/arrow/issues/48470
            self._data = array(
                [
                    None if pd.isna(x) else _to_uuid_pyarrow(x).cast(binary(16))
                    for x in values
                ],
                type=uuid(),
            )
        else:
            self._data = np.array(
                [_to_uuid_numpy(x).bytes for x in values],
                dtype=_UUID_NP_STORAGE_DTYPE,
            )

        if getattr(self._data, "ndim", 1) != 1:
            msg = "Array only supports 1-d arrays"
            raise ValueError(msg)

    @override
    def __hash__(self) -> int:
        return hash(self._data)

    # ExtensionArray essential API (11 class attrs and methods)

    @property
    @override
    def dtype(self) -> UuidDtype:
        match self._data:
            case pa.Array():
                return UuidDtype(storage="pyarrow")
            case np.ndarray():
                return UuidDtype(storage="numpy")
            case _:  # pragma: no cover
                msg = f"Unknown storage type: {type(self._data)}"
                raise AssertionError(msg)

    @override
    @classmethod
    def _from_sequence(  # pyright: ignore[reportGeneralTypeIssues]
        cls,
        scalars: Iterable[UuidLike],
        *,
        dtype: UuidDtype | None = None,
        copy: bool = False,
    ) -> Self:
        if dtype is None:
            dtype = UuidDtype()

        if not isinstance(dtype, UuidDtype):
            msg = f"{cls.__name__!r} only supports `UuidDtype` dtype"
            raise TypeError(msg)
        return cls(scalars, copy=copy)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> UUID: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    @override
    def __getitem__(self, item: ScalarIndexer | SequenceIndexer) -> Self | UUID:
        if isinstance(item, int | np.integer):
            match self._data[item]:
                case pa.UuidScalar() as elem:
                    return elem.as_py()
                case np.bytes_() as elem:
                    return UUID(bytes=elem.tobytes())
                case elem:
                    msg = f"Unknown type for Uuid: {type(elem)}"
                    raise AssertionError(msg)
        item = check_array_indexer(self, item)
        if (
            isinstance(item, np.ndarray)
            and item.dtype.kind == "b"
            and isinstance(self._data, pa.Array | pa.ChunkedArray)
        ):
            return self._simple_new(self._data.filter(item))
        return self._simple_new(self._data[item])

    # def __setitem__(self, index, value):

    @override
    def __len__(self) -> int:
        return len(self._data)

    @unpack_zerodim_and_defer("__eq__")
    @override
    def __eq__(self, other: object) -> BooleanArray:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._cmp("eq", other)

    @property
    @override
    def nbytes(self) -> int:
        return self._data.nbytes

    @override
    def isna(self) -> NDArray[np.bool_]:
        return pd.isna(self._data)

    @override
    def take(
        self,
        indexer: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: UUID | NAType | None = None,
    ) -> Self:
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(self._data, indexer, allow_fill=allow_fill, fill_value=fill_value)
        return self._simple_new(result)

    @override
    def copy(self) -> Self:
        return self._simple_new(
            self._data.copy() if isinstance(self._data, np.ndarray) else self._data
        )

    @override
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:  # pyright: ignore[reportGeneralTypeIssues]
        return cls._simple_new(np.concatenate([x._data for x in to_concat]))  # noqa: SLF001

    # Other overrides

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (len(self),)

    # Helpers

    @classmethod
    def _simple_new(cls, values: _UuidStorageArray) -> Self:
        result = UuidExtensionArray.__new__(cls)
        result._data = values  # noqa: SLF001
        return result

    def _cmp(self, op: str, other: object) -> BooleanArray:
        if isinstance(other, UuidExtensionArray):
            other = other._data  # noqa: SLF001
        elif isinstance(other, Sequence):
            other = np.asarray(other)
            if other.ndim > 1:
                msg = "can only perform ops with 1-d structures"
                raise NotImplementedError(msg)
            if len(self) != len(other):
                msg = "Lengths must match to compare"
                raise ValueError(msg)

        method = getattr(self._data, f"__{op}__")
        result = method(other)

        # TODO: deal with `result` being NotImplemented
        # https://github.com/scverse/pandas-uuid/issues/1

        return cast("BooleanArray", pd.array(result, dtype="boolean"))

    # IO

    def __arrow_array__(
        self,
        type: pa.DataType | None = None,  # noqa: A002
    ) -> pa.Array | pa.ChunkedArray:
        """Convert the underlying array values to a pyarrow Array.

        See :ref:`pyarrow-integration` for an example
        and :ref:`pyarrow:arrow_array_protocol` for details.
        """
        import pyarrow as pa

        if type is None:
            type = pa.uuid()  # noqa: A001

        if isinstance(self._data, pa.Array | pa.ChunkedArray):
            return self._data.cast(type)
        return pa.array(self._data, type=type)
