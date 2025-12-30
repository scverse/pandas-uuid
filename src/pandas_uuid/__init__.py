# SPDX-License-Identifier: MPL-2.0
"""Pandas ExtensionArray / ExtensionDType for UUIDs."""

from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from functools import cache, cached_property
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, TypeVar, cast, get_args, overload, override
from uuid import UUID

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.api.indexers import check_array_indexer
from pandas.arrays import ArrowExtensionArray, NumpyExtensionArray

from . import _pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from types import FunctionType
    from typing import Self

    import numpy.typing as npt
    from numpy.random import Generator
    from numpy.typing import NDArray

    npt._ArrayLikeInt_co = None  # type: ignore  # noqa: PGH003, SLF001

    from pandas._libs.missing import NAType
    from pandas._typing import ScalarIndexer, SequenceIndexer
    from pandas.arrays import BooleanArray


__all__ = [
    "ArrowUuidArray",
    "BaseUuidArray",
    "UuidArray",
    "UuidDtype",
    "UuidLike",
    "UuidStorage",
]


type UuidStorage = Literal["numpy", "pyarrow"]
"""Supported storage backend for :class:`~pandas_uuid.UuidDtype`."""
type UuidLike = UUID | pa.UuidScalar | bytes | int | str
"""Supported element types when creating a :class:`~pandas_uuid.BaseUuidArray` \
from a sequence.
"""

if TYPE_CHECKING or sys.version_info >= (3, 13):
    _DT = TypeVar("_DT", bound="pa.DataType", default=pa.UuidType)
else:  # pragma: no cover
    _DT = TypeVar("_DT", bound="pa.DataType")

# 16 void bytes: 128 bit, every pattern valid, no funky behavior like 0 stripping.
_UUID_NP_STORAGE_DTYPE: np.dtype[np.void] = np.dtype("V16")


@cache
def _default_storage_kind() -> UuidStorage:
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
    msg = f"Unknown type for Uuid: {type(v)} is not in {get_args(UuidLike.__value__)}"
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

    storage: UuidStorage = field(default_factory=_default_storage_kind)
    """Storage kind, either ``"numpy"`` or ``"pyarrow"``."""

    def __post_init__(self) -> None:
        """Validate storage kind."""
        if self.storage not in {"numpy", "pyarrow"}:
            msg = f"storage must be 'numpy' or 'pyarrow', not {self.storage}"
            raise ValueError(msg)

    # ExtensionDtype essential API (3 class attrs and methods)

    @cached_property
    @override
    def name(self) -> Literal["uuid"]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return "uuid"

    @cached_property
    @override
    def type(self) -> type[UUID]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return UUID

    @override
    def construct_array_type(self) -> type[UuidArray | ArrowUuidArray]:
        return UuidArray if self.storage == "numpy" else ArrowUuidArray

    # ExtensionDtype overrides

    index_class = pd.Index  # pandas accesses this from the class

    @cached_property
    @override
    def kind(self) -> Literal["O", "V"]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the dtype’s kind.

        Should be `"V"`, but `"O"` is used because of `pandas-dev/pandas#54810`_.

        .. _pandas-dev/pandas#54810: https://github.com/pandas-dev/pandas/issues/54810
        """
        return "O"

    @cached_property
    @override
    def na_value(self) -> NAType:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Returns :attr:`pandas.NA`, i.e. this dtype has missing value semantics."""
        return pd.NA

    # IO

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> ArrowExtensionArray:
        """PyArrow extension API for :meth:`pyarrow.Array.to_pandas`.

        Incomplete because :meth:`pyarrow.UuidType.to_pandas_dtype`
        does not refer to this (see :ref:`pyarrow:conversion-to-pandas`).

        See :ref:`pyarrow-integration` for an example.
        """
        return ArrowUuidArray(array)


class BaseUuidArray(ExtensionArray, abc.ABC):
    """Base class for :class:`~pandas_uuid.UuidArray` and :class:`~pandas_uuid.ArrowUuidArray`."""  # noqa: E501

    # Non-essential overrides

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (len(self),)

    @staticmethod
    @cache
    def _formatter(*, boxed: bool = False) -> Callable[[UuidLike], str]:
        del boxed  # part of the API
        # pandas converts to numpy object array before calling this,
        # so we need to convert it back.
        return lambda b: str(b if pd.isna(b) else _to_uuid_numpy(b))

    # Custom API

    @classmethod
    @abc.abstractmethod
    def random(cls, size: int, *, rng: int | np.random.Generator | None = None) -> Self:
        """Generate an array of random UUIDs."""


class UuidArray(BaseUuidArray, NumpyExtensionArray):
    """Extension array for storing uuid data in a :class:`numpy.ndarray`."""

    # Implementation details and convenience

    _typ = "extension"  # undo numpy extension array hack
    _ndarray: NDArray[np.void]

    def __init__(
        self,
        values: Iterable[UuidLike],
        *,
        copy: bool = False,
        dtype: UuidDtype | None = None,
    ) -> None:
        """Initialize the array from an iterable of UUIDs.

        Constructing from a :type:`UuidStorage` is fast.
        """
        if not isinstance(dtype, UuidDtype | None) or (
            dtype is not None and dtype.storage != "numpy"
        ):
            msg = (
                f"{type(self).__name__!r} only supports `UuidDtype(storage='numpy')`, "
                f"not {dtype}"
            )
            raise ValueError(msg)

        # TODO: implement conversion between storage kinds
        # https://github.com/scverse/pandas-uuid/issues/12

        # we treat object arrays as sequences (we can’t efficiently convert)
        if isinstance(values, np.ndarray) and values.dtype.kind != "O":
            values = values.astype(_UUID_NP_STORAGE_DTYPE, copy=copy)
        else:
            # TODO: make construction from elements more efficient
            #       (both numpy and pyarrow)
            # https://github.com/scverse/pandas-uuid/issues/2
            values = np.array(
                [_to_uuid_numpy(x).bytes for x in values],
                dtype=_UUID_NP_STORAGE_DTYPE,
            )

        if values.ndim != 1:
            msg = "Array only supports 1-dimensional arrays"
            raise ValueError(msg)

        super().__init__(values)

    # ExtensionArray essential API (11 class attrs and methods)

    @cached_property
    @override
    def dtype(self) -> UuidDtype:  # pyright: ignore[reportIncompatibleMethodOverride]
        return UuidDtype(storage="numpy")

    @override
    @classmethod
    def _from_sequence(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        scalars: Iterable[UuidLike],
        *,
        dtype: UuidDtype | None = None,
        copy: bool = False,
    ) -> Self:
        return cls(scalars, copy=copy, dtype=dtype)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> UUID: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    @override
    def __getitem__(self, item: ScalarIndexer | SequenceIndexer) -> Self | UUID:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(item, int | np.integer):
            elem = cast("np.void", self._ndarray[item])
            return UUID(bytes=elem.tobytes())
        item = check_array_indexer(self, item)
        return self._simple_new(self._ndarray[item])

    # TODO: def __setitem__(self, index, value)
    # https://github.com/scverse/pandas-uuid/issues/15

    # Some methods are implemented by NumpyExtensionArray:
    # __len__, __eq__, nbytes, take

    @override
    def isna(self) -> NDArray[np.bool_]:
        return np.zeros(len(self), dtype=bool)

    @override
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        if len(to_concat) == 0:
            return cls([])
        values = np.concatenate([x._ndarray for x in to_concat])  # noqa: SLF001
        return cls._simple_new(values)

    # Helpers

    @override
    @classmethod
    def _simple_new(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, values: NDArray[np.void], dtype: UuidDtype | None = None
    ) -> Self:
        if dtype is None:
            dtype = UuidDtype(storage="numpy")
        elif not isinstance(dtype, UuidDtype) or dtype.storage != "numpy":
            msg = (
                f"{type(cls).__name__!r} only supports `UuidDtype(storage='numpy')`, "
                f"not {dtype}"
            )
            raise ValueError(msg)
        return super()._simple_new(values, dtype=dtype)

    def _from_backing_data(self, values: NDArray[np.void]) -> Self:
        if values.dtype != _UUID_NP_STORAGE_DTYPE:
            name = type(self).__name__
            msg = (
                f"{name!r} only supports `values.dtype=='{_UUID_NP_STORAGE_DTYPE}'`, "
                f"not {values.dtype}"
            )
            raise ValueError(msg)
        return super()._from_backing_data(values)

    @override
    def _cmp_method(
        self, other: Sequence[UuidLike] | UuidArray, op: FunctionType
    ) -> BooleanArray:
        if isinstance(other, ArrowUuidArray):
            return NotImplemented  # delegate to it to support NAs
        if not isinstance(other, UuidArray):
            other = cast("UuidArray", pd.array(other, dtype=self.dtype))  # pyright: ignore[reportAssignmentType]

        method = getattr(self._ndarray, f"__{op.__name__}__")
        result = method(other._ndarray.view(np.void(16)))  # noqa: SLF001
        return cast("BooleanArray", pd.array(result, dtype="boolean"))

    # IO

    @overload
    def __arrow_array__(self, type: pa.UuidType | None = None) -> pa.UuidArray: ...
    @overload
    def __arrow_array__(self, type: _DT) -> pa.Array[pa.Scalar[_DT]]: ...
    def __arrow_array__(  # pyright: ignore[reportInconsistentOverload]
        self,
        type: _DT | None = None,  # noqa: A002
    ) -> pa.Array[pa.Scalar[_DT]] | pa.ChunkedArray[pa.Scalar[_DT]]:
        """PyArrow extension API for :meth:`pyarrow.Array.from_pandas`.

        See :ref:`pyarrow-integration` for an example
        and :ref:`pyarrow:arrow_array_protocol` for details.
        """
        import pyarrow as pa

        if type is None:
            type = pa.uuid()  # pyright: ignore[reportAssignmentType] # noqa: A001

        return pa.array(self._ndarray, type=type)  # pyright: ignore[reportReturnType]

    # Custom API

    @override
    @classmethod
    def random(cls, size: int, *, rng: int | Generator | None = None) -> Self:
        rng = np.random.default_rng(rng)
        values = rng.bytes(size * 16)
        return cls._simple_new(np.frombuffer(values, dtype=_UUID_NP_STORAGE_DTYPE))


class ArrowUuidArray(BaseUuidArray, ArrowExtensionArray):
    """Extension array for storing uuid data in a :class:`pyarrow.ChunkedArray`."""

    _pa_array: pa.ChunkedArray[pa.UuidScalar]

    def __init__(
        self,
        values: Iterable[UuidLike | NAType | None]
        | pa.UuidArray
        | pa.ChunkedArray[pa.UuidScalar],
        *,
        dtype: UuidDtype | None = None,
    ) -> None:
        """Initialize the array from an iterable of UUIDs.

        Constructing from a :type:`UuidStorage` is fast.
        """
        if not isinstance(dtype, UuidDtype | None) or (
            dtype is not None and dtype.storage != "pyarrow"
        ):
            msg = (
                f"{type(self).__name__!r} only supports `UuidDtype(storage='pyarrow')`, "  # noqa: E501
                f"not {dtype}"
            )
            raise ValueError(msg)

        import pyarrow as pa

        # TODO: implement conversion between storage kinds
        # https://github.com/scverse/pandas-uuid/issues/12
        if isinstance(values, np.ndarray) and values.dtype.kind != "O":
            raise NotImplementedError

        if isinstance(values, pa.Array | pa.ChunkedArray):
            self._pa_array = (
                pa.chunked_array([values.view(pa.uuid())])
                if isinstance(values, pa.Array)
                else values.cast(pa.uuid())
            )  # pyright: ignore[reportAttributeAccessIssue]
        else:
            # TODO: make construction from elements more efficient
            #       (both numpy and pyarrow)
            # https://github.com/scverse/pandas-uuid/issues/2
            # cast because of https://github.com/apache/arrow/issues/48470
            chunk = pa.array(
                [
                    None if pd.isna(x) else _to_uuid_pyarrow(x).cast(pa.binary(16))  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
                    for x in values
                ],
                type=pa.uuid(),
            )
            self._pa_array = pa.chunked_array([chunk])  # pyright: ignore[reportAttributeAccessIssue]

    @cached_property
    @override
    def _dtype(self) -> UuidDtype:  # pyright: ignore[reportIncompatibleVariableOverride]
        return UuidDtype(storage="pyarrow")

    # ExtensionArray essential API (11 class attrs and methods)

    dtype: UuidDtype  # pyright: ignore[reportIncompatibleMethodOverride]

    @override
    @classmethod
    def _from_sequence(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        scalars: Iterable[UuidLike],
        *,
        dtype: UuidDtype | None = None,
        copy: bool = False,
    ) -> Self:
        del copy  # part of the API, but underlying array is readonly
        return cls(scalars, dtype=dtype)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> UUID | NAType: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    @override
    def __getitem__(
        self, item: ScalarIndexer | SequenceIndexer
    ) -> Self | UUID | NAType:
        if isinstance(item, int | np.integer):
            elem = cast("pa.UuidScalar", self._pa_array[item])  # pyright: ignore[reportArgumentType, reportCallIssue]
            return elem.as_py() if elem.is_valid else self.dtype.na_value

        item = check_array_indexer(self, item)
        match item:
            case slice() if (
                item.step in {1, None}
                and isinstance(item.start, int | None)
                and isinstance(item.stop, int | None)
            ):
                r = range(len(self))[item]
                values = self._pa_array.slice(r.start, r.stop - r.start)
            case slice():
                item = np.array(range(len(self._pa_array))[item])
                values = self._pa_array.take(item)
            case np.ndarray() if item.dtype.kind == "b":
                values = self._pa_array.filter(item)
            case np.ndarray():
                values = self._pa_array.take(item)
            case _:  # pragma: no cover
                msg = f"Unexpected indexer type: {type(item)}"
                raise AssertionError(msg)

        return type(self)(values)

    # TODO: def __setitem__(self, index, value)
    # https://github.com/scverse/pandas-uuid/issues/15

    # Some APIs are defined in ArrowExtensionArray/OpsMixin:
    # __len__, __eq__, nbytes, isna, take, copy

    @override
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        if len(to_concat) == 0:
            return cls([])
        import pyarrow as pa

        # is there a pa.concat_arrays for chunked ones?
        values = pa.chunked_array(
            [chunk for x in to_concat for chunk in x._pa_array.chunks]  # noqa: SLF001
        )
        return cls(values)

    # Helpers

    def _cmp_method(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Sequence[UuidLike] | ArrowUuidArray, op: FunctionType
    ) -> BooleanArray:
        import pyarrow as pa
        from pandas.core.arrays.arrow.array import ARROW_CMP_FUNCS

        if not isinstance(other, ArrowUuidArray):
            other = ArrowUuidArray(other)

        result = ARROW_CMP_FUNCS[op.__name__](
            self._pa_array.cast(pa.binary(16)),
            other._pa_array.cast(pa.binary(16)),
        )
        return cast("BooleanArray", pd.array(result, dtype="boolean"))  # pyright: ignore[reportArgumentType]

    # Custom API

    @override
    @classmethod
    def random(cls, size: int, *, rng: int | Generator | None = None) -> Self:
        import pyarrow as pa

        # pyarrow’s random generator only does non-NaN floats, we want unbiased
        rng = np.random.default_rng(rng)
        values = rng.bytes(size * 16)
        buf_vals = pa.py_buffer(values)
        arr = pa.Array.from_buffers(pa.uuid(), size, [None, buf_vals])
        return cls(cast("pa.UuidArray", arr))
