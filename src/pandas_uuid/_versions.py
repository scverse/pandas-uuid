# SPDX-License-Identifier: MPL-2.0
"""UUID bit layout: storage, the version field, and generating random UUIDs.

The layout is defined by `RFC 9562`_, which specifies versions 1–8,
each interpreting the 128 bits differently.

.. _RFC 9562: https://datatracker.ietf.org/doc/html/rfc9562#section-4
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np

from .bit_acc import bits

if TYPE_CHECKING:
    from collections.abc import Buffer

    from numpy.random import Generator
    from numpy.typing import NDArray

    from . import UuidDtype
    from . import _pyarrow as pa


# 16 void bytes: 128 bit, every pattern valid, no funky behavior like 0 stripping.
NP_STORAGE_DTYPE: np.dtype[np.void] = np.dtype("V16")

VERSIONS = range(1, 9)
# Versions whose payload is (mostly) random, i.e. that `random_values` can generate.
RANDOM_VERSIONS = frozenset({4, 7})
# UUIDv7: 48 bit big-endian Unix timestamp in ms, then random bits.
TIME_ORDERED = 7

# Fields every version shares (https://datatracker.ietf.org/doc/html/rfc9562#section-4.1)
VERSION_BITS = slice(48, 52)
VARIANT_BITS = slice(64, 66)
VARIANT_RFC9562 = 0b10
# UUIDv7’s `unix_ts_ms` (https://datatracker.ietf.org/doc/html/rfc9562#section-5.7)
V7_TIMESTAMP_BITS = slice(0, 48)


def check_version(values: NDArray[np.void], version: int) -> None:
    """Raise if any of `values` isn’t of UUID version `version`."""
    found = bits(values)[VERSION_BITS]
    if not (mismatch := found != version).any():
        return
    i = int(mismatch.argmax())
    msg = (
        f"UUID at index {i} has version {found[i]}, "
        f"but dtype specifies version {version}"
    )
    raise ValueError(msg)


def arrow_to_void(array: pa.ChunkedArray[pa.UuidScalar]) -> NDArray[np.void]:
    """Copy an arrow UUID array’s non-null elements into 16-byte void records."""
    import pyarrow as pa

    # combine_chunks copies the whole array.
    # Only reached for dtypes that specify a version: the common path stays copy-free.
    combined = array.cast(pa.binary(16)).combine_chunks()
    if len(combined) == 0:
        return np.empty(0, dtype=NP_STORAGE_DTYPE)
    buf = cast("Buffer", combined.buffers()[-1])
    values = np.frombuffer(buf, dtype=NP_STORAGE_DTYPE)
    values = values[combined.offset : combined.offset + len(combined)]
    return values[combined.is_valid().to_numpy(zero_copy_only=False)]


def version_for(dtype: UuidDtype | None) -> int:
    """Version to generate: the one `dtype` specifies, else 4."""
    return 4 if dtype is None or dtype.version is None else dtype.version


def random_values(
    size: int, version: int, rng: int | Generator | None
) -> NDArray[np.void]:
    """Generate `size` random UUIDs of `version` as 16-byte void records."""
    if version not in RANDOM_VERSIONS:
        msg = (
            f"Can only generate random UUIDs of version "
            f"{' or '.join(map(str, sorted(RANDOM_VERSIONS)))}, not {version}"
        )
        raise NotImplementedError(msg)
    generator = np.random.default_rng(rng)
    # `integers().view()` and not `frombuffer(generator.bytes())`,
    # since the latter is read-only and `bits` needs to write.
    values = generator.integers(0, 2**32, size=size * 4, dtype=np.uint32).view(
        NP_STORAGE_DTYPE
    )
    if version == TIME_ORDERED:
        # one timestamp for the whole batch, not per element.
        # Uniqueness is unaffected (74 random bits remain),
        # but a batch isn’t internally time-ordered.
        # Per-element stamps need `bits` to take an array.
        bits(values)[V7_TIMESTAMP_BITS] = time.time_ns() // 1_000_000
    bits(values)[VERSION_BITS] = version
    bits(values)[VARIANT_BITS] = VARIANT_RFC9562
    return values
