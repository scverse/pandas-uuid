pandas-uuid
-----------

..  toctree::
    :hidden:
    :maxdepth: 1

    self
    api


API
---

..  autosummary::
    pandas_uuid.BaseUuidArray
    pandas_uuid.UuidArray
    pandas_uuid.ArrowUuidArray
    pandas_uuid.UuidDtype
    pandas_uuid.UuidStorage
    pandas_uuid.UuidLike

..  currentmodule:: pandas_uuid

.. _usage:

Usage
-----

.. include:: ../README.rst
   :start-after: .. usage-start
   :end-before: .. usage-end

.. _pyarrow-integration:

pyarrow integration
~~~~~~~~~~~~~~~~~~~

Convert from :mod:`pandas` :class:`~pandas.api.extensions.ExtensionArray`
to :class:`pyarrow.UuidArray` automatically …

>>> import pyarrow as pa
>>>
>>> arr = pa.array(s.array)
>>> arr  # doctest: +ELLIPSIS
<pyarrow.lib.UuidArray object at 0x...>
[
  CD072CD8BE6F4F62AC4C09C28206E7E3
]

And back manually.
:class:`pyarrow.UuidType` does not know about this package,
so we need to specifically use our :class:`~pandas_uuid.UuidDtype`:

>>> pd.Series(arr, dtype=UuidDtype())
0    cd072cd8-be6f-4f62-ac4c-09c28206e7e3
dtype: uuid

or (this would make more sense with a :class:`pyarrow.Table`):

>>> arr.to_pandas(types_mapper={pa.uuid(): UuidDtype()}.get)
0    cd072cd8-be6f-4f62-ac4c-09c28206e7e3
dtype: uuid

.. _pandas-integration:

pandas integration
~~~~~~~~~~~~~~~~~~

The ultimate goal for this package is to disappear and
the classes to move into the :mod:`pandas` package.
This would have several advantages:

#.  As mentioned before, passing a :class:`pyarrow.UuidArray`
    to any pandas API could make it automatically convert it
    to a :class:`~pandas_uuid.UuidArray`.
    Currently this results in a generic ``{Numpy,Arrow}ExtensionArray`` instead:

    >>> pd.Series(arr)
    0    b'\xcd\x07,\xd8\xbeoOb\xacL\t\xc2\x82\x06\xe7\...
    dtype: object

#.  Specifying ``dtype="uuid"`` would work the same as ``dtype=UuidDtype()``.
    Currently it fails:

    >>> pd.Series([uuid4()], dtype="uuid")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: data type 'uuid' not understood
