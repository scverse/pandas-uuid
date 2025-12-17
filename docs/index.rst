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
>>> arr
<pyarrow.lib.UuidArray object at 0x1169ccd40>
[
  E8F04C2EED42488E9E96FE6C80D06BF6
]

And back manually.
:class:`pyarrow.UuidType` does not know about this package,
so we need to specificallye use our :class:`~pandas_uuid.UuidDtype`:

>>> pd.Series(arr, dtype=UuidDtype())
0    e8f04c2e-ed42-488e-9e96-fe6c80d06bf6
dtype: uuid

or (this would make more sense with a :class:`pyarrow.Table`):

>>> arr.to_pandas(types_mapper={pa.uuid(): UuidDtype()}.get)
0    e8f04c2e-ed42-488e-9e96-fe6c80d06bf6
dtype: uuid
