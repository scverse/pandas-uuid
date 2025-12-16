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
    pandas_uuid.UuidExtensionArray
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

And back manually. There is no registry,
so neither :class:`pyarrow.UuidArray` nor :mod:`pandas` knows,
only our :class:`~pandas_uuid.UuidDtype`:

>>> pd.Series(arr, dtype=UuidDtype())
0    e8f04c2e-ed42-488e-9e96-fe6c80d06bf6
dtype: uuid
