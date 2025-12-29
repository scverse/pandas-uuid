|coverage| |docs| |tests|

.. |coverage| image:: https://codecov.io/github/scverse/pandas-uuid/graph/badge.svg?token=R9HFDFPBID
    :target: https://codecov.io/github/scverse/pandas-uuid
.. |docs| image:: https://app.readthedocs.com/projects/icb-pandas-uuid/badge/
    :target: https://icb-pandas-uuid.readthedocs-hosted.com/
.. |tests| image:: https://github.com/scverse/pandas-uuid/actions/workflows/test.yml/badge.svg
    :target: https://github.com/scverse/pandas-uuid/actions/workflows/test.yml

.. badges-end

Pandas ExtensionArray / ExtensionDType for UUID
===============================================

.. usage-start

Use `UuidDtype` as a `pandas` extension dtype:

>>> from uuid import uuid4
>>> import pandas as pd
>>> from pandas_uuid import UuidDtype
>>>
>>> s = pd.Series([uuid4()], dtype=UuidDtype())
>>> s
0    cd072cd8-be6f-4f62-ac4c-09c28206e7e3
dtype: uuid

Use specific storage types by importing `UuidArray` / `ArrowUuidArray`,
or by using the ``storage`` parameter of `UuidDtype`:

>>> cls = UuidDtype("numpy").construct_array_type()
>>> cls
<class 'pandas_uuid.UuidArray'>
>>> cls.random(2, rng=42)
<UuidArray>
[8826d916-cdfb-21c6-c1ff-91a761565a70, 2416da6e-c212-cddb-8d88-00160eb686b2]
Length: 2, dtype: uuid

..  note::
    There is probably no good reason to ever set ``rng``
    to a static seed apart from testing.

.. usage-end

For advanced usage, see the Documentation_.

.. _documentation: https://icb-pandas-uuid.readthedocs-hosted.com/en/latest/#usage
