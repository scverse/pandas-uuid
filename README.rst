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

.. usage-end

For advanced usage, see the Documentation_.

.. _documentation: https://icb-pandas-uuid.readthedocs-hosted.com/en/latest/#usage
