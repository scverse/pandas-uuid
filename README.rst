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
Usage
-----

Use `UuidDtype` as a `pandas` extension dtype:

>>> from uuid import uuid4
>>> import pandas as pd
>>> from pandas_uuid import UuidDtype
>>>
>>> pd.Series([uuid4()], dtype=UuidDtype())
0    7c1cde80-78ba-4f02-9565-adea3f9d6788
dtype: uuid

.. usage-end

For advanced usage, see the Examples_ notebook.

.. _Examples: https://icb-pandas-uuid.readthedocs-hosted.com/en/latest/notebooks/example.html
