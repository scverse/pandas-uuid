Examples
========

..  code:: ipython3

    from uuid import uuid4

    import pandas as pd

    from pandas_uuid import UuidDtype, UuidExtensionArray

..  code:: ipython3

    UuidExtensionArray(values=[0, uuid4()])

..  parsed-literal::

    <UuidExtensionArray>
    [UUID('00000000-0000-0000-0000-000000000000'), UUID('1b52636e-e863-471f-865d-f98627df10b1')]
    Length: 2, dtype: uuid

..  code:: ipython3

    s = pd.Series([uuid4()], dtype=UuidDtype(), name="s")
    s

..  parsed-literal::

    0    e8f04c2e-ed42-488e-9e96-fe6c80d06bf6
    Name: s, dtype: uuid

..  code:: ipython3

    pd.DataFrame(s)

..  raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>s</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>e8f04c2e-ed42-488e-9e96-fe6c80d06bf6</td>
        </tr>
      </tbody>
    </table>
    </div>

pyarrow integration
-------------------

convert from :mod:`pandas` :class:`~pandas.api.extensions.ExtensionArray` to
:class:`pyarrow.UuidArray` automatically …

..  code:: ipython3

    import pyarrow as pa

    arr = pa.array(s.array)
    arr

..  parsed-literal::

    <pyarrow.lib.UuidArray object at 0x1169ccd40>
    [
      E8F04C2EED42488E9E96FE6C80D06BF6
    ]

And back manually. There is no registry,
so neither :class:`pyarrow.UuidArray` nor :mod:`pandas` knows,
only our :class:`~pandas_uuid.UuidDtype`:

..  code:: ipython3

    pd.Series(arr, dtype=UuidDtype())

..  parsed-literal::

    0    e8f04c2e-ed42-488e-9e96-fe6c80d06bf6
    dtype: uuid
