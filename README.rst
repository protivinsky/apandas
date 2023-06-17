|pytest-badge| |doc-badge|

..  |pytest-badge| image:: https://github.com/protivinsky/apandas/actions/workflows/pytest.yaml/badge.svg
    :alt: pytest

..  |doc-badge| image:: https://github.com/protivinsky/apandas/actions/workflows/builddoc.yaml/badge.svg
    :alt: doc
    :target: https://protivinsky.github.io/apandas/index.html

APandas: Lightweight wrapper to support custom analytics in Pandas
==================================================================

- Allows for separation of the definition of the analytic calculation and its application. Composed analytics are
  supported.
- If other analytics are needed, they are calculated on-the-fly based on their definitions.
- Dataframe columns can be accessed directly by column variables (hence allowing for at least some intellisense).

At the moment, only basic arithmetic operations are supported - this is mostly a proof of concept. It should be
easy to add support for other operations or for numpy universal functions. Most complicated functions can be
passed in as lambdas.

Basic example
-------------

.. code:: python

    from apandas import AColumn, AFrame

    # define columns you will add to the dataframe in the beginning
    x = AColumn('x')
    y = AColumn('y')

    # and define the transformations of these columns
    u = AColumn('u', x + y)
    v = AColumn('v', x * y)
    z = AColumn('z', v - u)

    # create a dataframe
    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]

    # now you can just access the final analytic or use it in computation - everything will be created on the fly
    af[z]  # there is [-1, 1, 3] and the columns is named 'z'

