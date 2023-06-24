#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
|pytest-badge| |doc-badge|

..  |pytest-badge| image:: https://github.com/protivinsky/apandas/actions/workflows/pytest.yaml/badge.svg
    :alt: pytest

..  |doc-badge| image:: https://github.com/protivinsky/apandas/actions/workflows/builddoc.yaml/badge.svg
    :alt: doc
    :target: https://protivinsky.github.io/apandas/index.html

- Allows for separation between analytic definition and its calculation in the dataframe.
- The analytic definitions are composable.
- If analytic columns have not yet been added, they are calculated on-the-fly (including their dependencies if needed).
- Accessed directly by analytic instances (to leverage intellisense support).

At the moment, only basic arithmetic operations are supported - this is mostly a proof of concept. It should be
easy to add support for other operations or for numpy universal functions. Most complicated functions can be
passed in as lambdas or defined as functions.

"APandas" stands for "Analytic Pandas".

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

Installation
------------

At the moment, the library is available only on [test.pypi.org](https://test.pypi.org/project/apandas/) until it gets
more mature and stable. To install it, run:

.. code:: bash

    pip install -i https://test.pypi.org/simple/ apandas

"""

from .version import VERSION as __version__
from .acolumn import AFunction, ANamedFunction, AColumn
from .aframe import AFrame, ASeries

__all__ = ['AFunction', 'ANamedFunction', 'AColumn', 'AFrame', 'ASeries']
__author__ = 'Tomas Protivinsky'
