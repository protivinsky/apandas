import pytest
import pandas as pd
from apandas import AFrame, AColumn


def test_initialization():
    # Only testing that I can create all objects even with custom functions
    x = AColumn('x')
    y = AColumn('y')
    u = AColumn('u', x + y)
    af = AFrame(verbose=True)


def test_types():
    # define columns you will add to the dataframe in the beginning
    x = AColumn('x')
    y = AColumn('y')

    # create a dataframe
    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]

    # AFrame is a subclass of pandas.DataFrame
    assert isinstance(af, pd.DataFrame)
    # its keys are actually strings
    assert all(isinstance(c, str) for c in af.columns)
    assert all(af.columns == ['x', 'y'])


def test_calculation():
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

    # if you just access the custom defined analytics, they are created on the fly and named with the given names
    pd.testing.assert_series_equal(af[u], pd.Series([4, 5, 6], name='u'))
    pd.testing.assert_series_equal(af[v], pd.Series([3, 6, 9], name='v'))
    pd.testing.assert_series_equal(af[z], pd.Series([-1, 1, 3], name='z'))

    # you even do not need to create the intermediate columns, you can just ask for the final ones
    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]
    pd.testing.assert_series_equal(af[z], pd.Series([-1, 1, 3], name='z'))


def test_getitem_iterable():
    # define columns you will add to the dataframe in the beginning
    x = AColumn('x')
    y = AColumn('y')

    # and define the transformations of these columns
    u = AColumn('u', x + y)
    v = AColumn('v', x * y)

    # create a dataframe
    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]

    # if you just access the custom defined analytics, they are created on the fly and named with the given names
    pd.testing.assert_frame_equal(af[[u, v]], pd.DataFrame({'u': [4, 5, 6], 'v': [3, 6, 9]}))
    pd.testing.assert_frame_equal(af[[u, v]], af[['u', 'v']])
