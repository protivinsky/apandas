import pytest
import pandas as pd
from apandas import AFrame, AColumn
from apandas.aframe import AFrameGroupBy


def test_initialization():
    # Only testing that I can create all objects even with custom functions
    x = AColumn('x')
    y = AColumn('y')
    u = AColumn('u', x + y)
    af = AFrame(verbose=True)


@pytest.fixture()
def x_y_z_and_af():
    x = AColumn('x')
    y = AColumn('y')
    z = AColumn('z', x * y + x * y)

    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]
    return x, y, z, af


def test_types(x_y_z_and_af):
    # define columns you will add to the dataframe in the beginning
    x, y, _, af = x_y_z_and_af

    # AFrame is a subclass of pandas.DataFrame
    assert isinstance(af, pd.DataFrame)
    # its keys are actually strings
    assert all(isinstance(c, str) for c in af.columns)
    assert all(af.columns == ['x', 'y'])


def test_calculation(x_y_z_and_af):
    # define columns you will add to the dataframe in the beginning
    x, y, _, af = x_y_z_and_af

    # and define the transformations of these columns
    u = AColumn('u', x + y)
    v = AColumn('v', x * y)
    z = AColumn('z', v - u)

    # if you just access the custom defined analytics, they are created on the fly and named with the given names
    pd.testing.assert_series_equal(af[u], pd.Series([4, 5, 6], name='u'))
    pd.testing.assert_series_equal(af[v], pd.Series([3, 6, 9], name='v'))
    pd.testing.assert_series_equal(af[z], pd.Series([-1, 1, 3], name='z'))

    # you even do not need to create the intermediate columns, you can just ask for the final ones
    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]
    pd.testing.assert_series_equal(af[z], pd.Series([-1, 1, 3], name='z'))


def test_getitem_iterable(x_y_z_and_af):
    x, y, _, af = x_y_z_and_af

    # and define the transformations of these columns
    u = AColumn('u', x + y)
    v = AColumn('v', x * y)

    # if you just access the custom defined analytics, they are created on the fly and named with the given names
    pd.testing.assert_frame_equal(af[[u, v]], pd.DataFrame({'u': [4, 5, 6], 'v': [3, 6, 9]}))
    # pd.testing.assert_frame_equal(af[[u, v]], af[['u', 'v']])


def test_operator_priorities(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    pd.testing.assert_series_equal(af[z], pd.Series([6, 12, 18], name='z'))


def test_drop_columns(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    af.add_acolumn(z)

    new_af = af.drop(columns=[x, z])
    # orig af is unmodified
    assert all(c in af.columns for c in ['x', 'y', 'z'])
    # but the new one has only 'y'
    assert 'y' in new_af.columns
    assert all(c not in new_af.columns for c in ['x', 'z'])


def test_rename_columns(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    af.add_acolumn(z)

    new_af = af.rename(columns={x: 'a', z: 'b'})
    # orig af is unmodified
    assert all(c in af.columns for c in ['x', 'y', 'z'])
    # new cols are in the new af
    assert all(c in new_af.columns for c in ['a', 'y', 'b'])
    # old cols are not in the new af
    assert all(c not in new_af.columns for c in ['x', 'z'])
    # and new cols are equal to the original series (up to names)
    pd.testing.assert_series_equal(new_af['a'], af['x'].rename('a'))
    pd.testing.assert_series_equal(new_af['b'], af['z'].rename('b'))


def test_preserve_aframe(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    af.add_acolumn(z)

    # basic operations (copy, rename, drop, access) should preserve the AFrame
    assert isinstance(af, AFrame)
    assert isinstance(af.copy(), AFrame)
    assert isinstance(af[x, z], AFrame)
    assert isinstance(af.drop(columns=x), AFrame)
    assert isinstance(af.rename(columns={x: 'a'}), AFrame)


def test_groupby(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    # af.add_acolumn(z)

    # result of pandas operations and apandas has to be identical
    df = pd.DataFrame(af)
    pd_res = df.groupby('x')['y'].sum()
    apd_res = af.groupby(x)[y].sum()
    pd.testing.assert_series_equal(pd_res, apd_res)



    df

    dfg = pd.core.groupby.generic.DataFrameGroupBy





