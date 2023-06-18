import pytest
import pandas as pd
from apandas import AFrame, ASeries, AColumn


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


def test_string_keys(x_y_z_and_af):
    x, y, _, af = x_y_z_and_af
    # AFrame behaves correctly like a DataFrame with keys as strings
    af['foo'] = [10, 9, 8]
    pd.testing.assert_series_equal(af['foo'].to_pandas(), pd.Series([10, 9, 8], name='foo'))


def test_calculation(x_y_z_and_af):
    # define columns you will add to the dataframe in the beginning
    x, y, _, af = x_y_z_and_af

    # and define the transformations of these columns
    u = AColumn('u', x + y)
    v = AColumn('v', x * y)
    z = AColumn('z', v - u)

    # if you just access the custom defined analytics, they are created on the fly and named with the given names
    pd.testing.assert_series_equal(af[u].to_pandas(), pd.Series([4, 5, 6], name='u'))
    pd.testing.assert_series_equal(af[v].to_pandas(), pd.Series([3, 6, 9], name='v'))
    pd.testing.assert_series_equal(af[z].to_pandas(), pd.Series([-1, 1, 3], name='z'))

    # you even do not need to create the intermediate columns, you can just ask for the final ones
    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]
    pd.testing.assert_series_equal(af[z].to_pandas(), pd.Series([-1, 1, 3], name='z'))


def test_getitem_iterable(x_y_z_and_af):
    x, y, _, af = x_y_z_and_af

    # and define the transformations of these columns
    u = AColumn('u', x + y)
    v = AColumn('v', x * y)

    # if you just access the custom defined analytics, they are created on the fly and named with the given names
    assert isinstance(af[[u, v]], AFrame)
    pd.testing.assert_frame_equal(af[[u, v]], pd.DataFrame({'u': [4, 5, 6], 'v': [3, 6, 9]}))


def test_operator_priorities(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    pd.testing.assert_series_equal(af[z].to_pandas(), pd.Series([6, 12, 18], name='z'))


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
    # and new cols are ASeries
    assert isinstance(new_af['a'], ASeries)
    assert isinstance(af['x'].rename('a'), ASeries)


def test_preserve_aframe(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    af.add_acolumn(z)

    # basic operations (copy, rename, drop, access) should preserve the AFrame
    assert isinstance(af, AFrame)
    assert isinstance(af.copy(), AFrame)
    assert isinstance(af[[x, z]], AFrame)
    assert isinstance(af.drop(columns=x), AFrame)
    assert isinstance(af.rename(columns={x: 'a'}), AFrame)


def test_groupby(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af

    # result of pandas operations and apandas has to be identical
    df = af.to_pandas()
    assert isinstance(df, pd.DataFrame)
    pd_res = df.groupby('x').sum()
    apd_res = af.groupby(x).sum()
    assert isinstance(apd_res, AFrame)
    pd.testing.assert_frame_equal(pd_res, apd_res.to_pandas())

    # can also apply the operation a selected column
    pd_res = df.groupby('x')['y'].sum()
    apd_res = af.groupby(x)[y].sum()
    assert isinstance(apd_res, ASeries)
    pd.testing.assert_series_equal(pd_res, apd_res.to_pandas())

    # can construct the missing columns on the fly even in groupby
    apd_res = af.groupby(x)[[y, z]].sum()
    df['z'] = 2 * df['x'] * df['y']
    pd_res = df.groupby('x')[['y', 'z']].sum()
    assert isinstance(apd_res, AFrame)
    pd.testing.assert_frame_equal(pd_res, apd_res.to_pandas())


def test_series(x_y_z_and_af):
    x, y, u, af = x_y_z_and_af

    xs = af[x]
    # returns an ASeries
    assert isinstance(xs, ASeries)

    # conversions Series -> DataFrame returns AFrame
    assert isinstance(xs.reset_index(), AFrame)

    # series can be renamed via rename or name and the result has a string key
    u = AColumn('u')
    us = xs.rename(u)
    xs.rename('u')
    assert us.name == 'u'

    # ASeries can be copied correctly.
    xs_copied = xs.copy()
    assert isinstance(xs_copied, ASeries)
    pd.testing.assert_series_equal(xs.to_pandas(), xs_copied.to_pandas())

