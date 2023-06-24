import pytest
import pandas as pd
from apandas import AColumn, ASeries, AFrame


@pytest.fixture()
def x_y_z_and_af():
    x = AColumn('x')
    y = AColumn('y')
    z = AColumn('z', x * y + x * y)

    af = AFrame()
    af[x] = [1, 2, 3]
    af[y] = [3, 3, 3]
    return x, y, z, af


def test_series(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
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


def test_series_groupby(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    i = AColumn(name='i', func=AColumn('index'))
    af_big = AFrame(pd.concat([af, af + 5]).reset_index())
    # again, everything should be just calculated on the fly
    res = af_big.groupby(i)[z].agg(['min', 'max'])
    assert isinstance(res, AFrame)
    pd.testing.assert_frame_equal(res.to_pandas(), pd.DataFrame(
        {'min': [6, 12, 18], 'max': [96, 112, 128]}, index=pd.Series([0, 1, 2], name='i')))


def test_groupby_series_apply(x_y_z_and_af):
    x, y, z, af = x_y_z_and_af
    i = AColumn(name='i', func=AColumn('index'))
    af_big1 = AFrame(pd.concat([af, af + 5]).reset_index())
    af_big2 = af_big1.copy()
    af_big3 = af_big1.copy()
    af_big4 = af_big1.copy()

    # res_apply_z_in = af_big4.groupby(i).apply(lambda df: df[[x, z]].sum())

    # for ASeriesGroupBy
    af_big1 = AFrame(pd.concat([af, af + 5]).reset_index())
    af_big2 = af_big1.copy()
    af_big3 = af_big1.copy()
    af_big4 = af_big1.copy()

    # === compare .sum, .agg and .app ===
    # .sum()
    res_sum_y = af_big1.groupby(i)[y].sum()
    pd_res_sum_y = af_big1.to_pandas().groupby('i')['y'].sum()
    res_sum_z = af_big1.groupby(i)[z].sum()
    pd_res_sum_z = af_big1.to_pandas().groupby('i')['z'].sum()
    assert isinstance(res_sum_y, ASeries)
    assert isinstance(res_sum_z, ASeries)
    pd.testing.assert_series_equal(res_sum_y.to_pandas(), pd_res_sum_y)
    pd.testing.assert_series_equal(res_sum_z.to_pandas(), pd_res_sum_z)
    # .agg()
    res_agg_y = af_big2.groupby(i)[y].agg('sum')
    pd_res_agg_y = af_big2.to_pandas().groupby('i')['y'].agg('sum')
    res_agg_z = af_big2.groupby(i)[z].agg('sum')
    pd_res_agg_z = af_big2.to_pandas().groupby('i')['z'].agg('sum')
    assert isinstance(res_agg_y, ASeries)
    assert isinstance(res_agg_z, ASeries)
    pd.testing.assert_series_equal(res_agg_y.to_pandas(), pd_res_agg_y)
    pd.testing.assert_series_equal(res_agg_z.to_pandas(), pd_res_agg_z)
    # .apply()
    res_apply_y = af_big3.groupby(i)[y].apply(lambda df: df.sum())
    pd_res_apply_y = af_big3.to_pandas().groupby('i')['y'].apply(lambda df: df.sum())
    res_apply_z = af_big3.groupby(i)[z].apply(lambda df: df.sum())
    pd_res_apply_z = af_big3.to_pandas().groupby('i')['z'].apply(lambda df: df.sum())
    assert isinstance(res_apply_y, ASeries)
    assert isinstance(res_apply_z, ASeries)

