import pytest
import numpy as np
import pandas as pd
from apandas import ANamedFunction, AColumn, AFrame, ASeries


@pytest.fixture()
def x_y_and_af():
    af = AFrame()
    x = AColumn('x')
    af[x] = np.arange(-3, 4)
    y = AColumn('y')
    af[y] = np.arange(3, -4, -1)
    return x, y, af


def test_scalar_or_other_content(x_y_and_af):
    x, y, af = x_y_and_af
    u = AColumn('u', 2)
    pd.testing.assert_series_equal(af[u].to_pandas(), pd.Series([2] * 7, name='u'))
    v = AColumn('v', np.arange(1, 8))
    pd.testing.assert_series_equal(af[v].to_pandas(), pd.Series(np.arange(1, 8), name='v'))


def test_arithmetic(x_y_and_af):
    x, y, af = x_y_and_af
    # __add__
    z = AColumn('z', x + y)
    pd.testing.assert_series_equal(af[z], af[x] + af[y], check_names=False)
    # __sub__
    z = AColumn('z', x - y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] - af[y], check_names=False)
    # __mul__
    z = AColumn('z', x * y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] * af[y], check_names=False)
    # __truediv__
    z = AColumn('z', x / y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] / af[y], check_names=False)
    # __floordiv__
    two = AColumn('two', 2)
    z = AColumn('z', x // two, override=True)
    pd.testing.assert_series_equal(af[z], af[x] // af[two], check_names=False)
    # __mod__
    z = AColumn('z', x % y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] % af[y], check_names=False)
    # __pow__
    powers = AColumn('powers', [1, 2, 1, 2, 1, 2, 3])
    z = AColumn('z', x ** powers, override=True)
    pd.testing.assert_series_equal(af[z], af[x] ** af[powers], check_names=False)
    # __lt__
    z = AColumn('z', x < y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] < af[y], check_names=False)
    # __le__
    z = AColumn('z', x <= y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] <= af[y], check_names=False)
    # __eq__
    z = AColumn('z', x == y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] == af[y], check_names=False)
    # __ne__
    z = AColumn('z', x != y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] != af[y], check_names=False)
    # __gt__
    z = AColumn('z', x > y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] > af[y], check_names=False)
    # __ge__
    z = AColumn('z', x >= y, override=True)
    pd.testing.assert_series_equal(af[z], af[x] >= af[y], check_names=False)
    # __and__
    z = AColumn('z', (x < 2) & (y < 2), override=True)
    pd.testing.assert_series_equal(af[z], (af[x] < 2) & (af[y] < 2), check_names=False)
    # __or__
    z = AColumn('z', (x > 1) | (y > 1), override=True)
    pd.testing.assert_series_equal(af[z], (af[x] > 1) | (af[y] > 1), check_names=False)
    # __abs__
    z = AColumn('z', abs(x), override=True)
    pd.testing.assert_series_equal(af[z], abs(af[x]), check_names=False)
    # arithmetic as function
    f = x + y
    pd.testing.assert_series_equal(f(af), af[x] + af[y], check_names=False)


def test_operator_priorities(x_y_and_af):
    x, y, af = x_y_and_af
    z = AColumn('z', x * y + x * y)
    exp_z = [-18, -8, -2, 0, -2, -8, -18]
    pd.testing.assert_series_equal(af[z].to_pandas(), pd.Series(exp_z, name='z'))


def test_series_methods(x_y_and_af):
    x, y, af = x_y_and_af
    f = x.diff()
    pd.testing.assert_series_equal(f(af), af[x].diff(), check_names=False)
    f = x.diff(periods=-2)
    pd.testing.assert_series_equal(f(af), af[x].diff(periods=-2), check_names=False)
    f = (x / 2).round()
    pd.testing.assert_series_equal(f(af), (af[x] / 2).round(), check_names=False)
    z = AColumn('z', x.replace(to_replace=[-3, -2, -1], value=np.nan))
    pd.testing.assert_series_equal(af[z], af[x].replace(to_replace=[-3, -2, -1], value=np.nan), check_names=False)
    f = z.fillna(99)
    pd.testing.assert_series_equal(f(af), af[z].fillna(99), check_names=False)
    f = x.cumsum()
    pd.testing.assert_series_equal(f(af), af[x].cumsum(), check_names=False)
    f = x.cumprod()
    pd.testing.assert_series_equal(f(af), af[x].cumprod(), check_names=False)


def test_named_function(x_y_and_af):
    x, y, af = x_y_and_af
    f = x + y
    pd.testing.assert_series_equal(af[f], af[x] + af[y], check_names=False)
    assert af[f].name is None
    nf = ANamedFunction('f', x + y)
    pd.testing.assert_series_equal(af[nf], af[x] + af[y], check_names=False)
    assert 'f' not in af.columns
    assert af[nf].name == 'f'


def test_advanced_function(x_y_and_af):
    x, y, af = x_y_and_af
    y_filtered_diff = ANamedFunction('y_filtered_diff', lambda af: af[x % 2 == 0][y.diff()])
    pd.testing.assert_series_equal(af[y_filtered_diff], ASeries([np.nan, -2, -2], name='y_filtered_diff'),
                                   check_index=False)
    two = AColumn('two', 2)
    z = AColumn('z', x // two, override=True)
    magic = ANamedFunction('magic', lambda af: af.set_index(x, drop=False)[z % 2 != 0][x.diff()].rename(
        'x_diff').reset_index())
    af[magic]