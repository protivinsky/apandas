from __future__ import annotations

import functools
from typing import Optional, Callable, Union, Any
import operator
import pandas as pd


def _method_delegate(cls):
    """
    A list of operators and pandas.Series methods that are leveraged to work correctly in on AFunctions
    (and AColumns) - the operators and methods are applied to the underlying pd.Series after the lookup
    of the AFunction (or AColumn) in the AFrame (and possibly after the application of other calculations).
    """

    pd_series_methods = [
        '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__', '__pow__', '__matmul__',
        '__and__', '__or__', '__lt__', '__gt__', '__le__', '__ge__', '__eq__', '__ne__', '__abs__',
        'diff', 'round', 'fillna', 'replace', 'cumsum', 'cumprod',
    ]

    for name in pd_series_methods:
        # to deal with Python late binding correctly
        def wrapper(_func=getattr(pd.Series, name)):
            @functools.wraps(_func)
            def func(*args, **kwargs):
                return cls.function_wrapper(_func, *args, **kwargs)
            func.__doc__ = f'Wrapper for pd.Series.{name} method. See the doc at ' \
                           f'https://pandas.pydata.org/docs/reference/api/pandas.Series.{name}.html.'
            return func
        setattr(cls, name, wrapper())

    return cls


@_method_delegate
class AFunction:
    """ Represents any function that can be applied to an AFrame / pd.DataFrame. """
    def __init__(self, func: Union[Callable, Any]):
        # If func is not callable, treat it as a content to fill in.
        # Or do I want to provide a specific static function such as AFunction.content?
        if not callable(func) and func is not None:  # scalar or iterable content
            self.func = lambda _: func
        elif isinstance(func, AFunction) and func.func is not None:  # unwrap AFunction, to avoid unnecessary nesting
            self.func = func.func
        else:
            self.func = func

    def from_frame(self, af):
        return self.func(af)

    def __call__(self, af):
        return self.from_frame(af)

    def __repr__(self):
        return f"AFunction[{self.func.__name__ if hasattr(self.func, '__name__') else self.func}]"

    __str__ = __repr__

    @staticmethod
    def function_wrapper(func, *args, **kwargs):
        @functools.wraps(func)
        def applied_func(af):
            modified_args = [(x.from_frame(af) if isinstance(x, AFunction) else x) for x in args]
            modified_kwargs = {k: (v.from_frame(af) if isinstance(v, AFunction) else v) for k, v in kwargs.items()}
            return func(*modified_args, **modified_kwargs)
        return AFunction(applied_func)

    def __hash__(self):
        return hash(self.func)


class ANamedFunction(AFunction):
    """
    A function containing name. Can be useful for custom operations, where the results cannot be included back
    into the original frame (such as filtering or groupby aggregations).
    """
    def __init__(self, name: str, func: Union[Callable, Any]):
        self.name = name
        super().__init__(func)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"ANamedFunction['{self.name}']"

    def from_frame(self, af):
        result = self.func(af)
        if isinstance(result, pd.Series):
            return result.rename(self.name)
        else:
            return result


class AColumn(ANamedFunction):
    """
    Just a named function that can be accessed (and constructed on-the-fly if needed) from an AFrame.
    No check is done that the shape conforms the DataFrame (hence can be added). Cached in the frame on the calculation.
    """
    def __init__(self, name: str, func: Optional[Union[AFunction, Callable, Any]] = None, override: bool = False):
        super().__init__(name=name, func=func)
        self.override = override  # if True, will override the column with the same name in the AFrame on the first call
        self.been_applied = False

    def from_frame(self, af):
        res = af[self]
        return res

    def __repr__(self):
        return f"AColumn['{self.name}']"

    def __hash__(self):
        return hash(self.name) ^ hash(self.func)
