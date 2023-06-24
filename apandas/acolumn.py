from __future__ import annotations

import functools
from typing import Optional, Callable, Union, Any
import operator
import pandas as pd


def _arithmetic_delegate(cls):
    """
    A list of operators and pandas.Series methods that are leveraged to work correctly in on AFunctions
    (and AColumns) - the operators and methods are applied to the underlying pd.Series after the lookup
    of the AFunction (or AColumn) in the AFrame (and possibly after the application of other calculations).
    """
    operators = {
        '__add__': operator.add,
        '__sub__': operator.sub,
        '__mul__': operator.mul,
        '__truediv__': operator.truediv,
        '__floordiv__': operator.floordiv,
        '__mod__': operator.mod,
        '__pow__': operator.pow,
        '__matmul__': operator.matmul,
        '__and__': operator.and_,
        '__or__': operator.or_,
        '__lt__': operator.lt,
        '__gt__': operator.gt,
        '__le__': operator.le,
        '__ge__': operator.ge,
        '__eq__': operator.eq,
        '__ne__': operator.ne,
        '__abs__': operator.abs,
    }

    pd_series_methods = [
        'diff',
        'round',
        'fillna',
        'replace',
        'cumsum',
        'cumprod',
    ]

    for name, op in operators.items():
        # TODO: it might be better to do it as a proper wrapper with doc, name, type hints etc.
        # need to deal with Python late binding correctly
        def wrapper(_func=op):
            @functools.wraps(_func)
            def func(*args, **kwargs):
                return cls.function_wrapper(_func, *args, **kwargs)
            return func
        setattr(cls, name, wrapper())

    for name in pd_series_methods:
        if not hasattr(cls, name):
            def wrapper(_func=getattr(pd.Series, name)):
                @functools.wraps(_func)
                def func(*args, **kwargs):
                    return cls.function_wrapper(_func, *args, **kwargs)
                func.__doc__ = f'A Wrapper for pd.Series.{name} method. See the doc at ' \
                               f'https://pandas.pydata.org/docs/reference/api/pandas.Series.{name}.html.'
                return func
            # need to deal with Python late binding correctly
            setattr(cls, name, wrapper())

            # setattr(cls, name, lambda *args, _func=getattr(pd.Series, name), **kwargs: cls.function_wrapper(
            #     _func, *args, **kwargs))

    return cls


@_arithmetic_delegate
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

    def from_frame(self, af):
        result = super().__call__(af)
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

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"AColumn['{self.name}']"

    def __hash__(self):
        return hash(self.name) ^ hash(self.func)
