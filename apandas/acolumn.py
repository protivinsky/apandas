from __future__ import annotations
from typing import Optional, Union, Any
import operator


class AFunction:
    """ Represents any function that can be applied to an AFrame / pd.DataFrame. """
    def __init__(self, func):
        # If func is an AFunction, we can unwrap (if .func is defined - not the case for some AColumns)
        self.func = func.func if isinstance(func, AFunction) and func.func is not None else func

    def from_frame(self, af):
        return self.func(af)

    def __call__(self, af):
        return self.from_frame(af)

    def __add__(self, other):
        # this can be done differently, in metaclass or otherwise
        return AFunction.apply_operator(operator.add, self, other)

    def __sub__(self, other):
        return AFunction.apply_operator(operator.sub, self, other)

    def __mul__(self, other):
        return AFunction.apply_operator(operator.mul, self, other)

    def __truediv__(self, other):
        return AFunction.apply_operator(operator.truediv, self, other)

    def __lt__(self, other):
        return AFunction.apply_operator(operator.lt, self, other)

    def __gt__(self, other):
        return AFunction.apply_operator(operator.gt, self, other)

    def __le__(self, other):
        return AFunction.apply_operator(operator.le, self, other)

    def __ge__(self, other):
        return AFunction.apply_operator(operator.ge, self, other)

    def __eq__(self, other):
        return AFunction.apply_operator(operator.eq, self, other)

    def __ne__(self, other):
        return AFunction.apply_operator(operator.ne, self, other)

    @staticmethod
    def apply_operator(op, *args):
        return AFunction(lambda af: op(*[(x.from_frame(af) if isinstance(x, AFunction) else x) for x in args]))

    def __hash__(self):
        return hash(self.func)


class AColumn(AFunction):
    """
    Just a named function that can be accessed (and constructed on-the-fly if needed) from an AFrame.
    No check is done that the shape conforms the DataFrame (hence can be added). Cached in the frame on the calculation.
    """
    def __init__(self, name: str, func: Optional[AFunction] = None):
        self.name = name
        super().__init__(func)

    def from_frame(self, af):
        return af[self]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"AColumn['{self.name}']"

    def __hash__(self):
        return hash(self.name) ^ hash(self.func)
