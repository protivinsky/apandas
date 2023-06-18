from __future__ import annotations
from typing import Optional, Union, Any
import operator


class AFunction:
    """ Represents any function that can be applied to an AFrame / pd.DataFrame. """
    def __init__(self, func):
        self.func = func

    def from_frame(self, af):
        return self.func(af)

    def __call__(self, af):
        return self.from_frame(af)

    def __add__(self, other):
        return AFunction.apply_operator(operator.add, self, other)

    def __sub__(self, other):
        return AFunction.apply_operator(operator.sub, self, other)

    def __mul__(self, other):
        return AFunction.apply_operator(operator.mul, self, other)

    def __truediv__(self, other):
        return AFunction.apply_operator(operator.truediv, self, other)

    @staticmethod
    def apply_operator(op, *args):
        return AFunction(lambda af: op(*[(x.from_frame(af) if isinstance(x, AFunction) else x) for x in args]))


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
