from __future__ import annotations
from typing import Optional, Union, Any


class AFunction:
    def __init__(self, func):
        self.func = func

    def __add__(self, other):
        return AFunction(lambda x: self.func(x) + (x[other] if isinstance(other, AColumn) else other))

    def __sub__(self, other):
        return AFunction(lambda x: self.func(x) - (x[other] if isinstance(other, AColumn) else other))

    def __mul__(self, other):
        return AFunction(lambda x: self.func(x) * (x[other] if isinstance(other, AColumn) else other))

    def __truediv__(self, other):
        return AFunction(lambda x: self.func(x) / (x[other] if isinstance(other, AColumn) else other))


class AColumn:
    def __init__(self, name: str, func: Optional[AFunction] = None):
        self.name = name
        self.func = func

    def __str__(self):
        return self.name

    def __add__(self, other: Union[AColumn, Any]):
        return AFunction(lambda x: x[self] + (x[other] if isinstance(other, AColumn) else other))

    def __sub__(self, other: Union[AColumn, Any]):
        return AFunction(lambda x: x[self] - (x[other] if isinstance(other, AColumn) else other))

    def __mul__(self, other: Union[AColumn, Any]):
        return AFunction(lambda x: x[self] * (x[other] if isinstance(other, AColumn) else other))

    def __truediv__(self, other: Union[AColumn, Any]):
        return lambda x: x[self] / (x[other] if isinstance(other, AColumn) else other)

