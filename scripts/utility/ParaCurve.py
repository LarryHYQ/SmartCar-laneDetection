import numpy as np
from math import sqrt, cos, atan
from typing import List, Tuple


def calc(x: float, x0: float, y0: float, a: float, b: float, c: float):
    return 4 * a * a * x * x * x + 6 * a * b * x * x + 4 * a * c * x - 4 * a * x * y0 + 2 * b * b * x + 2 * b * c - 2 * b * y0 + 2 * x - 2 * x0


def calcd(x: float, y0: float, a: float, b: float, c: float):
    return 12 * a * a * x * x + 12 * a * b * x + 4 * a * c - 4 * a * y0 + 2 * b * b + 2


def newtonMethod(x0: float, calc, calcd):
    for _ in range(100):
        d = calcd(x0)
        if d < 0.000001:
            break
        x0 = x0 - calc(x0) / d
    return x0


class ParaCurve:
    def __init__(self, x0: float, y0: float):
        self.x0, self.y0 = x0, y0

    def set(self, a_: float, b_: float, c_: float) -> None:
        self.a, self.b, self.c = a_, b_, c_

    def val(self, x: float) -> float:
        return self.a * x * x + self.b * x + self.c

    def calc(self, x):
        return calc(x, self.x0, self.y0, self.a, self.b, self.c)

    def calcd(self, x):
        return calcd(x, self.y0, self.a, self.b, self.c)

    def perpendicular(self) -> float:
        return newtonMethod(self.x0, self.calc, self.calcd)

