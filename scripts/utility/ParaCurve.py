import numpy as np
from math import sqrt, cos, atan
from typing import List, Tuple


class ParaCurve:
    def __init__(self, x0: float, y0: float) -> None:
        self.x0, self.y0 = x0, y0

    def set(self, a_: float, b_: float, c_: float) -> None:
        self.a, self.b, self.c = a_, b_, c_
        self.t1 = [4 * self.a * self.a, 6 * self.a * self.b, 4 * self.a * (self.c - self.y0) + 2 * self.b * self.b + 2, 2 * self.b * (self.c - self.y0) - 2 * self.x0]
        self.t2 = [12 * self.a * self.a, 12 * self.a * self.b, 4 * self.a * (self.c - self.y0) + 2 * self.b * self.b + 2]

    def val(self, x: float) -> float:
        return self.a * x * x + self.b * x + self.c

    def vald(self, x: float) -> float:
        return 2 * self.a * x + self.b

    def calc(self, x: float) -> float:
        return self.t1[0] * x * x * x + self.t1[1] * x * x + self.t1[2] * x + self.t1[3]

    def calcd(self, x: float) -> float:
        return self.t2[0] * x * x + self.t2[1] * x + self.t2[2]

    def perpendicular(self) -> float:
        x = self.x0
        for _ in range(5):
            x = x - self.calc(x) / self.calcd(x)
        return x

    def dist(self):
        pass

