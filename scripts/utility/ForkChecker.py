from math import sqrt, cos, acos, degrees, radians, atan2
from scripts.transform import axisTransform as _axis
from functools import partial
from Config import *


def vectorCos(x1: float, y1: float, x2: float, y2: float) -> float:
    return (x1 * x2 + y1 * y2) / sqrt((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2))


def dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    x = x1 - x2
    y = y1 - y2
    return x * x + y * y


class FrontForkChecker:
    def __init__(self, perMat, line) -> None:
        self.n = 0
        self.pi = [0.0] * 4
        self.pj = [0.0] * 4
        self.axisTransform = partial(_axis, perMat=perMat)
        self.line = line
        self.res = False

    def reset(self) -> None:
        self.n = 0
        self.res = False

    def lost(self) -> None:
        self.n = 0

    def check(self) -> bool:

        if self.n < 4:
            return False
        x1 = self.pi[(self.n + 1) & 3] - self.pi[self.n & 3]
        x2 = self.pi[(self.n + 2) & 3] - self.pi[(self.n + 3) & 3]
        if x1 < 0 or x2 < 0:
            return False
        y1 = self.pj[self.n & 3] - self.pj[(self.n + 1) & 3]
        y2 = self.pj[(self.n + 3) & 3] - self.pj[(self.n + 2) & 3]
        res = vectorCos(x1, y1, x2, y2)
        return cos(radians(FORKHIGH)) < res < cos(radians(FORKLOW))

    def draw(self):
        pts = [(self.pi[i & 3], self.pj[i & 3]) for i in range(self.n, self.n + 4)]
        self.line(pts[0], pts[1], (255, 0, 0), 4)
        self.line(pts[2], pts[3], (255, 0, 0), 4)

    def update(self, i: int, j: int):
        if self.res:
            return
        if not FORKUPCUT < i < N - FORKDOWNCUT:
            self.reset()
        else:
            self.pi[self.n & 3], self.pj[self.n & 3] = self.axisTransform(i, j)
            self.n += 1
            if self.check():
                self.draw()
                self.res = True


class SideForkChecker:
    def __init__(self, line) -> None:
        self.line = line
        self.n = 0
        self.pi = [0.0] * 7
        self.pj = [0.0] * 7
        self.hasLost = False
        self.res = False

    def reset(self) -> None:
        self.n = 0
        self.res = False
        self.hasLost = False

    def lost(self) -> None:
        self.hasLost = True

    def checkDist(self, pi_: float, pj_: float) -> bool:
        if not self.n:
            return True
        return dist2(pi_, pj_, self.pi[(self.n - 1) % 7], self.pj[(self.n - 1) % 7]) < 25

    def check(self) -> bool:
        if self.n < 7:
            return False
        r1 = vectorCos(self.pi[(self.n + 1) % 7] - self.pi[(self.n + 2) % 7], self.pj[(self.n + 1) % 7] - self.pj[(self.n + 2) % 7], self.pi[(self.n + 5) % 7] - self.pi[(self.n + 4) % 7], self.pj[(self.n + 5) % 7] - self.pj[(self.n + 4) % 7])
        r2 = vectorCos(self.pi[self.n % 7] - self.pi[(self.n + 2) % 7], self.pj[self.n % 7] - self.pj[(self.n + 2) % 7], self.pi[(self.n + 6) % 7] - self.pi[(self.n + 4) % 7], self.pj[(self.n + 6) % 7] - self.pj[(self.n + 4) % 7])
        return cos(radians(FORKHIGH)) < r1 < cos(radians(FORKLOW)) and cos(radians(FORKHIGH)) < r2 < cos(radians(FORKLOW))

    def draw(self) -> None:
        I = (self.n % 7, (self.n + 2) % 7, (self.n + 4) % 7, (self.n + 6) % 7)
        pts = [(self.pi[i], self.pj[i]) for i in I]
        self.line(pts[0], pts[1], (255, 0, 0), 4)
        self.line(pts[2], pts[3], (255, 0, 0), 4)

    def update(self, pi_: float, pj_: float):
        if self.res or self.hasLost:
            return
        if not self.checkDist(pi_, pj_):
            self.lost()
            return
        self.pi[self.n % 7] = pi_
        self.pj[self.n % 7] = pj_
        self.n += 1
        if self.check():
            self.draw()
            self.res = True


__all__ = ["FrontForkChecker", "SideForkChecker"]

