"""
参考：
https://blog.csdn.net/u011023470/article/details/111381695
https://blog.csdn.net/u011023470/article/details/111381298
"""
from typing import List, Tuple


class linePredictor:
    def __init__(self, count) -> None:
        self.count = count
        self.reset()

    def reset(self, default: int = 0) -> None:
        self.default = default
        self.n = 0
        self.i = self.i_ = -1
        self.a = self.b = 0.0
        self.X, self.Y = [0] * self.count, [0] * self.count
        self.x = self.y = self.x2 = self.xy = 0.0

    def update(self, x: int, y: int) -> None:
        if self.n < self.count:
            self.n += 1
            self.x += (x - self.x) / self.n
            self.y += (y - self.y) / self.n
            self.x2 += (x * x - self.x2) / self.n
            self.xy += (x * y - self.xy) / self.n
        else:
            self.x += (x - self.X[self.i]) / self.n
            self.y += (y - self.Y[self.i]) / self.n
            self.x2 += (x * x - self.X[self.i] * self.X[self.i]) / self.n
            self.xy += (x * y - self.X[self.i] * self.Y[self.i]) / self.n
        self.i_ = self.i
        self.i = (self.i + 1) % self.count
        self.X[self.i] = x
        self.Y[self.i] = y

        if self.n > 1:
            self.a = (self.xy - self.x * self.y) / (self.x2 - self.x * self.x)
            self.b = (self.x2 * self.y - self.x * self.xy) / (self.x2 - self.x * self.x)

    def val(self, x: int) -> int:
        return round(self.a * x + self.b) if self.n > 1 else self.default

    def angleCheck(self, x: int, y: int) -> bool:
        if self.n < 2:
            return True
        x1, x2 = self.X[self.i] - self.X[self.i_], x - self.X[self.i]
        y1, y2 = self.Y[self.i] - self.Y[self.i_], y - self.Y[self.i]
        dot = x1 * x2 + y1 * y2
        return (dot * dot << 1) // ((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2)) >= 1

    def dist2(self, x: int, y: int) -> float:
        if self.n < 2:
            return 0
        t = self.a * x - y + self.b
        return t * t / (self.a * self.a + self.b * self.b)


def polyfit1d(X: List[float], Y: List[float]) -> List[float]:
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0

    N = len(X)
    for x, y in zip(X, Y):
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_xy += x * y

    sum_x /= N
    sum_y /= N
    sum_x2 /= N
    sum_xy /= N

    a = (sum_xy - sum_x * sum_y) / (sum_x2 - sum_x * sum_x)
    b = (sum_x2 * sum_y - sum_x * sum_xy) / (sum_x2 - sum_x * sum_x)
    return [a, b]


def polyfit2d(X: List[float], Y: List[float]) -> List[float]:
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_x3 = 0.0
    sum_x4 = 0.0
    sum_xy = 0.0
    sum_x2y = 0.0

    N = len(X)
    for x, y in zip(X, Y):
        sum_x += x
        sum_y += y
        x2 = x * x
        sum_x2 += x2
        sum_x3 += x2 * x
        sum_x4 += x2 * x2
        sum_xy += x * y
        sum_x2y += x2 * y

    sum_x /= N
    sum_y /= N
    sum_x2 /= N
    sum_x3 /= N
    sum_x4 /= N
    sum_xy /= N
    sum_x2y /= N

    b = ((sum_x * sum_y - sum_xy) / (sum_x3 - sum_x2 * sum_x) - (sum_x2 * sum_y - sum_x2y) / (sum_x4 - sum_x2 * sum_x2)) / ((sum_x3 - sum_x2 * sum_x) / (sum_x4 - sum_x2 * sum_x2) - (sum_x2 - sum_x * sum_x) / (sum_x3 - sum_x2 * sum_x))
    a = (sum_x2y - sum_x2 * sum_y - (sum_x3 - sum_x * sum_x2) * b) / (sum_x4 - sum_x2 * sum_x2)
    c = sum_y - sum_x2 * a - sum_x * b

    return [a, b, c]


class Polyfit2d:
    "二次曲线拟合类，通过 update() 添加数据，并通过 fit() 进行拟合"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.x = self.y = self.x2 = self.x3 = self.x4 = self.xy = self.x2y = 0

    def update(self, x: int, y: int):
        self.n += 1
        self.x += x
        self.y += y
        x2 = x * x
        self.x2 += x2
        self.x3 += x2 * x
        self.x4 += x2 * x2
        self.xy += x * y
        self.x2y += x2 * y

    def fit(self) -> List[float]:
        self.x /= self.n
        self.y /= self.n
        self.x2 /= self.n
        self.x3 /= self.n
        self.x4 /= self.n
        self.xy /= self.n
        self.x2y /= self.n
        b = ((self.x * self.y - self.xy) / (self.x3 - self.x2 * self.x) - (self.x2 * self.y - self.x2y) / (self.x4 - self.x2 * self.x2)) / ((self.x3 - self.x2 * self.x) / (self.x4 - self.x2 * self.x2) - (self.x2 - self.x * self.x) / (self.x3 - self.x2 * self.x))
        a = (self.x2y - self.x2 * self.y - (self.x3 - self.x * self.x2) * b) / (self.x4 - self.x2 * self.x2)
        c = self.y - self.x2 * a - self.x * b
        return [a, b, c]


__all__ = ["linePredictor", "polyfit1d", "polyfit2d", "Polyfit2d"]

