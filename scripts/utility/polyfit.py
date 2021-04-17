"""
参考：
https://blog.csdn.net/u011023470/article/details/111381695
https://blog.csdn.net/u011023470/article/details/111381298
"""
from typing import List, Tuple
from math import sqrt


class LinePredictor:
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

    def update(self, x: float, y: float):
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
        B = ((self.x * self.y - self.xy) / (self.x3 - self.x2 * self.x) - (self.x2 * self.y - self.x2y) / (self.x4 - self.x2 * self.x2)) / ((self.x3 - self.x2 * self.x) / (self.x4 - self.x2 * self.x2) - (self.x2 - self.x * self.x) / (self.x3 - self.x2 * self.x))
        A = (self.x2y - self.x2 * self.y - (self.x3 - self.x * self.x2) * B) / (self.x4 - self.x2 * self.x2)
        C = self.y - self.x2 * A - self.x * B
        self.res = [A, B, C]
        return [A, B, C]

    def val(self, x: float):
        A, B, C = self.res
        return A * x * x + B * x + C

    def val_(self, x: float):
        A, B, C = self.res_
        return A * x * x + B * x + C

    def get(self, PX: float, PY: float, x: float):
        ex, ey = self.extreme()
        if (self.val(x) - PY >= 0) ^ (ey - PY >= 0):
            x = min(x, ex)
        y = self.val(x)
        A = (y - PY) / ((x - PX) * (x - PX))
        B = -2 * PX * A
        C = PY + A * PX * PX
        self.res_ = [A, B, C]
        return self.res_

    def shift(self, x0: int, d: float, direction: bool) -> List[float]:
        A, B, C = self.res
        t = (B - A * x0 * x0 - B * x0 - C) / x0
        q = d / sqrt(t * t + 1)
        if direction:
            q = -q
        p = t * q
        if 2 * A * x0 + B < 0:
            p = -p
        self.res = [A, B - 2 * A * p, A * p * p - B * p + C + q]
        return self.res

    def extreme(self) -> List[float]:
        A, B, C = self.res
        x = -B / (2 * A)
        y = self.val(x)
        return [x, y]


def shift(abc: List[float], x0: int, d: float, direction: bool) -> List[float]:
    """将拟合得到的抛物线延x0处的切线的垂线平移一段距离

    Args:
        abc (List[float]): 原抛物线的3个参数 [a, b, c] -> y = a * x * x + b * x + c
        x0 (int): 原抛物线上目标点的横坐标
        d (float): 所要平移的距离
        direction (bool): 平移方向

    Returns:
        List[float]: 新抛物线的3个参数
    """
    A, B, C = abc
    t = (B - A * x0 * x0 - B * x0 - C) / x0
    q = d / sqrt(t * t + 1)
    if direction:
        q = -q
    p = t * q
    if 2 * A * x0 + B < 0:
        p = -p
    return [A, B - 2 * A * p, A * p * p - B * p + C + q]


__all__ = ["LinePredictor", "polyfit1d", "polyfit2d", "Polyfit2d", "shift"]

