"""
参考：
https://blog.csdn.net/u011023470/article/details/111381695
https://blog.csdn.net/u011023470/article/details/111381298
"""
from typing import List, Tuple


def polyfit1d(X: List[float], Y: List[float]) -> Tuple[float]:
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
    return a, b


def polyfit2d(X: List[float], Y: List[float]) -> Tuple[float]:
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

    return (a, b, c)


__all__ = ["polyfit1d", "polyfit2d"]

