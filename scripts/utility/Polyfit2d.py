"""
参考：
https://blog.csdn.net/u011023470/article/details/111381695
https://blog.csdn.net/u011023470/article/details/111381298
"""
from typing import List, Tuple
from math import sqrt


class Polyfit2d:
    "二次曲线拟合类"

    def reset(self) -> None:
        "重置状态"
        self.n = 0
        self.x = self.y = self.x2 = self.x3 = self.x4 = self.xy = self.x2y = 0

    def update(self, x: float, y: float) -> None:
        """增加一组数据

        Args:
            x (float): 自变量
            y (float): 因变量
        """
        self.n += 1
        self.x += x
        self.y += y
        x2 = x * x
        self.x2 += x2
        self.x3 += x2 * x
        self.x4 += x2 * x2
        self.xy += x * y
        self.x2y += x2 * y

    def fit(self) -> None:
        "最终拟合"
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

    def shift(self, x0: int, d: float, direction: bool) -> None:
        """将拟合得到的抛物线延x0处的切线的垂线平移一段距离

        Args:
            x0 (int): 原抛物线上目标点的横坐标
            d (float): 所要平移的距离
            direction (bool): 平移方向

        Returns:
            List[float]: 新抛物线的3个参数
        """
        A, B, C = self.res
        t = (B - A * x0 * x0 - B * x0 - C) / x0
        q = d / sqrt(t * t + 1)
        if direction:
            q = -q
        p = t * q
        if 2 * A * x0 + B < 0:
            p = -p
        self.res = [A, B - 2 * A * p, A * p * p - B * p + C + q]


__all__ = ["Polyfit2d"]

