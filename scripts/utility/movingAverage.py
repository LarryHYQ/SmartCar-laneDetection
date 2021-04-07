"""
参考：
https://zh.wikipedia.org/wiki/移動平均
"""


class SMA:
    "简单移动平均"

    def __init__(self, count: int):
        self.count = count
        self.reset()

    def reset(self):
        self.i = -1
        self.n = 0
        self.v = 0.0
        self.a = [0] * self.count

    def update(self, x: int):
        if self.n < self.count:
            self.n += 1
            self.v += (x - self.v) / self.n
        else:
            self.v += (x - self.a[self.i]) / self.n
        self.i = (self.i + 1) % self.count
        self.a[self.i] = x

    def dif(self, x: int):
        return abs(self.v - x) / self.v


class CMA:
    "累积移动平均"

    def __init__(self, count: int):
        self.count = count
        self.reset()

    def reset(self):
        self.n = 0
        self.v = 0.0

    def update(self, x: int):
        self.n += 1
        self.v += (x - self.v) / self.n

    def dif(self, x: int):
        return abs(self.v - x) / self.v
