class ParaCurve:
    "用于处理抛物线拟合得到的中线"

    def __init__(self, x0: float, y0: float) -> None:
        "初始化"
        self.x0, self.y0 = x0, y0

    def set(self, a_: float, b_: float, c_: float) -> None:
        "设置抛物线参数"
        self.a, self.b, self.c = a_, b_, c_
        self.t1 = [4 * self.a * self.a, 6 * self.a * self.b, 4 * self.a * (self.c - self.y0) + 2 * self.b * self.b + 2, 2 * self.b * (self.c - self.y0) - 2 * self.x0]
        self.t2 = [12 * self.a * self.a, 12 * self.a * self.b, 4 * self.a * (self.c - self.y0) + 2 * self.b * self.b + 2]

    def val(self, x: float) -> float:
        "计算函数值"
        return self.a * x * x + self.b * x + self.c

    def vald(self, x: float) -> float:
        "计算斜率"
        return 2 * self.a * x + self.b

    def perpendicular(self) -> float:
        "用牛顿迭代法求过小车点在抛物线上的垂足"

        def calc(x: float) -> float:
            return self.t1[0] * x * x * x + self.t1[1] * x * x + self.t1[2] * x + self.t1[3]

        def calcd(x: float) -> float:
            return self.t2[0] * x * x + self.t2[1] * x + self.t2[2]

        x = self.x0
        for _ in range(5):
            x = x - calc(x) / calcd(x)
        return x


__all__ = ["ParaCurve"]

