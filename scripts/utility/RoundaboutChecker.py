from Config import *

"""
flag:
0.初始
1.上升
2.下降
"""


class CircleHelper:
    def reset(self) -> None:
        self.pre = 0
        self.flag = 0
        self.isNot = False

    def update(self, x) -> None:
        if self.isNot:
            return
        if self.flag == 0:
            self.pre = x
            self.flag = 1
            self.count = 1
        elif self.flag == 1:
            if x < self.pre:
                if self.count < ROUND_UPCOUNT:
                    self.isNot = True
                else:
                    self.count = 0
                    self.flag = 2
        else:
            if x > self.pre:
                self.isNot = True
        self.pre = x
        self.count += 1

    def check(self) -> bool:
        return not self.isNot and self.flag == 2 and self.count >= ROUND_DOWNCOUNT


"""
flag:
0.丢线6次
1.圆环18个点
2.丢线一段距离
3.找到3次
"""


class RoundaboutChecker:
    def __init__(self) -> None:
        self.leftCheck = CircleHelper()
        self.rightCheck = CircleHelper()
        self.isNot = False
        self.flag = 0
        self.count = 0
        self.side = False

    def reset(self):
        self.isNot = False
        self.flag = 0
        self.count = 0

    def lost(self) -> None:
        if self.isNot:
            return

        if self.flag == 0:
            self.count += 1

        elif self.flag == 1:
            if not self.checkCircle():
                self.isNot = True
            else:
                self.flag = 2
                self.count = 1

        elif self.flag == 2:
            self.count += 1

        else:
            if self.count < ROUND_COUNT3:
                self.isNot = True

    def update(self, width: float, pi_: float, l: int, r: int) -> None:
        if self.isNot:
            return
        if width > ROUND_MAXWIDTH:
            return self.lost()

        if self.flag == 0:
            if self.count > ROUND_COUNT0:
                self.flag = 1
                self.count = 1
                self.leftCheck.reset()
                self.rightCheck.reset()
            else:
                self.isNot = True

        elif self.flag == 1:
            self.pi = pi_
            self.leftCheck.update(l)
            self.rightCheck.update(r)
            self.count += 1

        elif self.flag == 2:
            if self.pi - pi_ < ROUND_DIST2:
                self.isNot = True
            else:
                self.flag = 3
                self.count = 1
        else:
            self.count += 1

    def checkCircle(self) -> bool:
        l = self.leftCheck.check()
        r = self.rightCheck.check()
        if l ^ r:
            self.side = int(r)
        return l ^ r

    def check(self) -> int:
        if not self.isNot and self.flag == 3 and self.count >= ROUND_COUNT3:
            return self.side
        return 2


__all__ = ["RoundaboutChecker"]

