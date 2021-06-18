from Config import *


class HillChecker:
    def __init__(self):
        self.first = self.pre = 0.0

    def reset(self):
        self.notHill = False
        self.have = False
        self.first = self.pre = 0

    def update(self, width: float) -> None:
        if not self.have:
            self.first = self.pre = width
            self.have = True
        if self.notHill:
            return
        if width < self.pre:
            self.notHill = True
            return
        self.pre = width

    def isHill(self):
        return self.have and not self.notHill and self.pre - self.first > HILL_MINWIDTH


__all__ = ["HillChecker"]

