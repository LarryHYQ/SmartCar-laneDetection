import numpy as np
from typing import Tuple
from .transform import getPerMat, axisTransform, transfomImg
from .utility import *
from random import randint
from math import atan
from Config import *
from math import sqrt, cos, acos, degrees, radians, atan2


class PointEliminator:
    "通过判断新的点和前面点的连线斜率是否在特定区间来决定是否保留这个点"

    def __init__(self, main: "ImgProcess") -> None:
        self.main = main
        self.I = [0.0] * 2
        self.J = [0.0] * 2

    def reset(self, invert: bool, fitter: Polyfit2d, color: Tuple[int] = (255, 0, 255)) -> None:
        self.n = 0
        self.invert = invert
        self.fitter = fitter
        self.color = color

    def insert(self, i: float, j: float) -> None:
        self.I[self.n & 1] = i
        self.J[self.n & 1] = j
        self.n += 1

    def check(self, i: float, j: float) -> bool:
        k = (j - self.J[self.n & 1]) / (i - self.I[self.n & 1])
        if self.invert:
            k = -k
        return K_LOW < k < K_HIGH

    def update(self, i: float, j: float) -> None:
        if self.n < 2:
            self.insert(i, j)
        elif self.check(i, j):
            self.insert(i, j)
            self.fitter.update(i, j)
            self.main.ppoint((i, j), self.color)
        else:
            self.n = 0


class ImgProcess:
    "图像处理类"

    def __init__(self) -> None:
        """图像处理类

        Args:
            Config (dict): 通过 getConfig() 获取的配置
        """
        self.fitter = [Polyfit2d() for u in range(2)]
        self.pointEliminator = [PointEliminator(self) for u in range(2)]
        self.circleFit = [CircleFit(lambda x, y, r: (self.PerShow.circle((x + I_SHIFT, y + J_SHIFT), r), self.PerShow.point((x + I_SHIFT, y + J_SHIFT))), lambda s, pt: self.PerShow.putText(s, (pt[0] + I_SHIFT, pt[1] + J_SHIFT))) for u in range(2)]
        self.applyConfig()
        self.paraCurve = ParaCurve(self.PI, self.PJ)
        self.hillChecker = [HillChecker() for u in range(2)]
        self.frontForkChecker = FrontForkChecker(self.PERMAT, self.pline)
        self.sideForkChecker = [SideForkChecker(self.pline) for u in range(2)]
        self.sideFork = False
        self.roundaboutChecker = RoundaboutChecker()

        self.landmark = {"StartLine": False, "Hill": False, "Roundabout1": False, "Fork": False, "Yaw": 0.0}

    def setImg(self, img: np.ndarray) -> None:
        """设置当前需要处理的图像

        Args:
            img (np.ndarray): 使用 cv2.imread(xxx, 0) 读入的灰度图
        """
        self.img = img.tolist()

        # for i in range(N):
        #     for j in range(M):
        #         img[i, j] = 255 if self.isEdge(i, j) else 0
        # self.img = img.tolist()

        self.SrcShow = ZoomedImg(img, SRCZOOM)
        self.PerShow = ZoomedImg(transfomImg(img, self.PERMAT, N, M, N_, M_, I_SHIFT, J_SHIFT), PERZOOM)

    def applyConfig(self) -> None:
        "从main窗口获取图像处理所需参数"
        self.PERMAT = getPerMat(SRCARR, PERARR)  # 逆透视变换矩阵
        self.REPMAT = getPerMat(PERARR, SRCARR)  # 反向逆透视变换矩阵

        self.SI, self.SJ = N + 1, M >> 1
        self.PI, self.PJ = axisTransform(self.SI, self.SJ, self.PERMAT)
        self.PI = PI
        print(f"PI {self.PI}\nPJ {self.PJ}")
        print(f"FORKLOW {cos(radians(FORKHIGH))}f\nFORKHIGH {cos(radians(FORKLOW))}f")

    def point(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt
        self.SrcShow.point((i, j), color, r)
        I, J = axisTransform(i, j, self.PERMAT)
        self.PerShow.point((I + I_SHIFT, J + J_SHIFT), color, r)

    def ppoint(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt
        self.PerShow.point((i + I_SHIFT, j + J_SHIFT), color, r)
        I, J = axisTransform(i, j, self.REPMAT)
        self.SrcShow.point((I, J), color, r)

    def line(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2) -> None:
        (i1, j1), (i2, j2) = p1, p2
        self.SrcShow.line(p1, p2, color, thickness)
        pi1, pj1 = axisTransform(i1, j1, self.PERMAT)
        pi2, pj2 = axisTransform(i2, j2, self.PERMAT)
        self.PerShow.line((pi1 + I_SHIFT, pj1 + J_SHIFT), (pi2 + I_SHIFT, pj2 + J_SHIFT), color, thickness)

    def pline(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2) -> None:
        (i1, j1), (i2, j2) = p1, p2
        self.PerShow.line((i1 + I_SHIFT, j1 + J_SHIFT), (i2 + I_SHIFT, j2 + J_SHIFT), color, thickness)
        pi1, pj1 = axisTransform(i1, j1, self.REPMAT)
        pi2, pj2 = axisTransform(i2, j2, self.REPMAT)
        self.SrcShow.line((pi1, pj1), (pi2, pj2), color, thickness)

    def resetState(self) -> None:
        "重置状态"
        # 标尺
        self.SrcShow.line((N - 5, 5), (N - 5, 15))
        self.SrcShow.line((N - 5, 5), (N - 15, 5))
        self.PerShow.line((N_ - 5, 5), (N_ - 5, 15))
        self.PerShow.line((N_ - 5, 5), (N_ - 15, 5))

        # 前沿线范围
        # self.SrcShow.line((CUT, 0), (CUT, M))
        # self.SrcShow.line((FORKUPCUT, 0), (FORKUPCUT, M))
        # self.SrcShow.line((N - FORKDOWNCUT, 0), (N - FORKDOWNCUT, M))
        # self.SrcShow.line((0, CORNERCUT), (N, 0))
        # self.SrcShow.line((0, M - CORNERCUT), (N, M))

        # 起跑线检测
        # self.line((STARTLINE_I1, STARTLINE_PADDING), (STARTLINE_I1, M - STARTLINE_PADDING), (255, 0, 0))
        # self.line((STARTLINE_I2, STARTLINE_PADDING), (STARTLINE_I2, M - STARTLINE_PADDING), (255, 0, 0))

        # 坡道有效线
        # self.line((N - HILLCUT, 0), (N - HILLCUT, M))

        # 环岛
        self.PerShow.line((0, ROUND_MAXWIDTH), (N_, ROUND_MAXWIDTH))

    def sobel(self, i: int, j: int, lr: int = LRSTEP) -> int:
        "魔改的sobel算子"
        il = max(CUT, i - UDSTEP)
        ir = min(N - 1, i + UDSTEP)
        jl = max(PADDING, j - lr)
        jr = min(M - PADDING - 1, j + lr)
        return abs(self.img[il][jl] - self.img[ir][jr]) + abs(self.img[il][j] - self.img[ir][j]) + abs(self.img[i][jl] - self.img[i][jr]) + abs(self.img[il][jr] - self.img[ir][jl])

    def isEdge(self, i: int, j: int):
        "检查(i, j)是否是边界"
        return self.sobel(i, j) >= THRESHLOD

    def checkI(self, i: int) -> bool:
        "检查i是否没有越界"
        return CUT <= i < N

    def checkJ(self, j: int) -> bool:
        "检查j是否没有越界"
        return PADDING <= j < M - PADDING

    def checkCornerIJ(self, i: int, j: int) -> bool:
        "找前沿线时限定范围"
        return self.checkI(i) and j * N > CORNERCUT * (N - i) and N * j < CORNERCUT * i + N * (M - CORNERCUT)

    def calcK(self, i, k):
        "以行号和'斜率'计算列号"
        b = (M >> 1) - (k * (N - 1) // 3)
        return ((k * i) // 3) + b

    def searchK(self, k: int, draw: bool = False, color: Tuple[int] = None) -> int:
        "沿'斜率'k搜索黑色"
        if draw and color is None:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))

        i = N - 1
        while True:
            i -= 1
            j = self.calcK(i, k)
            if draw:
                self.SrcShow.point((i, self.calcK(i, k)), color)
            if not (self.checkCornerIJ(i, j) and not self.isEdge(i, j)):
                return i + 1

    def searchRow(self, i: int, j: int, isRight: bool, draw: bool = False, color: Tuple[int] = None) -> int:
        "按行搜索左右的黑色"
        if draw and color is None:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
        STEP = 1 if isRight else -1
        while self.checkJ(j) and not self.isEdge(i, j):
            if draw:
                self.SrcShow.point((i, j), color)
            j += STEP
        return j

    def getK(self, draw: bool = False) -> None:
        "获取最远前沿所在的'斜率'K"
        self.SrcShow.point((N - 1, M >> 1), (255, 0, 0), 6)
        self.I = self.K = 0x7FFFFFFF
        self.frontForkChecker.reset()
        for k in range(-6, 7):
            i = self.searchK(k, draw)
            if self.checkCornerIJ(i - 1, self.calcK(i - 1, k)):
                self.frontForkChecker.update(i - 1, self.calcK(i - 1, k))
            else:
                self.frontForkChecker.lost()
            if i < self.I:
                self.I, self.K = i, k
        self.SrcShow.point((self.I, self.calcK(self.I, self.K)), (255, 0, 0))
        self.SrcShow.line((N - 1, M >> 1), (self.I, self.calcK(self.I, self.K)))

    def getEdge(self, draw: bool = False):
        "逐行获取边界点"
        self.sideFork = False
        self.roundaboutChecker.reset()
        for u in range(2):
            self.fitter[u].reset()
            self.hillChecker[u].reset()
            self.pointEliminator[u].reset(u ^ 1, self.fitter[u], COLORS[u + 4])
            self.sideForkChecker[u].reset()

        for I in range(N - 1, self.I - 1, -2):
            J = self.calcK(I, self.K)
            side = [self.searchRow(I, J, u, draw) for u in range(2)]
            pj = [0.0] * 2
            nolost = True
            for u in range(2):
                if self.checkJ(side[u]):
                    self.point((I, side[u]), COLORS[u + 2])
                    pi, pj[u] = axisTransform(I, side[u], self.PERMAT)
                    self.sideForkChecker[u].update(pi, pj[u])
                    self.pointEliminator[u].update(pi, pj[u])
                    if I < HILL_CUT:
                        self.hillChecker[u].update(-pj[u] if u else pj[u])
                else:
                    nolost = False
                    self.sideForkChecker[u].lost()

            if nolost:
                width = pj[1] - pj[0]
                self.PerShow.point((pi + I_SHIFT, width + J_SHIFT))
                self.roundaboutChecker.update(width, pi, side[0], -side[1])
            else:
                self.roundaboutChecker.lost()

            self.sideFork |= self.sideForkChecker[u].res

    def getMid(self, drawEdge: bool = False) -> bool:
        "获取中线"
        self.PerShow.point((self.PI + I_SHIFT, self.PJ + J_SHIFT), r=6)
        px = list(range(-I_SHIFT, N_ - I_SHIFT))

        for u in range(2):
            if self.fitter[u].n > 5:
                self.fitter[u].fit()
                if drawEdge:
                    py = [self.fitter[u].val(v) for v in px]
                    self.PerShow.polylines(px, py, COLORS[u + 2], i_shift=I_SHIFT, j_shift=J_SHIFT)
                self.fitter[u].shift(X_POS, WIDTH, u)
                if drawEdge:
                    py = [self.fitter[u].val(v) for v in px]
                    self.PerShow.polylines(px, py, COLORS[u + 2], i_shift=I_SHIFT, j_shift=J_SHIFT)

        if min(self.fitter[u].n for u in range(2)) > 5:
            N = sum(self.fitter[u].n for u in range(2))
            a, b, c = [sum(self.fitter[u].res[i] * self.fitter[u].n for u in range(2)) / N for i in range(3)]
        elif max(self.fitter[u].n for u in range(2)) > 5:
            a, b, c = self.fitter[0].res if self.fitter[0].n > 5 else self.fitter[1].res
        else:
            return False

        self.paraCurve.set(a, b, c)
        py = [self.paraCurve.val(v) for v in px]
        self.PerShow.polylines(px, py, COLORS[u], i_shift=I_SHIFT, j_shift=J_SHIFT)
        return True

    def getTarget(self):
        "获取参考点位置"
        x = self.paraCurve.perpendicular()
        y = self.paraCurve.val(x)
        self.PerShow.line((self.PI + I_SHIFT, self.PJ + J_SHIFT), (round(x) + I_SHIFT, round(y) + J_SHIFT), (255, 255, 0))
        self.ppoint((round(x), round(y)), (0, 0, 255))

        l, r = x - DIST, x
        for _ in range(5):
            self.X1 = (l + r) / 2
            self.Y1 = self.paraCurve.val(self.X1)
            dx = x - self.X1
            dy = y - self.Y1
            d = dx * dx + dy * dy
            if d < DIST * DIST:
                r = self.X1
            else:
                l = self.X1
        self.ppoint((round(self.X1), round(self.Y1)), (255, 127, 255), 6)

    def solve(self):
        "获取目标偏航角"
        self.PerShow.point((self.PI - X0 + I_SHIFT, self.PJ + J_SHIFT), (255, 0, 0), 6)
        self.PerShow.line((self.PI - X0 + I_SHIFT, self.PJ + J_SHIFT), (self.X1 + I_SHIFT, self.Y1 + J_SHIFT))
        self.landmark["Yaw"] = atan2(self.Y1 - self.PJ, self.PI - X0 - self.X1)

    def checkStartLine(self, i: int) -> bool:
        "检测及起跑线"
        pre = self.sobel(i, STARTLINE_PADDING, 1) > THRESHLOD
        count = 0
        for j in range(STARTLINE_PADDING + 1, M - STARTLINE_PADDING):
            cur = self.sobel(i, j, 1) > THRESHLOD
            count += pre ^ cur
            pre = cur
        return count > STARTLINE_COUNT

    def showRes(self):
        for i, (k, v) in enumerate(self.landmark.items()):
            self.PerShow.putText(k + ": " + str(v), (i * 5 + 100, 170))

    def work(self):
        "图像处理的完整工作流程"
        self.resetState()
        self.landmark["StartLine"] = self.checkStartLine(STARTLINE_I1) or self.checkStartLine(STARTLINE_I2)
        self.getK()
        self.getEdge()
        self.landmark["Hill"] = self.hillChecker[0].check() and self.hillChecker[1].check() and self.hillChecker[0].calc() + self.hillChecker[1].calc() > HILL_DIFF
        self.landmark["Roundabout1"] = "None" if not self.roundaboutChecker.check() else "Right" if self.roundaboutChecker.side else "Left"
        self.landmark["Fork"] = self.frontForkChecker.res & self.sideFork
        if self.getMid():
            self.getTarget()
            self.solve()
        self.showRes()


__all__ = ["ImgProcess"]

