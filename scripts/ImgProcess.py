import numpy as np
from typing import Tuple
from .transform import getPerMat, axisTransform, transfomImg
from .utility import *
from random import randint
from math import atan
from Config import *


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
        self.pointEliminator = PointEliminator(self)
        self.applyConfig()
        self.paraCurve = ParaCurve(self.PI, self.PJ)

    def setImg(self, img: np.ndarray) -> None:
        """设置当前需要处理的图像

        Args:
            img (np.ndarray): 使用 cv2.imread(xxx, 0) 读入的灰度图
        """
        self.img = img.tolist()
        self.SrcShow = ZoomedImg(img, SRCZOOM)
        self.PerShow = ZoomedImg(transfomImg(img, self.PERMAT, N, M, N_, M_, I_SHIFT, J_SHIFT), PERZOOM)

    def applyConfig(self) -> None:
        "从main窗口获取图像处理所需参数"
        self.PERMAT = getPerMat(SRCARR, PERARR)  # 逆透视变换矩阵
        self.REPMAT = getPerMat(PERARR, SRCARR)  # 反向逆透视变换矩阵

        self.SI, self.SJ = N + 10, M >> 1
        self.PI, self.PJ = axisTransform(self.SI, self.SJ, self.PERMAT)

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

    def resetState(self) -> None:
        "重置状态"
        for u in range(2):
            self.fitter[u].reset()
        self.SrcShow.line((CUT, 0), (CUT, M))

    def sobel(self, i: int, j: int) -> int:
        "魔改的sobel算子"
        il = max(CUT, i - UDSTEP)
        ir = min(N - 1, i + UDSTEP)
        jl = max(PADDING, j - LRSTEP)
        jr = min(M - PADDING - 1, j + LRSTEP)
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
            if not (self.checkI(i) and self.checkJ(j) and not self.isEdge(i, j)):
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
        for k in range(-9, 10):
            i = self.searchK(k, draw)
            if i < self.I:
                self.I, self.K = i, k
        self.SrcShow.point((self.I, self.calcK(self.I, self.K)), (255, 0, 0))
        self.SrcShow.line((N - 1, M >> 1), (self.I, self.calcK(self.I, self.K)))

    def getEdge(self, draw: bool = False):
        "逐行获取边界点"
        for u in range(2):
            self.pointEliminator.reset(u ^ 1, self.fitter[u], COLORS[u + 4])
            for I in range(N - 1, self.I - 1, -1):
                J = self.calcK(I, self.K)
                j = self.searchRow(I, J, u, draw)
                if PADDING < j < M - PADDING - 1:
                    self.point((I, j), COLORS[u + 2])
                    self.pointEliminator.update(*axisTransform(I, j, self.PERMAT))

    def getMid(self) -> bool:
        "获取中线"
        self.PerShow.point((self.PI + I_SHIFT, self.PJ + J_SHIFT), r=6)
        px = list(range(-I_SHIFT, N_ - I_SHIFT))

        for u in range(2):
            if self.fitter[u].n > 5:
                self.fitter[u].fit()
                self.fitter[u].shift(X_POS, WIDTH, u)

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
        "生成路径并获得曲率"
        k = self.paraCurve.vald(self.X1)
        tx, ty = move_to_pose(self.PI, self.PJ, np.pi, self.X1, self.Y1, atan(k) + np.pi)
        self.PerShow.polylines(tx, ty, (0, 255, 0), i_shift=I_SHIFT, j_shift=J_SHIFT)

        rho = curvatureSolve(self.PI, self.PJ, self.X1, self.Y1, atan(k) + np.pi)
        if rho != 0:
            r = 1 / rho
            self.PerShow.circle((self.PI + I_SHIFT, self.PJ - r + J_SHIFT), abs(r))
        else:
            self.PerShow.line((0, self.PJ + J_SHIFT), (N_, self.PJ + J_SHIFT))

    def work(self):
        "图像处理的完整工作流程"
        self.resetState()
        self.getK()
        self.getEdge()
        if self.getMid():
            self.getTarget()
            self.solve()


__all__ = ["ImgProcess"]

