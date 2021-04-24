import numpy as np
from typing import List, Tuple
from .transform import getPerMat, axisTransform, transfomImg
from .utility import *
from random import randint as rdit
from math import sqrt

colors = ((255, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0), (0, 127, 127), (127, 127, 0))


class AngleEliminator:
    "通过判断新的点和前几个点的夹角是否小于45度来判断是否保留这个点"
    N = 4

    def __init__(self, main: "ImgProcess", fitter: Polyfit2d, color: Tuple[int] = (255, 0, 255)):
        self.main = main
        self.fitter = fitter
        self.color = color
        self.reset()

    def reset(self):
        self.t = 0
        self.I = [0] * self.N
        self.J = [0] * self.N

    def check(self, i: int, j: int):
        if self.t < self.N:
            return True
        i1, i2 = self.I[(self.t - 1) % self.N] - self.I[(self.t - self.N) % self.N], i - self.I[(self.t - 2) % self.N]
        j1, j2 = self.J[(self.t - 1) % self.N] - self.J[(self.t - self.N) % self.N], j - self.J[(self.t - 2) % self.N]
        dot = i1 * i2 + j1 * j2
        return dot > 0 and (dot * dot << 1) // ((i1 * i1 + j1 * j1) * (i2 * i2 + j2 * j2)) >= 1

    def update(self, i: int, j: int):
        if self.check(i, j):
            self.main.point((i, j), self.color)
            self.fitter.update(*axisTransform(i, j, self.main.PERMAT))
            self.I[self.t % self.N], self.J[self.t % self.N] = i, j
            self.t += 1


class DistEliminator:
    "通过判断相邻两个点的距离是否小于阈值来决定是否剔除这个点，并通过 AngleEliminator 进一步筛选"

    def __init__(self, main: "ImgProcess", angleEliminator: AngleEliminator, color: Tuple[int] = (255, 0, 255)):
        self.main = main
        self.angleEliminator = angleEliminator
        self.color = color
        self.reset()

    def reset(self):
        self.t = 0
        self.I = [0] * 2
        self.J = [0] * 2

    def update(self, i: int, j: int):
        if (self.t == 1 and abs(self.J[0] - j) <= 2) or (self.t > 1 and abs(j - self.J[self.t & 1]) <= 5):
            self.main.point((self.I[self.t & 1 ^ 1], self.J[self.t & 1 ^ 1]), self.color)
            self.angleEliminator.update(self.I[self.t & 1 ^ 1], self.J[self.t & 1 ^ 1])
        self.I[self.t & 1], self.J[self.t & 1] = i, j
        self.t += 1


class ImgProcess:
    "图像处理类"

    def __init__(self, Config: dict) -> None:
        """图像处理类

        Args:
            Config (dict): 通过 getConfig() 获取的配置
        """
        self.Config = Config
        self.fitter = [Polyfit2d() for u in range(2)]
        self.angleEliminator = [AngleEliminator(self, self.fitter[u], colors[u + 4]) for u in range(2)]
        self.distEliminator = [DistEliminator(self, self.angleEliminator[u], colors[u]) for u in range(2)]
        self.applyConfig()

    def setImg(self, img: np.ndarray) -> None:
        """设置当前需要处理的图像

        Args:
            img (np.ndarray): 使用 cv2.imread(xxx, 0) 读入的灰度图
        """
        self.img = img.tolist()
        self.SrcShow = ZoomedImg(img, self.SRCZOOM)
        self.PerShow = ZoomedImg(transfomImg(img, self.PERMAT, self.N, self.M, self.N_, self.M_, self.I_SHIFT, self.J_SHIFT), self.PERZOOM)

    def applyConfig(self) -> None:
        "从main窗口获取图像处理所需参数"
        self.THRESHLOD = 80
        self.N, self.M = self.Config["N"], self.Config["M"]  # 图片的高和宽
        self.CUT = self.Config["CUT"]  # 裁剪最上面的多少行
        self.NOISE = self.Config["NOISE"]  # 灰度梯度最小有效值
        self.SRCZOOM = self.Config["SRCZOOM"]  #
        self.H, self.W = self.Config["H"], self.Config["W"]  # 框框的高和宽
        self.PADDING = self.Config["PADDING"]  # 舍弃左右边界的大小
        self.DERI_THRESHOLD = self.Config["DERI_THRESHOLD"]  # 小框内灰度梯度总和最小有效值
        self.SUM_THRESHOLD = self.Config["SUM_THRESHOLD"]  # 小框内灰度总和最小有效值
        self.N_, self.M_ = self.Config["N_"], self.Config["M_"]  # 新图的高和宽
        self.I_SHIFT = self.Config["I_SHIFT"]  # 新图向下平移
        self.J_SHIFT = self.Config["J_SHIFT"]  # 新图向右平移
        self.PERZOOM = self.Config["PERZOOM"]  #
        self.PERMAT = getPerMat(self.Config["SRCARR"], self.Config["PERARR"])  # 逆透视变换矩阵
        self.REPMAT = getPerMat(self.Config["PERARR"], self.Config["SRCARR"])  # 反向逆透视变换矩阵
        self.FITX = 20

        self.SI, self.SJ = self.N + 10, self.M >> 1
        self.PI, self.PJ = axisTransform(self.SI, self.SJ, self.PERMAT)

    def point(self, pt: Tuple[int], color: Tuple[int] = (255, 255, 0), r: int = 4) -> None:
        "输入原图上的坐标，同时在原图和新图上画点"
        i, j = pt
        self.SrcShow.point((i, j), color, r)
        I, J = axisTransform(i, j, self.PERMAT)
        self.PerShow.point((I + self.I_SHIFT, J + self.J_SHIFT), color, r)

    def resetState(self) -> None:
        "重置状态"
        for u in range(2):
            self.angleEliminator[u].reset()
            self.distEliminator[u].reset()
            self.fitter[u].reset()
        self.SrcShow.line((self.CUT, 0), (self.CUT, self.M))

    def sobel(self, i: int, j: int) -> int:
        LRSTEP = 2
        UDSTEP = 1
        il = max(self.CUT, i - UDSTEP)
        ir = min(self.N - 1, i + UDSTEP)
        jl = max(self.PADDING, j - LRSTEP)
        jr = min(self.M - self.PADDING - 1, j + LRSTEP)
        return abs(self.img[il][jl] - self.img[ir][jr]) + abs(self.img[il][j] - self.img[ir][j]) + abs(self.img[i][jl] - self.img[i][jr]) + abs(self.img[il][jr] - self.img[ir][jl])

    def isEdge(self, i: int, j: int):
        "检查(i, j)是否是边界"
        return self.sobel(i, j) >= self.THRESHLOD

    def checkI(self, i: int) -> bool:
        "检查i是否没有越界"
        return self.CUT <= i < self.N

    def checkJ(self, j: int) -> bool:
        "检查j是否没有越界"
        return self.PADDING <= j < self.M - self.PADDING

    def calcK(self, i, k):
        "以行号和'斜率'计算列号"
        b = (self.M >> 1) - (k * (self.N - 1) // 3)
        return ((k * i) // 3) + b

    def searchK(self, k: int, draw: bool = False, color: Tuple[int] = None) -> int:
        "沿'斜率'k搜索黑色"
        if draw and color is None:
            color = (rdit(0, 255), rdit(0, 255), rdit(0, 255))

        i = self.N - 1
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
            color = (rdit(0, 255), rdit(0, 255), rdit(0, 255))
        STEP = 1 if isRight else -1
        while self.checkJ(j) and not self.isEdge(i, j):
            if draw:
                self.SrcShow.point((i, j), color)
            j += STEP
        return j - STEP

    def getK(self, draw: bool = False) -> None:
        "获取最远前沿所在的'斜率'K"
        self.SrcShow.point((self.N - 1, self.M >> 1), (255, 0, 0), 6)
        self.I = self.K = 0x7FFFFFFF
        for k in range(-9, 10):
            i = self.searchK(k, draw)
            if i < self.I:
                self.I, self.K = i, k
        self.SrcShow.point((self.I, self.calcK(self.I, self.K)), (255, 0, 0))
        self.SrcShow.line((self.N - 1, self.M >> 1), (self.I, self.calcK(self.I, self.K)))

    def getEdge(self, draw: bool = False):
        "逐行获取边界点"
        for u in range(2):
            for I in range(self.N - 1, self.I - 1, -1):
                J = self.calcK(I, self.K)
                j = self.searchRow(I, J, u, draw)
                if self.PADDING < j < self.M - self.PADDING - 1:
                    self.point((I, j), colors[u + 2])
                    self.distEliminator[u].update(I, j)
                else:
                    self.distEliminator[u].reset()

    def fitEdge(self):
        "拟合边界"
        self.PerShow.line((0, self.PJ + self.J_SHIFT), (self.N_, self.PJ + self.J_SHIFT))
        self.PerShow.point((self.PI + self.I_SHIFT, self.PJ + self.J_SHIFT), r=6)
        px = list(range(-self.I_SHIFT, self.N_ - self.I_SHIFT))
        for u in range(2):
            if self.fitter[u].n > 3:
                self.fitter[u].fit()

                # py = [self.fitter[u].val(v) for v in px]
                # self.PerShow.polylines(px, py, colors[u], i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)
                if self.fitter[u].n + u > self.fitter[u ^ 1].n:
                    self.fitter[u].shift(110, 14, u)
                    self.PerShow.point((self.PI - self.FITX + self.I_SHIFT, self.fitter[u].val(self.PI - self.FITX) + self.J_SHIFT))
                    py = [self.fitter[u].val(v) for v in px]
                    self.PerShow.polylines(px, py, colors[u], i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)

                    self.fitter[u].get(self.PI, self.PJ, self.PI - self.FITX)
                    py = [self.fitter[u].val_(v) for v in px]
                    self.PerShow.polylines(px, py, (0, 127, 255), i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)
                    r = 1 / (self.fitter[u].res_[0] * 2)
                    self.PerShow.circle((self.PI + self.I_SHIFT, self.PJ + r + self.J_SHIFT), r, (127, 255, 127))

    def work(self):
        "图像处理的完整工作流程"
        self.resetState()
        self.getK()
        self.getEdge()
        self.fitEdge()


__all__ = ["ImgProcess"]

