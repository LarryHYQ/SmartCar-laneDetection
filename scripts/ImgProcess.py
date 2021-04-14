import numpy as np
import cv2
from collections import deque
from typing import List, Tuple
from .transform import getPerMat, axisTransform, transfomImg
from .utility import *
from random import randint as rdit
from math import sqrt

colors = ((0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255))


def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x1 - x2, y1 - y2
    return sqrt(dx * dx + dy * dy)


class ImgProcess:
    "图像处理类"

    def __init__(self, Config: dict) -> None:
        """图像处理类

        Args:
            Config (dict): 通过 getConfig() 获取的配置
        """
        self.Config = Config
        self.edges = []
        self.valid = []
        self.sum = []
        self.Sum = 0
        self.res = [None] * 2
        self.firstFrame = True
        self.predictor = [LinePredictor(4), LinePredictor(4)]
        self.applyConfig()
        self.resetState()

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

    def resetState(self) -> None:
        "重置状态"
        count = (self.N - self.CUT) // self.H
        self.front = [0] * (self.M // (self.W >> 2))
        self.edges = [[-1] * count for _ in range(2)]
        self.valid = [[False] * count for _ in range(2)]
        self.sum = [[0] * count for _ in range(2)]
        self.whiteCMA = CMA()

    def calcK(self, i, k):
        b = (self.M >> 1) - (k * (self.N - 1) >> 3)
        return ((k * i) >> 3) + b

    def searchLinear(self, k: float, draw: bool = False, color: Tuple[int] = None) -> int:
        i = self.N - 1
        if draw and color is None:
            color = (rdit(0, 255), rdit(0, 255), rdit(0, 255))
        STEP = self.H
        while STEP:
            i_ = i - STEP
            j_ = self.calcK(i_, k)
            while i_ > self.CUT and self.PADDING <= j_ < self.M - self.PADDING and self.whiteCMA.val() - self.img[i_][j_] < 20:
                self.whiteCMA.update(self.img[i_][j_])
                if draw:
                    self.SrcShow.point((i_, j_), color)
                i, i_ = i_, i_ - STEP
                j_ = self.calcK(i_, k)
            STEP >>= 1
        return i

    def searchRow(self, i: int, j: int, draw: bool = False, color: Tuple[int] = None) -> List[int]:
        if draw and color is None:
            color = (rdit(0, 255), rdit(0, 255), rdit(0, 255))
        self.SrcShow.point((i, j), (255, 0, 0), 5)
        L = R = j
        STEP = self.W
        while STEP:
            while L - STEP >= self.PADDING and self.whiteCMA.val() - self.img[i][L - STEP] < 20:
                L -= STEP
                self.whiteCMA.update(self.img[i][L])
                if draw:
                    self.SrcShow.point((i, L), color)
            STEP >>= 1
        STEP = self.W
        while STEP:
            while R + STEP < self.M - self.PADDING and self.whiteCMA.val() - self.img[i][R + STEP] < 20:
                R += STEP
                self.whiteCMA.update(self.img[i][R])
                if draw:
                    self.SrcShow.point((i, R), color)
            STEP >>= 1
        if L != self.PADDING:
            self.SrcShow.point((i, L))
        if R != self.M - self.PADDING - 1:
            self.SrcShow.point((i, R))
        return [L, R]

    def getK(self) -> None:
        self.SrcShow.point((self.N - 1, self.M >> 1), (255, 0, 0))
        mi = self.K = 0x7FFFFFFF
        for k in range(-15, 15):
            i = self.searchLinear(k, False)
            if i < mi:
                mi, self.K = i, k

    def getEdge(self):
        I = self.N - 1
        J = self.calcK(I, self.K)
        while I > self.CUT and self.PADDING <= J < self.M - self.PADDING and self.whiteCMA.val() - self.img[I][J] < 20:
            L, R = self.searchRow(I, J, True)

            if L != self.PADDING:
                i, j = map(int, axisTransform(I, L, self.PERMAT))
                self.PerShow.point((i + self.I_SHIFT, j + self.J_SHIFT))
            if R != self.M - self.PADDING - 1:
                i, j = map(int, axisTransform(I, R, self.PERMAT))
                self.PerShow.point((i + self.I_SHIFT, j + self.J_SHIFT))

            I -= self.H
            J = self.calcK(I, self.K)

    def fitLine(self) -> List[np.array]:
        """选取两侧边线中有效点多的一侧进行拟合，拟合后在曲线最下面的点处延该点切线的垂线方向将曲线平移半个赛道的宽度，
        得到的就是赛道中线的抛物线方程。

        Returns:
            List[np.array]: 抛物线的参数 [a, b, c] -> y = a * x * x + b * x + c
        """
        count = [0] * 2
        fit = Polyfit2d()
        for u in range(2):
            fit.reset()
            hasValided = False
            for t, i in enumerate(range(self.N - (self.H >> 1), self.CUT - 1, -self.H)):
                if hasValided and not self.valid[u][t] and self.sum[u][t] < self.Sum:
                    self.SrcShow.point((i, self.edges[u][t]), colors[2 + u ^ 1])
                    for t in range(t, (self.N - self.CUT) // self.H):
                        self.valid[u][t] = False
                    break
                if self.valid[u][t]:
                    hasValided = True
                    i_, j_ = map(round, axisTransform(i, self.edges[u][t], self.PERMAT))
                    self.PerShow.point((i_ + self.I_SHIFT, j_ + self.J_SHIFT), colors[u ^ 1])
                    fit.update(i_, j_)
            count[u] = fit.n
            if fit.n > 3:
                self.res[u] = fit.fit()
                px = list(range(self.N_))
                py = np.polyval(self.res[u], px)
                self.PerShow.polylines(px, py, colors[u], i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)
        if max(count) < 3:
            return

        tmp, u = (self.res[0], 0) if count[0] > count[1] else (self.res[1], 1)
        print(u)
        print(tmp)
        tmp = shift(tmp, 110, 15, u)
        print(np.polyval(tmp, 120))
        print(tmp)
        px = list(range(self.N_))
        py = np.polyval(tmp, px)
        self.PerShow.polylines(px, py, (255, 0, 127), i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)

    def work(self):
        self.resetState()
        self.getK()
        self.getEdge()
        # self.fitLine()

    def getConstrain(self, j: int) -> int:
        """让小框框的横坐标保证处在图片内，防止数组越界

        Args:
            j (int): 需要开始搜索的列数

        Returns:
            int: 保证合法的开始搜索的列数
        """
        return min(max(j, self.PADDING), self.M - self.W - self.PADDING)

    def rectEdge(self, I: int, J: int, right: bool, H: int, W: int) -> Tuple[int]:
        """在小框框内搜索边线，具体做法是把每一个位置左右两边灰度的差值 (图像梯度)的平方
        作为权重乘以相应位置的横坐标累加到变量 Pos 上，同时把灰度差直接累加到dSum上
        (用平方是为了效果更加明显)，最终边界点的横坐标就可以通过 Pos // dSum 得到。

        如果边界点很明显(或小框选取的位置很合适)，得到的dSum值就会很大，而这个值越大，
        说明得到的边界点越可信；相反，如果框框里大多数点都是黑点或都是白点，得到的dSum
        就会很小，说明这个边界点不可信。在下方 getEdge() 获取边界点时就会通过这个特性
        来排除不合适的点。


        Args:
            I (int): 小框框左上角的行数
            J (int): 小框框左上角的列数
            right (bool): 如果为True则搜索右边界，反之搜索左边界
            H (int): 小框框的高度
            W (int): 小框框的宽度

        Returns:
            j (int): 得到的边界点横坐标
            dSum (int): 框框内像素灰度梯度平方的总和
        """
        Pos, dSum = 0, 1
        for i in range(I, I + H):
            for j in range(J + 1, J + W - 1):
                cur = self.img[i][j + 1] - self.img[i][j - 1]
                if right:
                    cur = -cur
                if cur > self.NOISE:
                    cur *= cur
                    Pos += cur * j
                    dSum += cur
        return Pos // dSum, dSum

    def show(self):
        self.SrcShow.show("src")
        self.PerShow.show("perspective")


__all__ = ["ImgProcess"]

