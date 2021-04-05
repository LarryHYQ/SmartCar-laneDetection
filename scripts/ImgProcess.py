import numpy as np
import cv2
from typing import List, Tuple
from .utility.ZoomedImg import ZoomedImg
from .transform import getPerMat, axisTransform, transfomImg

colors = ((0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255))


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
        self.applyConfig()

    def setImg(self, img: np.ndarray) -> None:
        img = cv2.resize(img[self.CUT :, :], (self.M, self.N))
        self.img = img.tolist()
        self.SrcShow = ZoomedImg(img, self.SRCZOOM)
        self.PerShow = ZoomedImg(transfomImg(img, self.PERMAT, self.N, self.M, self.N_, self.M_, self.I_SHIFT, self.J_SHIFT), self.PERZOOM)

    def applyConfig(self) -> None:
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

        count = self.N // self.H
        if len(self.edges) != count:
            self.edges = [[-1] * count for _ in range(2)]
            self.valid = [[False] * count for _ in range(2)]

    def getConstrain(self, j: int) -> int:
        return min(max(j, self.PADDING), self.M - self.W - self.PADDING)

    def rectEdge(self, I: int, J: int, right: bool, H: int, W: int) -> int:
        Sum, Pos = 1, 0
        for i in range(I, I + H):
            for j in range(J + 1, J + W - 1):
                cur = self.img[i][j + 1] - self.img[i][j - 1]
                if right:
                    cur = -cur
                if cur > self.NOISE:
                    cur *= cur
                    Pos += cur * j
                    Sum += cur
        return Pos // Sum, Sum

    def rectSum(self, I: int, J: int) -> int:
        return sum(sum(self.img[i][j] for j in range(J, J + self.W)) for i in range(I, I + self.H))

    def getButtom(self, draw: bool = True) -> Tuple[int]:
        l, s = self.rectEdge(self.N - self.H * 2, self.PADDING, False, self.H * 2, self.M // 2 - self.PADDING * 2)
        if s < self.DERI_THRESHOLD:
            l = 0
        r, s = self.rectEdge(self.N - self.H * 2, self.M // 2, True, self.H * 2, self.M // 2 - self.PADDING * 2)
        if s < self.DERI_THRESHOLD:
            r = self.M
        if draw:
            self.SrcShow.point((self.N - self.H, l), colors[3])
            self.SrcShow.point((self.N - self.H, r), colors[2])
        return l, r

    def getEdge(self, LR: Tuple[int], draw: bool = True):
        for u in range(2):
            J = LR[u]
            dj = 0
            print(" t  j   dSum  Sum")
            for t, i in enumerate(range(self.N - self.H, -1, -self.H)):
                j = self.getConstrain(J + dj - (self.W >> 1))
                self.edges[u][t], s = self.rectEdge(i, j, u, self.H, self.W)
                print("%2d %3d %6d %d" % (t, self.edges[u][t], s, self.rectSum(i, j)))
                if s < self.DERI_THRESHOLD:
                    self.valid[u][t] = False
                    if draw:
                        self.SrcShow.rectangle((i, j), (i + self.H, j + self.W), colors[2 + u])
                    if self.rectSum(i, j) < self.SUM_THRESHOLD:
                        break
                    J += dj
                else:
                    self.valid[u][t] = True
                    if t == 1:
                        dj = self.edges[u][1] - self.edges[u][0]
                        J = self.edges[u][1]
                    elif t > 1:
                        dj = self.edges[u][t] - J
                        J = self.edges[u][t]
                    if draw:
                        self.SrcShow.rectangle((i, j), (i + self.H, j + self.W), colors[u])
                        self.SrcShow.point((i + (self.H >> 1), self.edges[u][t]), colors[u ^ 1])
            print()

    def fitLine(self) -> List[np.array]:
        res = []
        for u in range(2):
            x, y = [], []
            for t, i in enumerate(range(self.N - (self.H >> 1), -1, -self.H)):
                if self.valid[u][t]:
                    i_, j_ = map(round, axisTransform(i, self.edges[u][t], self.PERMAT))
                    self.PerShow.point((i_ + self.I_SHIFT, j_ + self.J_SHIFT), colors[u ^ 1])
                    x.append(i_)
                    y.append(j_)
            res.append(np.polyfit(x, y, 2))
            px = np.linspace(0, self.N_, self.N_)
            py = np.polyval(res[-1], px)
            self.PerShow.polylines(px, py, colors[u], i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)
        return res

    def work(self):
        self.applyConfig()
        self.getEdge(self.getButtom())
        print(self.fitLine())

    def show(self):
        self.SrcShow.show("origin")
        self.PerShow.show("perspective")


__all__ = ["ImgProcess"]

