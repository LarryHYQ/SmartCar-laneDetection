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
        self.sum = []
        self.Sum = 0
        self.res = [None] * 2
        self.firstFrame = True
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
        self.REPMAT = getPerMat(self.Config["PERARR"], self.Config["SRCARR"])  # 反向逆透视变换矩阵

        count = self.N // self.H
        if len(self.edges) != count:
            self.edges = [[-1] * count for _ in range(2)]
            self.valid = [[False] * count for _ in range(2)]
            self.sum = [[0] * count for _ in range(2)]

    def getConstrain(self, j: int) -> int:
        return min(max(j, self.PADDING), self.M - self.W - self.PADDING)

    def rectEdge(self, I: int, J: int, right: bool, H: int, W: int) -> int:
        Pos, dSum, Sum = 0, 1, 0
        for i in range(I, I + H):
            for j in range(J + 1, J + W - 1):
                Sum += self.img[i][j]
                cur = self.img[i][j + 1] - self.img[i][j - 1]
                if right:
                    cur = -cur
                if cur > self.NOISE:
                    cur *= cur
                    Pos += cur * j
                    dSum += cur
        return Pos // dSum, dSum, Sum

    def getButtom(self) -> Tuple[int]:

        l, dSum, Sum = self.rectEdge(self.N - self.H * 2, self.PADDING, False, self.H * 2, self.M // 2 - self.PADDING * 2)
        if dSum < self.DERI_THRESHOLD:
            l = 0
        r, dSum, Sum = self.rectEdge(self.N - self.H * 2, self.M // 2, True, self.H * 2, self.M // 2 - self.PADDING * 2)
        if dSum < self.DERI_THRESHOLD:
            r = self.M

        self.SrcShow.point((self.N - self.H, l), colors[3])
        self.SrcShow.point((self.N - self.H, r), colors[2])
        return l, r

    def getEdge(self, LR: Tuple[int]):
        n = S = 0
        for u in range(2):
            cur = [0, LR[u]]
            J = LR[u]
            dj = 0
            print()
            print(" t  j   dSum   Sum")
            for t, i in enumerate(range(self.N - self.H, -1, -self.H)):
                # if not self.firstFrame and not (t > 2 and self.valid[u][t - 1] and abs(self.edges[u][t] - self.edges[u][t - 1]) < 40):
                #     j = self.getConstrain(self.edges[u][t] - (self.W >> 1))
                #     self.edges[u][t], dSum, self.sum[u][t] = self.rectEdge(i, j, u, self.H, self.W)
                # else:
                #     dSum = 0

                # if self.firstFrame or dSum < self.DERI_THRESHOLD:
                #     j = self.getConstrain(J + dj - (self.W >> 1))
                #     self.edges[u][t], dSum, self.sum[u][t] = self.rectEdge(i, j, u, self.H, self.W)
                j = self.getConstrain(J + dj - (self.W >> 1))
                self.edges[u][t], dSum, self.sum[u][t] = self.rectEdge(i, j, u, self.H, self.W)

                print("%2d %3d %6d %5d" % (t, self.edges[u][t], dSum, self.sum[u][t]))
                if dSum < self.DERI_THRESHOLD:
                    self.valid[u][t] = False
                    self.SrcShow.rectangle((i, j), (i + self.H, j + self.W), colors[2 + u])
                    J += dj
                else:
                    self.valid[u][t] = True
                    S += self.sum[u][t]
                    n += 1
                    if t == 1:
                        dj = self.edges[u][1] - self.edges[u][0]
                        J = self.edges[u][1]
                    elif t > 1:
                        dj = self.edges[u][t] - J if self.valid[u][t - 1] else 0
                    J = self.edges[u][t]

                    self.SrcShow.rectangle((i, j), (i + self.H, j + self.W), colors[u])
                    self.SrcShow.point((i + (self.H >> 1), self.edges[u][t]), colors[u ^ 1])
        if n:
            self.Sum = S // n
        self.firstFrame = False
        print(self.Sum)

    def fitLine(self) -> List[np.array]:
        for u in range(2):
            x, y = [], []
            hasValided = False
            for t, i in enumerate(range(self.N - (self.H >> 1), -1, -self.H)):
                if hasValided and not self.valid[u][t] and self.sum[u][t] < self.Sum:
                    self.SrcShow.point((i, self.edges[u][t]), colors[2 + u ^ 1])
                    for t in range(t, self.N // self.H):
                        self.valid[u][t] = False
                    break
                if self.valid[u][t]:
                    hasValided = True
                    i_, j_ = map(round, axisTransform(i, self.edges[u][t], self.PERMAT))
                    self.PerShow.point((i_ + self.I_SHIFT, j_ + self.J_SHIFT), colors[u ^ 1])
                    x.append(i_)
                    y.append(j_)
            if len(x) > 2:
                self.res[u] = np.polyfit(x, y, 2)
                px = list(range(self.N_))
                py = np.polyval(self.res[u], px)
                self.PerShow.polylines(px, py, colors[u], i_shift=self.I_SHIFT, j_shift=self.J_SHIFT)

    def work(self):
        self.getEdge(self.getButtom())
        self.fitLine()

    def show(self):
        self.SrcShow.show("src")
        self.PerShow.show("perspective")


__all__ = ["ImgProcess"]

