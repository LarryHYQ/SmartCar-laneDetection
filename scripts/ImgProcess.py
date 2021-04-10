import numpy as np
import cv2
from collections import deque
from typing import List, Tuple
from .transform import getPerMat, axisTransform, transfomImg
from .utility import *

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
        self.predictor = [linePredictor(4), linePredictor(4)]
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
        self.edges = [[-1] * count for _ in range(2)]
        self.valid = [[False] * count for _ in range(2)]
        self.sum = [[0] * count for _ in range(2)]

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

        Sum 是用来存储整个框框里所有像素灰度总和的变量，为的是在 dSum 较小(即判断为丢线)时，
        判断究竟是撞到了边界(几乎全黑)还是十字路口的丢边(几乎全白)：如果是撞到边界则停止向上
        搜边，如果是十字路口则要继续向上扩展。

        Args:
            I (int): 小框框左上角的行数
            J (int): 小框框左上角的列数
            right (bool): 如果为True则搜索右边界，反之搜索左边界
            H (int): 小框框的高度
            W (int): 小框框的宽度

        Returns:
            j (int): 得到的边界点横坐标
            dSum (int): 框框内像素灰度梯度平方的总和
            Sum (int): 框框内像素灰度的总和
        """
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

    def getEdge(self):
        """搜线的主要部分，目前实现的是两侧的线分别搜的方法

        1 首先找到所要搜的边的最低有效位置
            1.1 首先从底线中心开始，每 self.H 行向上搜一行
            对于每一行，横着每个几个点用累计滑动平均值 (horiCMA) 的差是否超过一个阈值来判断是否碰到黑点：
                1.1.1 如果碰到黑点，则使用 rectEdge() 来判断这个黑点是否满足要求，如果满足要求则进入边缘生长搜线；
                1.1.2 如果不满足要求，或一直没有碰到黑点，则跳过这一行。
            1.2 向上同样使用累计滑动平均值 (vertCMA) 来判断是否撞到黑点，如果撞到则说明碰到了另一侧的赛道 (对应的情况为大弯道丢一侧线)。

        2 在找到最低有效位置后，每 self.H 行先使用一阶最小二乘拟合最近的4个点来预测这一次搜线小框的位置，再用 rectEdge() 来确定这一个
        小框内的边线位置，如果这一个小框内的梯度平方总和 dSum 大于指定阈值，且与前两个点形成的向量夹角小于60°时才被认定为有效，否则舍弃这个点。
            2.1 这个步骤里并没有对于 dSum 小于阈值时的情况 (全黑或全白) 进行特判，而是直接跳过这个点，同时记录下所有有效点的框内灰度总和，
            取平均后在拟合的步骤里再把这些无效点的框内灰度和 Sum 与所有有效点的灰度平均值 self.Sum 进行比较，如果小于平均值过多则判定为全黑，
            终止扩展(直接break)，否则视为十字路口丢线，继续扩展。
        """
        self.resetState()
        n = S = 0
        vertCMA = CMA()
        horiCMA = CMA()
        for u in range(2):
            print()
            print(" t  j   dSum   Sum")
            hasTracedBottom = False
            MIDJ = self.M >> 1
            vertCMA.reset(self.img[self.N - self.H][MIDJ])
            for t, i in enumerate(range(self.N - self.H, self.CUT - 1, -self.H)):
                if not hasTracedBottom:
                    self.SrcShow.point((i, MIDJ), (127, 255, 127), 6)
                    if i <= 2 or abs(vertCMA.v - self.img[i][MIDJ]) > 20:
                        self.SrcShow.point((i, MIDJ), (127, 255, 127), 8)
                        break
                    vertCMA.update(self.img[i][MIDJ])
                    horiCMA.reset(self.img[i][MIDJ])
                    for j in range(MIDJ, self.M - self.PADDING, 10) if u else range(MIDJ, self.PADDING - 1, -10):
                        if abs(horiCMA.v - self.img[i][j]) > 20:
                            self.SrcShow.point((i, j), (127, 0, 127), 8)
                            j -= self.W >> 1
                            j = self.getConstrain(j)
                            j_, dSum_, _ = self.rectEdge(i - self.H, j, u, self.H << 1, self.W)
                            if dSum_ >= self.DERI_THRESHOLD << 1:
                                self.predictor[u].reset(j_)
                                hasTracedBottom = True
                            break

                if hasTracedBottom:
                    J = self.predictor[u].val(i)
                    if self.predictor[u].n > 2 and ((u and J > self.M) or (not u and J < 0)):
                        self.valid[u][t] = False
                        break
                    j = self.getConstrain(J - (self.W >> 1))
                    self.edges[u][t], dSum, self.sum[u][t] = self.rectEdge(i, j, u, self.H, self.W)
                    print("%2d %3d %6d %5d" % (t, self.edges[u][t], dSum, self.sum[u][t]))

                    if not (dSum > self.DERI_THRESHOLD and self.predictor[u].angleCheck(i, self.edges[u][t])):
                        self.valid[u][t] = False
                        self.SrcShow.rectangle((i, j), (i + self.H, j + self.W), colors[2 + u])
                        self.SrcShow.point((i + (self.H >> 1), self.edges[u][t]), (0, 0, 0))
                    else:
                        self.predictor[u].update(i, self.edges[u][t])
                        self.valid[u][t] = True
                        S += self.sum[u][t]
                        n += 1

                        self.SrcShow.rectangle((i, j), (i + self.H, j + self.W), colors[u])
                        self.SrcShow.point((i + (self.H >> 1), self.edges[u][t]), colors[u ^ 1])
        if n:
            self.Sum = S // n
        self.firstFrame = False
        print(self.Sum)

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
        self.getEdge()
        self.fitLine()

    def show(self):
        self.SrcShow.show("src")
        self.PerShow.show("perspective")


__all__ = ["ImgProcess"]

