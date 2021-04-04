DIR = "D:\\CarImg\\"
from ultities import *
from transform import getPerMat, axisTransform, transfomImg

index = 381

N, M = 80, 188  # 图片的高和宽
NOISE = 5
H, W = 8, 35  # 框框的高和宽
PADDING = 1  # 舍弃左右边界


DeriThreshold = 60000
SumThreshold = 50000


N_ = 130  # 新图的高
M_ = 200  # 新图的宽
i_shift = 1  # 向下平移
j_shift = 20  # 向右平移

srcArr = [  # 原图上的四个点
    (0, 49),  # 左上角
    (0, 76),  # 右上角
    (59, 0),  # 左下角
    (59, 119),  # 右下角
]
perArr = [  # 新图上的四个点
    (0, 49),  # 左上角
    (0, 86),  # 右上角
    (120, 49),  # 左下角
    (120, 86),  # 右下角
]


origin = cv2.imread(DIR + str(index) + ".png", 0)
origin = cv2.resize(origin[20:, :], (M, N))
perImg = np.zeros((N_, M_), "uint8")

perMat = getPerMat(srcArr, perArr)
perImg = transfomImg(origin, perMat, N, M, N_, M_, i_shift, j_shift)

OriginShow = ShowImg(origin, "origin", 5)
PerShow = ShowImg(perImg, "per", 4)
img = origin.tolist()


def getConstrain(j: int, W: int = W):
    return min(max(j, PADDING), M - W - PADDING)


def rectEdge(I: int, J: int, right: bool, H: int = H, W: int = W):
    Sum, Pos = 1, 0
    for i in range(I, I + H):
        for j in range(J + 1, J + W - 1):
            cur = img[i][j - 1] - img[i][j + 1] if right else img[i][j + 1] - img[i][j - 1]
            if cur > NOISE:
                cur *= cur
                Pos += cur * j
                Sum += cur
    return Pos // Sum, Sum


def rectSum(I: int, J: int):
    return sum(sum(img[i][j] for j in range(J, J + W)) for i in range(I, I + H))


colors = ((255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255))


def getButtom():
    l, s = rectEdge(N - H * 2, PADDING, False, H * 2, M // 2 - PADDING * 2)
    r, s = rectEdge(N - H * 2, M // 2, True, H * 2, M // 2 - PADDING * 2)
    OriginShow.plot((N - H, l), colors[3])
    OriginShow.plot((N - H, r), colors[2])
    return l, r


edges = [[-1] * (N // H) for _ in range(2)]


def getEdge(LR: Tuple[int], draw: bool = True):
    for u in range(2):
        J = LR[u]
        dj = 0
        print("t  j   dSum  Sum")
        for t, i in enumerate(range(N - H, -1, -H)):
            j = getConstrain(J + dj - (W >> 1))
            edges[u][t], s = rectEdge(i, j, u)
            print("%1d %3d %6d %d" % (t, edges[u][t], s, rectSum(i, j)))
            if s < DeriThreshold:
                edges[u][t] = -1
                if draw:
                    OriginShow.rectangle((i, j), (i + H, j + W), colors[2 + u])
                if rectSum(i, j) < SumThreshold:
                    break
                J += dj
            else:
                if t == 1:
                    dj = edges[u][1] - edges[u][0]
                    J = edges[u][1]
                elif t > 1:
                    dj = edges[u][t] - J
                    J = edges[u][t]
                if draw:
                    OriginShow.rectangle((i, j), (i + H, j + W), colors[u])
                    OriginShow.plot((i + (H >> 1), edges[u][t]), colors[u ^ 1])
        print()


LR = getButtom()
getEdge(LR, True)


for u in range(2):
    x, y = [], []
    for t, i in enumerate(range(N - (H >> 1), -1, -H)):
        if edges[u][t] != -1:
            i_, j_ = map(round, axisTransform(i, edges[u][t], perMat))
            PerShow.plot((i_ + i_shift, j_ + j_shift), colors[u ^ 1])
            x.append(i_)
            y.append(j_)
    fit = np.polyfit(x, y, 2)
    px = np.linspace(0, N_, N_)
    py = np.polyval(fit, px)
    PerShow.polylines(px, py, colors[u], i_shift=i_shift, j_shift=j_shift)
    # cv2.polylines(PerShow.canva, [pts], False, (0, 0, 255))


OriginShow.show()
PerShow.show()

cv2.waitKey(0)
