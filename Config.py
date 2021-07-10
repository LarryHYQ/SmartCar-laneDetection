# 图像
N, M = 80, 188  # 图片的高和宽
CUT = 1  # 裁剪最上面的多少行
PADDING = 1  # 舍弃左右边界的大小
CORNERCUT = 40  # 搜索前沿时舍弃最上面角的宽度

FORKUPCUT = 5  # 三岔路口前沿点最小有效行
FORKDOWNCUT = 10  # 三岔路口前沿点最大有效行
FORKLOW = 110  # 三岔路口最小角度
FORKHIGH = 140  # 三岔路口最大角度
FORKMAXDIST2 = 25

# sobel
LRSTEP = 2
UDSTEP = 1
THRESHLOD = 230  # sobel的阈值

# 用于排除点的斜率范围
K_LOW = -0.2  # 斜率下限
K_HIGH = 1.5  # 斜率上限

# 拟合曲线
X_POS = 102  # 平移点的x位置
WIDTH = 18  # 赛道宽度

# 获取目标点
PI = 144.0
DIST = 37  # 垂足向上扩展的长度

X0 = 27.0

# 起跑线检测
STARTLINE_I1 = 30
STARTLINE_I2 = 60
STARTLINE_PADDING = 30
STARTLINE_COUNT = 25

# 坡道
HILL_DIFF = 15
HILL_CUT = 30
HILL_COUNT = 10

# 环岛
ROUND_MAXWIDTH = 50  # 最大有效宽度，大于这个宽度视为丢线
ROUND_COUNT0 = 3  # 最开始至少丢的行数，设成0就可以不丢
ROUND_DIST2 = 16  # 圆形结束后最丢线的最小距离
ROUND_COUNT3 = 3  # 再次搜到线的最小行数

ROUND_UPCOUNT = 12  # 原图上圆环边先变小的最小个数
ROUND_DOWNCOUNT = 3  # 原图上圆环边变小后变大的最小个数

ROUNDENTER_GETCOUNT = 3  # 入环时从上往下搜的最少有效点
ROUNDENTER_LOSTCOUNT = 3  # 搜到有效点后再至少丢的行数


# 逆透视变换
SRCARR = [  # 原图上的四个点
    (0, 0),  # 左上角
    (0, 187),  # 右上角
    (78, 0),  # 左下角
    (78, 187),  # 右下角
]
PERARR = [  # 新图上的四个点
    (4, -10),  # 左上角
    (4, 245),  # 右上角
    (125, 100),  # 左下角
    (125, 140),  # 右下角
]

# 可视化
SRCZOOM = 5  # 原图放大倍数
N_, M_ = 130, 235  # 新图的高和宽
I_SHIFT = -30  # 新图向下平移
J_SHIFT = 0  # 新图向右平移
PERZOOM = 4  # 新图放大倍数
COLORS = ((255, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0), (0, 127, 127), (127, 127, 0))  # 画点的颜色
