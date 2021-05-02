# 图像
N, M = 80, 188  # 图片的高和宽
CUT = 20  # 裁剪最上面的多少行
PADDING = 1  # 舍弃左右边界的大小

# sobel
LRSTEP = 2
UDSTEP = 1
THRESHLOD = 100  # sobel的阈值

# 用于排除点的斜率范围
KLOW = -0.2  # 斜率下限
KHIGH = 1.5  # 斜率上限

# 获取目标点
DIST = 20  # 垂足向上扩展的长度

# 路径生成
K_RHO = 5  # ρ增益
K_ALPHA = 15  # α增益
K_BETA = -3  # β增益

# 逆透视变换
SRCARR = [  # 原图上的四个点
    (20, 49),  # 左上角
    (20, 76),  # 右上角
    (59, 0),  # 左下角
    (59, 119),  # 右下角
]
PERARR = [  # 新图上的四个点
    (0, 49),  # 左上角
    (0, 86),  # 右上角
    (120, 69),  # 左下角
    (120, 106),  # 右下角
]

# 可视化
SRCZOOM = 5  # 原图放大倍数
N_, M_ = 130, 235  # 新图的高和宽
I_SHIFT = -50  # 新图向下平移
J_SHIFT = 20  # 新图向右平移
PERZOOM = 4  # 新图放大倍数
