"图像的逆透视变换"
from typing import List, Tuple
import numpy as np
import cv2


def getPerMat(fromPoints: List[Tuple[int]], toPoints: List[Tuple[int]]) -> List[float]:
    """用cv2生成变换矩阵

    Args:
        fromPoints (List[Tuple[int]]): 原图上的4个点
        toPoints (List[Tuple[int]]): 新图上的4个点

    Returns:
        List[float]: 生成的变换矩阵
    """
    return (
        cv2.getPerspectiveTransform(np.array(fromPoints, dtype="float32"), np.array(toPoints, dtype="float32"))
        .astype("float32")
        .flatten()
    )


def axisTransform(i: int, j: int, perMat: np.array) -> Tuple[float]:
    """使用变换矩阵映射坐标

    Args:
        i (int): 行号
        j (int): 列号
        perMat (np.array): 变换矩阵

    Returns:
        Tuple[float]: 以浮点数形式返回变换后的 (i, j)
    """

    "使用变换矩阵映射坐标"
    a = i * perMat[0] + j * perMat[1] + perMat[2]
    b = i * perMat[3] + j * perMat[4] + perMat[5]
    c = i * perMat[6] + j * perMat[7] + perMat[8]
    return a / c, b / c


def transfomImg(
    src: np.ndarray, perMat: np.array, N: int, M: int, N_: int, M_: int, i_shift: int, j_shift: int
) -> np.ndarray:
    """使用变换矩阵对图像进行逆透视变换并返回变换后的图像

    Args:
        src (np.ndarray): 原图
        perMat (np.array): 变换矩阵
        N (int): 原图的高
        M (int): 原图的宽
        N_ (int): 输出图片的高度
        M_ (int): 输出图片的宽度
        i_shift (int): 输出向下偏移
        j_shift (int): 输出向右偏移

    Returns:
        np.ndarray: 变换后的图像
    """
    per = np.zeros((N_, M_), "uint8")
    for i in range(N):
        for j in range(M):
            u, v = axisTransform(i, j, perMat)
            u = round(u + i_shift)
            v = round(v + j_shift)
            if 0 <= u < N_ and 0 <= v < M_:
                per[u, v] = src[i, j]
    return per


def writeFile(perMat: np.array) -> None:
    """将变换矩阵写入文件

    Args:
        perMat (np.array): 变换矩阵
    """
    with open("PERMAT.cpp", "w") as f:
        f.write("typedef unsigned int uint32;  // clang-format off\nconst uint32 PERMAT[9]={0x")
        tmp = perMat.tobytes().hex(" ", 4).split()
        tmp = tmp = ["".join(a[i : i + 2] for i in range(6, -1, -2)) for a in tmp]
        f.write(",0x".join(tmp) + "};")


__all__ = ["getPerMat", "axisTransform", "transfomImg", "writeFile"]

if __name__ == "__main__":
    N = 80  # 原图的高
    M = 188  # 原图的宽

    # 这几项只用于显示，不参与运算
    N_ = 200  # 新图的高
    M_ = 200  # 新图的宽
    i_shift = 50  # 向下平移
    j_shift = 20  # 向右平移
    #
    index = 62

    DIR = "D:\\CarImg\\"
    IMGDIR = "2.BMP"  # 图片的路径
    srcArr = [  # 原图上的四个点
        (0, 49),  # 左上角
        (0, 76),  # 右上角
        (59, 0),  # 左下角
        (59, 119),  # 右下角
    ]
    perArr = [  # 新图上的四个点
        (0, 49),  # 左上角
        (0, 76),  # 右上角
        (120, 49),  # 左下角
        (120, 76),  # 右下角
    ]

    perMat = getPerMat(srcArr, perArr)
    writeFile(perMat)
    img1 = cv2.imread(DIR + str(index) + ".png", 0)
    img2 = np.zeros((N_, M_), "uint8")
    for i in range(N):
        for j in range(M):
            u, v = axisTransform(i, j, perMat)
            u = round(u + i_shift)
            v = round(v + j_shift)
            if 0 <= u < N_ and 0 <= v < M_:
                img2[u, v] = img1[i, j]
    img1 = cv2.resize(img1, dsize=(0, 0), fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    img2 = cv2.resize(img2, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("old", img1)
    cv2.imshow("new", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
