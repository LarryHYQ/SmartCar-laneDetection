"一些杂项工具"
import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List
from copy import deepcopy


class ZoomedImg:
    "指定缩放的图片帮助类"

    def __init__(self, img: np.ndarray, zoom: int, name: str):
        """指定缩放的图片帮助类

        Args:
            img (np.ndarray): 原图矩阵，需要先使用cv2.imread(xxx, 0)读入
            name (str): 窗口的标题
            zoom (int): 缩放倍数
        """
        self.img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), dsize=(0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        self.name = name
        self.clear()
        self.zoom = zoom

    def clear(self):
        "清空图像画布上后画的东西"
        self.canvas = deepcopy(self.img)

    def rectangle(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2):
        """在图像画布上画一个矩形

        Args:
            p1 (Tuple[int]): 矩形的左上角 (i1, j1)
            p2 (Tuple[int]): 矩形的右下角 (i2, j2)
            color (Tuple[int], 可选): 矩形的颜色元组，默认为红色 (0, 0, 255)
            thickness (int, 可选): 矩形线条的粗细，默认为 2
        """
        cv2.rectangle(self.canvas, tuple(v * self.zoom for v in reversed(p1)), tuple(v * self.zoom for v in reversed(p2)), color, thickness)

    def point(self, pt: Tuple[int], color: Tuple[int] = (0, 0, 255), r: int = 4):
        """在图像画布上画点

        Args:
            pt (Tuple[int]): 点的坐标 (i, j)
            color (Tuple[int], 可选): 点的颜色元组，默认为红色 (0, 0, 255)
            r (int, 可选): 点的大小，默认为 4
        """
        cv2.circle(self.canvas, tuple(v * self.zoom for v in reversed(pt)), r, color, -1)

    def polylines(self, pi: List[int], pj: List[int], color: Tuple[int], thickness: int = 2, i_shift: int = 0, j_shift: int = 0, closed: bool = False):
        """在图像画布上画出函数图像

        Args:
            pi (List[int] | np.array): 纵向点集
            pj (List[int] | np.array): 横向点集
            color (Tuple[int]): 点的颜色元组
            thickness (int, 可选): 线条的粗细，默认为 2
            i_shift (int, 可选): 纵向偏移，默认为 0
            j_shift (int, 可选): 横向偏移，默认为 0
            closed (bool, 可选): 图像是否闭合，默认为 False
        """
        pi = np.array(pi)
        pj = np.array(pj)
        pts = (np.asarray([pj + j_shift, pi + i_shift]).T).astype("int32")
        cv2.polylines(self.canvas, [pts * self.zoom], closed, color, thickness)

    def show(self):
        cv2.imshow(self.name, self.canvas)
