import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
from copy import deepcopy


def resize(img, zoom=6):
    return cv2.resize(img, dsize=(0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def waitKey():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rectangle(img, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), zoom=6):
    cv2.rectangle(img, tuple(v * zoom for v in p1), tuple(v * zoom for v in p2), color)


class ZoomedImg:
    def __init__(self, img: np.ndarray, name: str, zoom=6):
        tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.img = cv2.resize(tmp, dsize=(0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        self.name = name
        self.clear()
        self.zoom = zoom

    def clear(self):
        self.canva = deepcopy(self.img)

    def rectangle(self, p1: Tuple[int], p2: Tuple[int], color: Tuple[int] = (0, 0, 255), thickness: int = 2):
        cv2.rectangle(
            self.canva,
            tuple(v * self.zoom for v in reversed(p1)),
            tuple(v * self.zoom for v in reversed(p2)),
            color,
            thickness,
        )

    def plot(self, pt: Tuple[int], color: Tuple[int] = (0, 0, 255), r: int = 4):
        cv2.circle(self.canva, tuple(v * self.zoom for v in reversed(pt)), r, color, -1)

    def polylines(
        self, px, py, color: Tuple[int], thickness: int = 2, i_shift: int = 0, j_shift: int = 0, closed: bool = False
    ):
        pts = (np.asarray([py + j_shift, px + i_shift]).T).astype("int32")
        cv2.polylines(self.canva, [pts * self.zoom], closed, color, thickness)

    def show(self):
        cv2.imshow(self.name, self.canva)
