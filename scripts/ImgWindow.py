from .Main import Main
from Config import *


class ImgWindow:
    def __init__(self, main: Main) -> None:
        from tkinter import Toplevel
        from .ImgProcess import ImgProcess

        self.main = main
        self.imgProcess = ImgProcess()
        self.root = Toplevel(self.main.root)
        self.setProperty()
        self.setImg = self.imgProcess.setImg
        self.holding = False

    def setProperty(self) -> None:
        from tkinter import Canvas
        from tkinter.ttk import LabelFrame

        self.srcFrame = LabelFrame(self.root, text="原图")
        self.srcCanvas = Canvas(self.srcFrame, height=N * SRCZOOM, width=M * SRCZOOM)
        self.perFrame = LabelFrame(self.root, text="逆透视变换")
        self.perCanvas = Canvas(self.perFrame, height=N_ * PERZOOM, width=M_ * PERZOOM)

        self.root.title("图像")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.main._onClose)

        self.srcFrame.pack()
        self.srcCanvas.pack(padx=3, pady=3)
        self.perFrame.pack()
        self.perCanvas.pack(padx=3, pady=3)

        self.root.bind("<a>", lambda e: self.main.pre())
        self.root.bind("<Left>", lambda e: self.main.pre())
        self.root.bind("<d>", lambda e: self.main.nxt())
        self.root.bind("<Right>", lambda e: self.main.nxt())
        self.root.bind("<space>", lambda e: self.hold())

    def showImg(self) -> None:
        from PIL import Image
        from PIL.ImageTk import PhotoImage

        self.srcImgTk = PhotoImage(image=Image.fromarray(self.imgProcess.SrcShow.canvas))
        self.srcCanvas.create_image(0, 0, anchor="nw", image=self.srcImgTk)
        self.perImgTk = PhotoImage(image=Image.fromarray(self.imgProcess.PerShow.canvas))
        self.perCanvas.create_image(0, 0, anchor="nw", image=self.perImgTk)

    def hold(self) -> None:
        if not self.holding:
            self.holding = True
            self.continuously()
        else:
            self.holding = False

    def continuously(self) -> None:
        if self.holding:
            self.main.nxt()
            self.root.after(100, self.continuously)
