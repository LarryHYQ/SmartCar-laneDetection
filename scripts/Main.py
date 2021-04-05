import numpy as np


class Main:
    def __init__(self, Config) -> None:
        from tkinter import Tk
        from .ImgProcess import ImgProcess

        self.Config = Config
        self.getConfig()

        self.imgProcess = ImgProcess(Config["IMGPROCESS"])
        self.imgProcess.setImg(np.zeros((self.N, self.M), "uint8"))

        self.root = Tk()
        self.setProperty()

    def getConfig(self) -> None:
        self.N, self.M, self.N_, self.M_ = self.Config["IMGPROCESS"]["N"], self.Config["IMGPROCESS"]["M"], self.Config["IMGPROCESS"]["N_"], self.Config["IMGPROCESS"]["M_"]
        self.SRCZOOM, self.PERZOOM = self.Config["IMGPROCESS"]["SRCZOOM"], self.Config["IMGPROCESS"]["PERZOOM"]

    def setProperty(self) -> None:
        from tkinter import Canvas
        from tkinter.ttk import LabelFrame, Combobox, Button, Label, Frame

        self.imgFrame = Frame(self.root)
        self.srcFrame = LabelFrame(self.imgFrame, text="原图")
        self.srcCanvas = Canvas(self.srcFrame, height=self.N * self.SRCZOOM, width=self.M * self.SRCZOOM)
        self.perFrame = LabelFrame(self.imgFrame, text="逆透视变换")
        self.perCanvas = Canvas(self.perFrame, height=self.N_ * self.PERZOOM, width=self.M_ * self.PERZOOM)

        self.imgFrame.pack(padx=3, pady=3)
        self.srcFrame.pack()
        self.srcCanvas.pack(padx=3, pady=3)
        self.perFrame.pack()
        self.perCanvas.pack(padx=3, pady=3)
        self.showImg()

    def showImg(self):
        from PIL import Image
        from PIL.ImageTk import PhotoImage

        self.srcImgTk = PhotoImage(image=Image.fromarray(self.imgProcess.SrcShow.canvas))
        self.srcCanvas.create_image(0, 0, anchor="nw", image=self.srcImgTk)
        self.perImgTk = PhotoImage(image=Image.fromarray(self.imgProcess.PerShow.canvas))
        self.perCanvas.create_image(0, 0, anchor="nw", image=self.perImgTk)

    def mainloop(self) -> None:
        self.root.mainloop()
