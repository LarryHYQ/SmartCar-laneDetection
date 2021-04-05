class Main:
    def __init__(self, Config) -> None:
        from tkinter import Tk
        from .ImgProcess import ImgProcess
        from .ImgWindow import ImgWindow

        self.Config = Config
        self.root = Tk()
        self.setProperty()
        self.readDir()
        self.imgWindow = ImgWindow(self)
        self.applyImg()
        self.mainloop = self.root.mainloop

    def setProperty(self) -> None:
        from tkinter import Canvas
        from tkinter.ttk import LabelFrame, Combobox, Button, Label, Frame

        self.root.bind("<a>", lambda event: self.pre())
        self.root.bind("<Left>", lambda event: self.pre())
        self.root.bind("<d>", lambda event: self.nxt())
        self.root.bind("<Right>", lambda event: self.nxt())

    def readDir(self):
        from os import listdir

        self.names = listdir(self.Config["IMGDIR"])
        self.names.sort(key=lambda s: ("".join(filter(str.isalpha, s)), int("".join(filter(str.isdigit, s)))))

    def pre(self):
        if self.Config["INDEX"] > 0:
            self.Config["INDEX"] -= 1
            self.applyImg()

    def nxt(self):
        if self.Config["INDEX"] < len(self.names) - 1:
            self.Config["INDEX"] += 1
            self.applyImg()

    def applyImg(self):
        from cv2 import imread
        from numpy import zeros

        self.Config["INDEX"] = max(0, min(self.Config["INDEX"], len(self.names) - 1))
        img = imread(self.Config["IMGDIR"] + self.names[self.Config["INDEX"]], 0) if self.names else zeros((self.Config["IMGPROCESS"]["N"], self.Config["IMGPROCESS"]["M"]), "uint8")
        self.imgWindow.setImg(img)
        self.imgWindow.imgProcess.work()
        self.imgWindow.showImg()

    def _onClose(self):
        self.Config.write()
        quit()
