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
        from tkinter import Canvas, StringVar
        from tkinter.ttk import LabelFrame, Combobox, Button, Label, Frame, Entry

        self.indexLabel = Label(self.root, text="索引:")
        self.indexVar = StringVar()
        self.indexCheck = lambda strVar=self.indexVar, config=self.Config, key="INDEX": self.entryCallback(strVar, config, key)
        self.indexCheck()
        self.indexEntry = Entry(self.root, textvariable=self.indexVar, validate="focusout", validatecommand=self.indexCheck, width=10)
        self.indexEntry.bind("<Return>", lambda e: (self.indexCheck(), self.applyImg()))
        self.indexButton = Button(self.root, text="确认", command=self.applyImg)
        self.indexLabel.grid(row=0, column=0, padx=3, pady=3)
        self.indexEntry.grid(row=0, column=1, padx=3, pady=3)
        self.indexButton.grid(row=0, column=2, padx=3, pady=3)

        self.root.title("设置")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.attributes("-toolwindow", True)

        self.root.protocol("WM_DELETE_WINDOW", self._onClose)

    def readDir(self):
        from os import listdir, rename

        self.names = listdir(self.Config["IMGDIR"])
        self.names.sort(key=lambda s: ("".join(filter(str.isalpha, s)), int("".join(filter(str.isdigit, s)))))
        for i, c in enumerate(self.names):
            rename(self.Config["IMGDIR"] + c, self.Config["IMGDIR"] + str(i) + ".png")
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

    def delImg(self):
        from os import remove

        if len(self.names) != 0:
            remove(self.names.pop(self.Config["INDEX"]))
            self.applyImg()

    def entryCallback(self, strVar, config: dict, key: str, Type=int):
        try:
            config[key] = Type(strVar.get())
            return True
        except:
            strVar.set(str(config[key]))
            return False

    def applyImg(self):
        from cv2 import imread
        from numpy import zeros

        self.indexVar.set(str(self.Config["INDEX"]))
        self.Config["INDEX"] = max(0, min(self.Config["INDEX"], len(self.names) - 1))
        img = imread(self.Config["IMGDIR"] + self.names[self.Config["INDEX"]], 0) if self.names else zeros((self.Config["IMGPROCESS"]["N"], self.Config["IMGPROCESS"]["M"]), "uint8")
        self.imgWindow.setImg(img)
        self.imgWindow.imgProcess.work()
        self.imgWindow.showImg()

    def _onClose(self):
        self.Config.write()
        quit()
