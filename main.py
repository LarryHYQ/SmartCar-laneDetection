if __name__ == "__main__":
    from os.path import dirname, realpath
    from scripts.getConfig import getConfig
    from scripts.Main import Main
    from scripts.ImgProcess import ImgProcess
    import cv2

    dir = dirname(realpath(__file__))
    Config = getConfig(dir)
    imgProcess = ImgProcess(Config["IMGPROCESS"])
    main = Main(Config)
    dir = "img\\5.png"
    img = cv2.imread(dir, 0)
    main.imgProcess.setImg(img)
    main.imgProcess.work()
    main.showImg()
    main.mainloop()
