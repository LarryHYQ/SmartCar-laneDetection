if __name__ == "__main__":
    from os.path import dirname, realpath
    from scripts.getConfig import getConfig
    from scripts.ImgProcess import ImgProcess
    import cv2

    dir = dirname(realpath(__file__))
    Config = getConfig(dir)
    imgProcess = ImgProcess(Config["IMGPROCESS"])

    dir = "img\\1.png"
    img = cv2.imread(dir, 0)
    imgProcess.setImg(img)
    imgProcess.work()
    imgProcess.show()
    cv2.waitKey(0)
