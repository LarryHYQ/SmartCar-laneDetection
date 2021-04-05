if __name__ == "__main__":
    from os.path import dirname, realpath
    from scripts.getConfig import getConfig
    from scripts.Main import Main

    dir = dirname(realpath(__file__))
    Config = getConfig(dir)
    main = Main(Config)
    main.imgWindow.imgProcess.work()
    main.imgWindow.showImg()
    main.mainloop()
