from .Defaults import *
from configobj import ConfigObj


def getConfig(dir):
    ConfigDir = dir + "\\config.ini"
    Config = ConfigObj(ConfigDir)

    Config["IMGDIR"] = Config["IMGDIR"] if "IMGDIR" in Config else "img\\"
    if Config["IMGDIR"][-1] not in "/\\":
        Config["IMGDIR"] += "\\"
    Config["INDEX"] = Config["INDEX"] if "INDEX" in Config else 0

    if "IMGPROCESS" not in Config:
        Config["IMGPROCESS"] = {}
    IMGPROCESS = Config["IMGPROCESS"]
    IMGPROCESS["N"] = int(IMGPROCESS["N"]) if "N" in IMGPROCESS else N
    IMGPROCESS["M"] = int(IMGPROCESS["M"]) if "M" in IMGPROCESS else M
    IMGPROCESS["NOISE"] = int(IMGPROCESS["NOISE"]) if "NOISE" in IMGPROCESS else NOISE
    IMGPROCESS["H"] = int(IMGPROCESS["H"]) if "H" in IMGPROCESS else H
    IMGPROCESS["W"] = int(IMGPROCESS["W"]) if "W" in IMGPROCESS else W
    IMGPROCESS["PADDING"] = int(IMGPROCESS["PADDING"]) if "PADDING" in IMGPROCESS else PADDING
    IMGPROCESS["DERI_THRESHOLD"] = int(IMGPROCESS["DERI_THRESHOLD"]) if "DERI_THRESHOLD" in IMGPROCESS else DERI_THRESHOLD
    IMGPROCESS["SUM_THRESHOLD"] = int(IMGPROCESS["SUM_THRESHOLD"]) if "SUM_THRESHOLD" in IMGPROCESS else SUM_THRESHOLD
    IMGPROCESS["N_"] = int(IMGPROCESS["N_"]) if "N_" in IMGPROCESS else N_
    IMGPROCESS["M_"] = int(IMGPROCESS["M_"]) if "M_" in IMGPROCESS else M_
    IMGPROCESS["I_SHIFT"] = int(IMGPROCESS["I_SHIFT"]) if "I_SHIFT" in IMGPROCESS else I_SHIFT
    IMGPROCESS["J_SHIFT"] = int(IMGPROCESS["J_SHIFT"]) if "J_SHIFT" in IMGPROCESS else J_SHIFT
    IMGPROCESS["SRCARR"] = [eval(li) for li in IMGPROCESS["SRCARR"]] if "SRCARR" in IMGPROCESS else SRCARR
    IMGPROCESS["PERARR"] = [eval(li) for li in IMGPROCESS["PERARR"]] if "PERARR" in IMGPROCESS else PERARR

    Config.write()
    return Config


__all__ = ["getConfig"]

