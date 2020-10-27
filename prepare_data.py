import os
from PIL import Image
import skimage.io as io
import numpy as np
from pathlib import Path
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from params import Hpyerparams as hp



if __name__ == '__main__':
    img_h = 224
    img_w = 224
    # pre_path = "/home/qin/qinjiaheng/dataset/jpg9_add/"
    # out_path = "/home/qin/qinjiaheng/dataset/jpg9_add_resize/"
    pre_path = "D:/Dataset/shortvideo/jpg9_add/"
    out_path = "D:/Dataset/shortvideo/jpg9_add_resize/"
    classDir = os.listdir(pre_path)
    for className in classDir:
        classPath = Path(pre_path)/Path(className)
        classPath_out = Path(out_path)/Path(className)
        if not os.path.exists(classPath_out):
            os.makedirs(classPath_out)
        listDir = os.listdir(classPath)
        listDir.sort()
        for allDir in listDir:
            splitPath = classPath / allDir
            splitPath_out = classPath_out / allDir
            if not os.path.exists(splitPath_out):
                os.makedirs(splitPath_out)
            imgPaths = os.listdir(splitPath)
            for img in imgPaths:
                readPath = splitPath / img
                readPath_out = splitPath_out / img
                im = Image.open(readPath)
                out = im.resize((img_h, img_w), Image.ANTIALIAS)
                out.save(readPath_out)
                print("=> modify out at {}".format(readPath_out))