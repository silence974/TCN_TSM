import torch
import os

class Hpyerparams:
    data_path = "D:/Dataset/shortvideo/jpg9_add_resize/"
    save_path = "D:/Code/PyCharmProject/TCN_TSM/log/"
    # data_path = "/home/qin/qinjiaheng/dataset/jpg9_add_resize/"

    seed = 114514

    max_len = 128
    sample_len = 7
    max_splits = 30
    batch_size = 4
    img_h = 560
    img_w = 960