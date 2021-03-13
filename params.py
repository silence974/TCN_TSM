import torch
import os

class Hpyerparams:
    # data_path = "D:/Dataset/shortvideo/jpg9_add_resize/"
    # save_path = "D:/Code/PyCharmProject/TCN_TSM/log/"
    data_path = "/home/qin/qinjiaheng/dataset/jpg9_add_resize/"
    save_path = "/home/qin/qinjiaheng/log/"



    # Training
    lr = 0.001
    momentum = 0.9
    beta = 0.999
    weight_decay = 0
    nb_epoch = 100
    epoch_size = 2000

    seed = 114514
    n_class = 3
    n_segment = 32
    n_div = 8

    # ResNet part
    res_layers = 18
    res_pre = True

    # ED_TCN part
    n_feat = 512
    n_nodes = [64, 96]
    n_conv = 25
    causal = False

    sample_len = 7
    max_splits = 30
    batch_size = 4
    img_h = 560
    img_w = 960