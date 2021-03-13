import os
import skimage.io as io
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from params import Hpyerparams as hp

def split_dir(rootpath):
    classDir = os.listdir(rootpath)
    split_dic = {}
    for i, className in enumerate(classDir):
        classPath = Path(rootpath) / Path(className)
        listDir = os.listdir(classPath)
        listDir.sort()
        trainsize = int(0.8*len(listDir))
        tr_set = listDir[:trainsize]
        te_set = listDir[trainsize:]
        split_dic[className] = [tr_set, te_set]
    return split_dic

class myDataset(data.Dataset):
    def __init__(self, type, split_dic, rootpath):
        self.maxlen = hp.n_segment
        self.labels = hp.n_class
        self.datapath = rootpath
        self.classNames = split_dic.keys()
        self.splitsList = []
        for i, className in enumerate(self.classNames):
            classPath = Path(rootpath) / Path(className)
            listDir = split_dic[className][type]
            for allDir in listDir:
                imgPaths = os.listdir(classPath / allDir)
                splitList = [classPath / allDir / imgPath for imgPath in imgPaths]
                length = len(splitList)
                if length < self.maxlen:
                    continue
                space = length // self.maxlen
                new_splitList = []
                for k in range(self.maxlen):
                    new_splitList.append(splitList[k * space])
                self.splitsList.append((new_splitList, className, i))
                # if length > self.maxlen:
                #     for k in range(hp.max_splits):
                #         # 最多收集 30 个序列，每个序列和相邻序列间隔 7 帧
                #         start = hp.sample_len * k
                #         end = start + self.maxlen
                #         if end > length: end = length
                #         new_splitList = splitList[start:end]
                #         self.splitsList.append((new_splitList, className, i))
                #         if end == length or start + self.maxlen > length:
                #             break

    def __len__(self):
        return len(self.splitsList)

    def __getitem__(self, index):
        paths, name, id = self.splitsList[index]
        imgs = []
        for path in paths:
            img = np.transpose(io.imread(path), (2, 0, 1)).astype(np.float32)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        imgs, mask = on_mask(imgs, self.maxlen)
        imgs = torch.tensor(imgs, dtype=torch.float32)
        mask = torch.tensor(mask)
        label = F.one_hot(torch.tensor(id), self.labels)
        return imgs, mask, label

def on_mask(arr, len, mask_value=0):
    # arr [n, c, h, w]
    # len maxlen
    n, c, h, w = arr.shape
    Arr = np.zeros([len, c, h, w]) + mask_value
    mask = np.zeros([len])
    Arr[:n] = arr
    mask[:n] = 1
    return Arr, mask


def get_dataloader():
    split_dic = split_dir(hp.data_path)
    names = split_dic.keys()
    train_dataset = myDataset(type=0, split_dic=split_dic, rootpath=hp.data_path)
    valid_dataset = myDataset(type=1, split_dic=split_dic, rootpath=hp.data_path)
    print("=> train: {} test: {}".format(len(train_dataset), len(valid_dataset)))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=hp.batch_size,
                              shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=hp.batch_size,
                              shuffle=False, num_workers=4)
    return train_loader, valid_loader, list(names)


if __name__ == "__main__":
    tr, te, na = get_dataloader()
    print(na)