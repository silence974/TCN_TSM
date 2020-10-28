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

class myDataset(data.Dataset):
    def __init__(self, rootpath):
        self.datapath = rootpath
        self.maxlen = hp.max_len
        self.labels = 3
        classDir = os.listdir(self.datapath)
        self.classNames = classDir
        self.splitsList = []
        for i, className in enumerate(classDir):
            classPath = Path(self.datapath) / Path(className)
            # print('=> ', className, classPath)
            listDir = os.listdir(classPath)
            print("=> class {} :".format(className), len(listDir))
            listDir.sort()
            for allDir in listDir:
                imgPaths = os.listdir(classPath / allDir)
                splitList = [classPath/allDir/imgPath for imgPath in imgPaths]
                length = len(splitList)
                if length > self.maxlen:
                    for k in range(hp.max_splits):
                        # 最多收集 30 个序列，每个序列和相邻序列间隔 7 帧
                        start = hp.sample_len*k
                        end = start + self.maxlen
                        if end > length: end = length
                        new_splitList = splitList[start:end]
                        self.splitsList.append((new_splitList, className, i))
                        if end == length or start + self.maxlen > length:
                            break

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
    full_dataset = myDataset(hp.data_path)
    names = full_dataset.classNames
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print("=> train: {} test: {}".format(len(train_dataset), len(test_dataset)))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=hp.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=hp.batch_size,
                             shuffle=False, num_workers=4)
    return train_loader, test_loader, names

if __name__ == '__main__':
    # mydataset = myDataset(hp.data_path)
    # print("=> maxlen {}".format(mydataset.maxlen))
    # data_loader = data.DataLoader(mydataset, batch_size=4)
    # for imgs, mask, label in data_loader:
    #     print("=>", imgs.shape, mask.shape, label.shape)
    # --------------------------------------------------
    # mydataset = myDataset(hp.data_path)
    # train_size = int(0.8 * len(mydataset))
    # test_size = len(mydataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(mydataset, [train_size, test_size])
    # print(len(train_dataset), len(test_dataset))
    # ---------------------------------------------------
    tr, te, na = get_dataloader()
    for i, (img, m, y) in enumerate(tr):
        print(img.shape, m.shape, y.shape)
        a = input()