import os
import skimage.io as io
import numpy as np
from pathlib import Path
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from params import Hpyerparams as hp

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# train_sets = datasets.ImageFolder(hp.data_path, transform)
# train_loader = data.DataLoader(train_sets, batch_size=hp.batch_size,
#                                shuffle=False, num_workers=4)
# print(train_sets.class_to_idx)
# print(train_sets.imgs)

class myDataset(data.Dataset):
    def __init__(self, rootpath):
        self.datapath = rootpath
        self.maxlen = 0
        classDir = os.listdir(self.datapath)
        # print(classDir)
        self.splitsList = []
        for i, className in enumerate(classDir):
            classPath = Path(self.datapath) / Path(className)
            # print('=> ', className, classPath)
            listDir = os.listdir(classPath)
            listDir.sort()
            for allDir in listDir:
                imgPaths = os.listdir(classPath / allDir)
                splitList = [classPath/allDir/imgPath for imgPath in imgPaths]
                length = len(splitList)
                if length > self.maxlen:
                    self.maxlen = length
                self.splitsList.append((splitList, className, i))
        # for (lists, name, i) in self.splitsList:
        #     print("=> {}|{}".format(name, i))
        #     print(lists)
    def __len__(self):
        return len(self.splitsList)

    def __getitem__(self, index):
        paths, name, id = self.splitsList[index]
        imgs = []
        for path in paths:
            img = np.transpose(io.imread(path), (2, 0, 1)).astype(np.float32)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        print(imgs.shape)
        imgs, mask = on_mask(imgs, self.maxlen)
        return imgs, mask, id


def on_mask(arr, len, mask_value=0):
    # arr [n, c, h, w]
    # len maxlen
    n, c, h, w = arr.shape
    Arr = np.zeros([len, c, h, w]) + mask_value
    mask = np.zeros([len])
    Arr[:n] = arr
    mask[:n] = 1
    return Arr, mask


if __name__ == '__main__':
    # classDir = os.listdir(hp.data_path)
    # print(classDir)
    # for className in classDir:
    #     classPath = Path(hp.data_path) / Path(className)
    #     print('=> ', className, classPath)
    #     listDir = os.listdir(classPath)
    #     listDir.sort()
    #     for allDir in listDir:
    #         imgPaths = os.listdir(classPath/allDir)
    #         for imgPath in imgPaths:
    #             print(classPath/allDir/imgPath)
    #         print('-'*30)
    mydataset = myDataset(hp.data_path)
    data_loader = data.DataLoader(mydataset)
    for imgs, mask, id in data_loader:
        print("=>", imgs.shape, mask.shape, id)
# inputs, classes = next(iter(train_loader))