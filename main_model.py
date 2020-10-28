import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim

import os
import time
import datetime
from path import Path

from Model import *
from params import Hpyerparams as hp
from dataloader import myDataset


class myModel(object):
    def __init__(self):
        pass

    def prepare_info(self):
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = Path(hp.save_path)
        self.save_path = save_path/timestamp
        print('=> will save everything to {}'.format(self.save_path))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)

        cudnn.daterministic = True
        cudnn.benchmark = True

    def prepare_data(self):
        mydataset = myDataset(hp.data_path)
        print("=> load dataset from {}".format(hp.data_path))
        print("=> segment maxlen {}".format(mydataset.maxlen))
        self.data_loader = data.DataLoader(mydataset)


    