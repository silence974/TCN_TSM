import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim

from tensorboardX import SummaryWriter

import os
import time
import datetime
from path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

import metrics
from utils import *
from Model import *
from params import Hpyerparams as hp
from dataloader import get_dataloader
from logger import TermLogger, AverageMeter

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


class myModel(object):
    def __init__(self):
        self.prepare_info()
        self.prepare_data()
        self.BaseNet()
        self.prepare_optim()

    def prepare_info(self):
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        save_path = Path(hp.save_path)
        self.save_path = save_path/timestamp
        print('=> will save everything to {}'.format(self.save_path))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)

        cudnn.daterministic = True
        cudnn.benchmark = True

        self.training_writer = SummaryWriter(self.save_path / 'train')
        self.validing_writer = SummaryWriter(self.save_path / 'valid')

    def prepare_data(self):
        print("=> load dataset from {}".format(hp.data_path))
        train_loader, valid_loader, names = get_dataloader()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.classNames = names
        self.trial_metrics = metrics.ComputeMetrics(overlap=0.1, bg_class=0)
        self.trial_metrics.set_classes(hp.n_class)

    def BaseNet(self):
        global device
        self.tsm = TemporalShift(n_segment=hp.n_segment, n_div=hp.n_div)
        self.tsm = torch.nn.DataParallel(self.tsm)
        self.resenc = ResnetEncoder(self.tsm, hp.res_layers, pretrained=hp.res_pre)
        self.tcn = ED_TCN_Model(hp.n_feat, hp.n_nodes, hp.n_class, hp.n_conv,
                                causal=hp.causal, dropout=0.3, activation='norm_relu')
        self.resenc = torch.nn.DataParallel(self.resenc)
        self.tcn = torch.nn.DataParallel(self.tcn)

    def prepare_optim(self):
        print("=> setting adam solver")
        optim_params = [
            {'params': self.resenc.parameters(), 'lr': hp.lr},
            {'params': self.tcn.parameters(),    'lr': hp.lr},
        ]
        self.optimizer = optim.Adam(optim_params,
                                    betas=(hp.momentum, hp.beta),
                                    weight_decay=hp.weight_decay)

    def compute_loss(self):
        global n_iter, device
        end = time.time()
        self.logger.train_bar.update(0)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)
        accres = AverageMeter(precision=4)

        self.resenc.train()
        self.tcn.train()

        for i, (img, mask, y) in enumerate(self.train_loader):
            if img.shape[0] != hp.batch_size:
                break

            log_losses = i > 0 and n_iter % 5 == 0

            data_time.update(time.time() - end)
            img = img.to(device)
            mask = mask.to(device)
            y = y.to(device)

            n, t, c, h, w = img.shape
            img = img.reshape([n*t, c, h, w])
            img = self.tsm(img)
            feats = self.resenc(img).reshape([n, t, -1]).permute(0, 2, 1)
            # [n, 512, t]
            out = self.tcn(feats)[:, :, -1]
            # [n, 3, t] -> [n, 3]

            loss = -( y * torch.log(out)).sum(dim=1, keepdim=False)
            loss = loss.mean()

            index = out.argmax(dim=1, keepdim=False)
            pred_y = index.detach().cpu().numpy()
            true = y.argmax(dim=1, keepdim=False)
            true_y = true.detach().cpu().numpy()

            accr = (pred_y == true_y).astype(np.int32)
            for x in accr:
                accres.update(x)

            if log_losses:
                self.training_writer.add_scalar('loss', loss.item(), n_iter)
                self.training_writer.add_scalar('accr', accres.avg[0], n_iter)

            losses.update(loss.item(), hp.batch_size)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()


            if n_iter % 10 == 0:
                self.logger.train_writer.write('Train: Time {} Data {} Loss {} Accr {:.3f}'.format(
                    batch_time, data_time, losses, accres.avg[0]*100
                ))
            n_iter += 1
            if i >= hp.epoch_size:
                break
            self.logger.train_bar.update(i+1)
        return losses.avg[0]

    def save_model(self, iter_, model_dir, model):
        torch.save({'iteration': iter_, 'model_state_dict': model.state_dict()}, model_dir)

    def train(self):
        global best_error, n_iter, device
        self.logger = TermLogger(n_epochs=hp.nb_epoch,
                                 train_size=min(len(self.train_loader), hp.epoch_size),
                                 valid_size=len(self.valid_loader))
        self.logger.epoch_bar.start()
        for epoch in range(hp.nb_epoch):
            self.logger.epoch_bar.update(epoch)
            self.logger.reset_train_bar()

            train_loss = self.compute_loss()
            self.logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

            self.logger.reset_valid_bar()
            accr, edit, f1 = self.validate_with_gt(epoch)
            self.logger.valid_writer.write(' * Avg Accr {} Edit {} F1 {}'\
                                           .format(accr, edit, f1))
            self.save_model(n_iter, self.save_path / 'resenc_{}.pth'.format(epoch), self.resenc)
            self.save_model(n_iter, self.save_path / 'tcn_{}.pth'.format(epoch), self.tcn)


    @torch.no_grad()
    def validate_with_gt(self, epoch):
        global device
        batch_time = AverageMeter()
        losses = AverageMeter(precision=4)
        conf_matrix = torch.zeros(hp.n_class, hp.n_class)

        preds = []
        trues = []

        self.resenc.eval()
        self.tcn.eval()
        end = time.time()
        self.logger.valid_bar.update(0)
        length = len(self.valid_loader)

        for i, (img, mask, y) in enumerate(self.valid_loader):
            if img.shape[0] != hp.batch_size:
                break

            img = img.to(device)
            mask = mask.to(device)
            y = y.to(device)

            n, t, c, h, w = img.shape
            img = img.reshape([n * t, c, h, w])
            img = self.tsm(img)
            feats = self.resenc(img).reshape([n, t, -1]).permute(0, 2, 1)
            # [n, 512, t]
            out = self.tcn(feats)[:, :, -1]
            # [n, 3, t] -> [n, 3]

            loss = -(y * torch.log(out)).sum(dim=1, keepdim=False)
            loss = loss.mean()
            losses.update(loss.item(), hp.batch_size)


            index = out.argmax(dim=1, keepdim=False)
            pred_y = index.detach().cpu().numpy()
            true = y.argmax(dim=1, keepdim=False)
            true_y = true.detach().cpu().numpy()
            conf_matrix = confusion_matrix(pred_y, true_y, conf_matrix)

            preds.append(pred_y)
            trues.append(true_y)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                self.logger.valid_writer.write('Valid: Time {} Loss {}'\
                                               .format(batch_time, losses))
            if i >= length:
                break

            self.logger.valid_bar.update(i+1)
        conf_matrix = conf_matrix.numpy()
        self.validing_writer.add_figure('confusion matrix',
                                        figure=plot_confusion_matrix(conf_matrix, self.classNames),
                                        global_step=epoch)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        name = "E{}".format(epoch)
        self.trial_metrics.add_predictions(name, preds, trues)
        accr, edit, f1 = self.trial_metrics.get_scores(name)
        self.validing_writer.add_scalar('accr_val', accr, epoch)
        self.validing_writer.add_scalar('edit_val', edit, epoch)
        self.validing_writer.add_scalar('f1_val', f1, epoch)
        return accr, edit, f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)
    cm = cm.astype(np.int32)
    fig = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    return fig


if __name__ == "__main__":
    mymodel = myModel()
    mymodel.train()