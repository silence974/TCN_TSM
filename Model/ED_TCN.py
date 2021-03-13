import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchsummary import summary
from Model.BaseModules import *
from params import Hpyerparams as hp
import torchsnooper

class ED_TCN_Model(nn.Module):
    def __init__(self, n_feat, n_nodes, n_classes,
                 conv_len, causal=False, dropout=0.3,
                 activation='norm_relu'):
        super(ED_TCN_Model, self).__init__()
        n_layers = len(n_nodes)

        en_layers = []
        for i in range(n_layers):
            in_channel = n_feat if i == 0 else n_nodes[i-1]
            out_channel = n_nodes[i]
            en_layers += [TCN_Block(in_channel, out_channel, conv_len, padding=conv_len//2,
                                    causal=causal, dropout=dropout, activation=activation),
                          nn.MaxPool1d(2)]
        self.en_net = nn.Sequential(*en_layers)
        de_layers = []
        for i in range(n_layers):
            in_channel = n_nodes[-1] if i == 0 else n_nodes[-i]
            out_channel = n_nodes[-i-1]
            de_layers += [nn.Upsample(scale_factor=2),
                          TCN_Block(in_channel, out_channel, conv_len, padding=conv_len//2,
                                    causal=causal, dropout=dropout, activation=activation)]
        self.de_net = nn.Sequential(*de_layers)
        self.fc = TimeDistributed(nn.Linear(n_nodes[0], n_classes)) # [B, n_classes, L]

    def forward(self, x):
        out = self.en_net(x)
        out = self.de_net(out)
        out = F.softmax(self.fc(out), dim=1)
        return out

if __name__ == '__main__':
    model = ED_TCN_Model(hp.n_feat, hp.n_nodes, hp.n_class, hp.n_conv)
    print(summary(model, (256, 128)))