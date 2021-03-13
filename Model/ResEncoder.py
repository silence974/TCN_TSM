from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, TSM, num_layers, pretrained=True):
        super(ResnetEncoder, self).__init__()

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}
        self.tsm = TSM
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        self.encoder = resnets[num_layers](pretrained)


    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(self.tsm(x))
        x = self.encoder.layer2(self.tsm(x))
        x = self.encoder.layer3(self.tsm(x))
        x = self.encoder.layer4(self.tsm(x))

        x = self.encoder.avgpool(x)
        out = torch.flatten(x, 1)

        return out


if __name__ == "__main__":
    inputs = torch.zeros([4, 12, 3, 224, 224], dtype=torch.float32)
    inputs = inputs.reshape([-1, 3, 224, 224])
    resenc = ResnetEncoder(18, True)
    outs = resenc(inputs)
    print(outs.shape)