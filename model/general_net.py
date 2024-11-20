# -*- coding: utf-8 -*-
# @Time : 2023/1/12 17:02 
# @Author : Mingzheng 
# @File : general_net.py
# @desc :



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.hub import load_state_dict_from_url


class BaseCNN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dr = nn.Dropout(0.5)

    def forward(self, x):
        # return self.sd(self.bn(F.relu(self.conv(x))))
        return self.dr(self.bn(F.leaky_relu(self.conv(x))))

class GCNN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dr = nn.Dropout(0.5)

    def forward(self, x):
        # return self.sd(self.bn(F.relu(self.conv(x))))
        return self.dr(self.bn(F.leaky_relu(self.conv(x))))


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(ModifiedAlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        pre_train_model = torch.load("./model/pre_trained_model/alex_net/alex_net_pre_train_model.pth")
        model.load_state_dict(pre_train_model)

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            self.init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)

        # self._init_weights(pretrained=pretrained) # don't init weight, we use offline pre_train file

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)

        return x, out

    def init_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)


class ModifiedTransformer(nn.Module):
    # TODO finish this
    pass

