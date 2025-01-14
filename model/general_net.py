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
from torchsummary import summary
from transformers import Wav2Vec2Model
import torchvision.transforms as transforms
import numpy as np


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
    def __init__(self, num_classes=4, in_ch=3, pretrained=False):
        super(ModifiedAlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=False)
        if pretrained:
            pre_train_model = torch.load("./pre_trained_model/alex_net/alex_net_pre_train_model.pth")
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

        return out

    def init_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)


class AENet(nn.Module):
    def __init__(self, num_classes=28):
        super(AENet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(in_features=128 * 86 * 10, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch_size, 64, 173, 40)
        x = F.relu(self.conv2(x))  # (batch_size, 64, 173, 40)
        x = self.pool1(x)          # (batch_size, 64, 173, 20)
        x = F.relu(self.conv3(x))  # (batch_size, 128, 173, 20)
        x = F.relu(self.conv4(x))  # (batch_size, 128, 173, 20)
        pool_2_out = self.pool2(x)          # (batch_size, 128, 86, 10)
        x = pool_2_out.view(pool_2_out.size(0), -1)  # (batch_size, 128 * 86 * 10)
        x = F.relu(self.fc1(x))    # (batch_size, 1024)
        x = F.relu(self.fc2(x))    # (batch_size, 1024)
        out = self.fc3(x)            # (batch_size, num_classes)

        pool_2_out = pool_2_out.view(pool_2_out.size(0), pool_2_out.size(1) * pool_2_out.size(3), pool_2_out.size(2))  # [1, 128*10, 86]

        pool_2_out = pool_2_out.permute(0, 2, 1)
        return pool_2_out, out


class ModifiedTransformer(nn.Module):
    def __init__(self, n_features, n_head):
        super(ModifiedTransformer, self).__init__()
        self.n_features = n_features
        self.n_head = n_head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_features, nhead=self.n_head, batch_first=True,dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.batch_norm = nn.BatchNorm1d(n_features)
        self.layer1 = nn.Sequential(
            self.transformer_encoder)
        self.layer4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(16))  # 2d pooling input (batch_size, seq, time)
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc3(out)
        return out

class ModifiedEncoderOnlyTransformer(nn.Module):
    def __init__(self, n_features, n_head):
        super(ModifiedEncoderOnlyTransformer, self).__init__()
        self.n_features = n_features
        self.n_head = n_head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_features, nhead=self.n_head, batch_first=True,dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.layer1 = nn.Sequential(
            self.transformer_encoder)

    def forward(self, x):
        out = self.layer1(x)
        return out

class ModifiedResnet18(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(ModifiedResnet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        if pretrained:
            self.resnet18.load_state_dict(torch.load("./model/pre_trained_model/resnet/resnet18-5c106cde.pth"))
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1)
        # self.base_model = torch.nn.Sequential(*list(origin_model.children())[:-1])
    def forward(self, x):
        out = self.resnet18(x)
        return out

class ModifiedWaveVec2(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(ModifiedWaveVec2, self).__init__()
        self.wav2_layer = Wav2Vec2Model.from_pretrained("./model/pre_trained_model/wav2vec2")
        self.avg_layer = nn.Sequential(nn.AdaptiveAvgPool2d(16))
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.wav2_layer(x).last_hidden_state
        out = self.avg_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc3(out)
        return out

class ModifiedCNN(nn.Module):
    def __init__(self):
        super(ModifiedCNN, self).__init__()
        self.layer1 = nn.Sequential(
            BaseCNN_Conv(173, 128, kernel_size=3, padding=2, dilation=1))
        self.layer2 = nn.Sequential(
            BaseCNN_Conv(128, 256, kernel_size=3, padding=2, dilation=1))
        self.layer3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).squeeze()
        out = self.fc1(out)
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity

class ModifiedLSTM(nn.Module):
    def __init__(self, n_features):
        super(ModifiedLSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.LSTM(input_size=n_features, hidden_size=256))
        self.layer2 = nn.Sequential(
            nn.LSTM(input_size=256, hidden_size=256))
        self.layer3 = nn.Sequential(
            nn.Linear(256,512),nn.ReLU(),nn.BatchNorm1d(173))
        self.linear = nn.Sequential(
            nn.Linear(512, 128),nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(173, 40)

    def forward(self, x):
        out1, state1 = self.layer1(x)
        out2, state2 = self.layer2(out1)
        out = self.layer3(out2) # (batch_size, 173, 512)
        out = self.linear(out).squeeze()
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity

class ModifiedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=None):
        super(ModifiedBiLSTM, self).__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64]
        self.num_layers = len(hidden_layer_sizes)

        self.bi_lstm_layers = nn.ModuleList()

        self.bi_lstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        for i in range(1, self.num_layers):
            self.bi_lstm_layers.append(
                nn.LSTM(hidden_layer_sizes[i - 1] * 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

    def forward(self, x):
        bi_lstm_out = x
        for bi_lstm in self.bi_lstm_layers:
            bi_lstm_out, _ = bi_lstm(bi_lstm_out)

        out = bi_lstm_out
        return out

#
# X = torch.randn(1, 173, 40).cuda()
# model = ModifiedBiLSTM(40).cuda()
# summary(model, X)
