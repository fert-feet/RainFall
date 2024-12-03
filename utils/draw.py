# -*- coding: utf-8 -*-
# @Time : 2023/2/22 15:07 
# @Author : Mingzheng 
# @File : draw.py
# @desc :
import os

import numpy as np
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib.pyplot import xticks
# from utils.dataloader import USRADataset,USRADataset_collate,USRADataset_CR,USRADataset_collate_CR
from torch.utils.data import DataLoader
# from nets.baseline_training import get_lr_scheduler, set_optimizer_lr
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from tqdm import tqdm
# from nets.general_net import *
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

class Plotter:
    def __init__(self, labels, outputs, R2, RMSE, MSE, MAE, feature_name):
        self.labels = labels
        self.outputs = outputs
        self.R2 = R2
        self.RMSE = RMSE
        self.MSE = MSE
        self.MAE = MAE
        self.feature_name = feature_name
        self.img_save_path = "./data/img"

    def simply_draw(self):
        plt.figure(figsize=(14, 7))

        plt.plot(self.R2, label='R2', color='#1f77b4', linestyle='-', linewidth=1.5, marker='o', markersize=3)

        plt.plot(self.MSE, label='MSE', color='#ff7f0e', linestyle='--', linewidth=1.5, marker='s', markersize=3)

        plt.plot(self.RMSE, label='RMSE', color='#2ca02c', linestyle='-.', linewidth=1.5, marker='^', markersize=3)

        plt.plot(self.MAE, label='MAE', color='#d62728', linestyle=':', linewidth=1.5, marker='x', markersize=3)

        plt.legend(fontsize=12)

        plt.title('Metrics over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        plt.savefig(os.path.join(self.img_save_path, self.feature_name), dpi=300)

