import librosa
import pywt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# file = np.load('/home/project/papercode/EstimatingRainFall/SARID/data/train_features_mfcc_n_40.npy')
#
# print(file[0].shape)

# y, sr = librosa.load("../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_test/2022-11-21 23-58-00_2.06_15.94_98.1_1.776_0.353_hiv00099_60_road(concrete)_segment[16,20].mp3", sr=16000)

# import torch.nn as nn
#
# # 定义 DataEmbedding_inverted 类
# class DataEmbedding_inverted(nn.Module):
#     def __init__(self, c_in, d_model, dropout=0.1):
#         super(DataEmbedding_inverted, self).__init__()
#         self.value_embedding = nn.Linear(c_in, d_model)  # 线性层，将 c_in 映射到 d_model
#         self.dropout = nn.Dropout(p=dropout)  # Dropout 层
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # 将 x 的维度从 [Batch, Time, Variate] 重排为 [Batch, Variate, Time]
#         x = self.value_embedding(x)  # 直接嵌入 x
#         return self.dropout(x)  # 应用 Dropout
#
# # 示例输入
# x = torch.randn(1, 173, 40)  # [Batch, Time, Variate] = [2, 10, 3]
#
# # 初始化 DataEmbedding_inverted
# embedding = DataEmbedding_inverted(c_in=173, d_model=8, dropout=0.1)
#
# # 前向传播
# output = embedding(x)
# print("Output shape:", output.shape)  # 输出嵌入后的形状

# import numpy as np

# Define the adjacency matrix