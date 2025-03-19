import librosa
import pywt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
from torch import nn

#
# file = np.load('/home/project/papercode/EstimatingRainFall/SARID/data/train_features_mfcc_n_40.npy')
#
# print(file[0].shape)
audio_name = "audio_without_background_split_nocoverage_train/2022-10-26 23-48-00_0.06_15.57_98.15_1.736_0_hiv00080_60_water_segment[4,8].mp3"
y, sr = librosa.load("../SARIDDATA/SARID/split/" + audio_name)

print(y.shape)
