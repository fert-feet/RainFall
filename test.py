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
audio_name = "audio_without_background_split_nocoverage_test/2022-11-21 23-56-00_2.82_15.885_98.1_1.769_0.06_hiv00099_60_road(concrete)_cut[0,54]_segment[40,44].mp3"
y, sr = librosa.load("../SARIDDATA/SARID/split/" + audio_name, sr=16000)
print(len(y))
yt, index = librosa.effects.trim(y, top_db=10)
print(index)

fig, axs = plt.subplots(nrows=2, ncols=1)
librosa.display.waveshow(y, sr=sr, ax=axs[0])
librosa.display.waveshow(yt, sr=sr, ax=axs[1])

plt.show()