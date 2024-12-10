import librosa
import pywt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# file = np.load('/home/project/papercode/EstimatingRainFall/SARID/data/train_features_mel_n_128.npy')
#
# print(file.shape)

y, sr = librosa.load("../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_test/2022-11-21 23-58-00_2.06_15.94_98.1_1.776_0.353_hiv00099_60_road(concrete)_segment[16,20].mp3")
seq_length = 128
scaler = StandardScaler()
features = []
wavelet = pywt.wavedec(y, 'db4', level=5)
for coeff in wavelet:
    scaled_coeff = scaler.fit_transform(coeff.reshape(-1, 1)).flatten()
    features.append(scaled_coeff)
feature = np.hstack(features)

pca = PCA(n_components=seq_length)

reduced_feature = pca.fit_transform(feature)
