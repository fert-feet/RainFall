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
# Normalize audio data between -1 and 1
features = []
wavelet = pywt.wavedec(y, 'db4', level=5)
for coeff in wavelet:
    scaled_coeff = scaler.fit_transform(coeff.reshape(-1, 1)).flatten()
    features.append(scaled_coeff)
feature = np.hstack(features)
# features = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
# if len(features) < seq_length:
#             features = np.pad(features, (0, seq_length - len(features)), mode='constant')
# else:
#     features = features[:seq_length]
pca = PCA(n_components=12)
# feature = feature.reshape(-1, 1)
feature = torch.randn(230, 25800)
# print(feature.shape)
reduced_feature = pca.fit_transform(feature)
print(len(reduced_feature))
print(len(y))
