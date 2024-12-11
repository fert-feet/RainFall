import librosa
import pywt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# file = np.load('/home/project/papercode/EstimatingRainFall/SARID/data/train_features_wavelet_n_5.npy')
#
# print(file.shape)

y, sr = librosa.load("../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_test/2022-11-21 23-58-00_2.06_15.94_98.1_1.776_0.353_hiv00099_60_road(concrete)_segment[16,20].mp3")
# y, sr = librosa.load("/home/fengjf/test.mp3")
seq_length = 128
scaler = StandardScaler()
# Normalize audio data between -1 and 1
features = []
# wavelets = pywt.wavedec(y, 'db4', mode='symmetric', level=3)
# for coeff in wavelet:
#     scaled_coeff = scaler.fit_transform(coeff.reshape(-1, 1)).flatten()
#     features.append(scaled_coeff)
# feature = np.hstack(features)
# features = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
# if len(features) < seq_length:
#             features = np.pad(features, (0, seq_length - len(features)), mode='constant')
# else:
#     features = features[:seq_length]
pca = PCA(n_components=12)
# feature = feature.reshape(-1, 1)
# feature = torch.randn(230, 25800)
# print(feature.shape)
# reduced_feature = pca.fit_transform(feature)


# 3. 绘制原始信号和分解后的系数
def plot_modwt_results(signal, coeffs, sr):
    plt.figure(figsize=(12, 8))

    # 绘制原始信号
    plt.subplot(len(coeffs) + 1, 1, 1)
    plt.plot(np.arange(len(signal)) / sr, signal)
    plt.title("Original Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    # 绘制分解后的系数
    for i, coeff in enumerate(coeffs):
        plt.subplot(len(coeffs) + 1, 1, i + 2)
        plt.plot(np.arange(len(coeff)) / sr, coeff)
        plt.title(f"Detail Coefficients (Level {i + 1})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j

def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)
#
wavelets = modwt(y, 'db4', 1)
mfcc = librosa.feature.mfcc(y=wavelets[0], sr=sr, n_mfcc=40)
plot_modwt_results(y, wavelets, sr)

# a = np.random.randn(40, 173)
# b = [a, a, a, a]
# print(np.vstack(b).shape)