import logging
import os.path

import config
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
import pywt
import spafe.features.pncc as pncc
from moviepy.editor import *
import logging
from logging import debug, error

logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s', level = logging.ERROR)

class FeaturesExtract:
    def __init__(self):
        self.dataset_path_train = '../../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_train'
        self.dataset_path_test = '../../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_test'
        self.dataset_path = '../../SARIDDATA/SARID/audio'
        self.n_mfcc = config.N_MFCC
        self.n_mel = config.N_MEL
        self.n_pncc = config.N_PNCC
        self.n_level_wavelet = config.N_LEVEL_WAVELET

        self.extract_type_fuc = {
            config.NAME_FEATURES_MFCC: self.get_new_mfcc,
            config.NAME_FEATURES_MEL: self.get_mel_spectrogram,
            config.NAME_FEATURES_PNCC: self.get_pncc,
            config.NAME_FEATURES_SPEC: self.generate_stft_spectrogram,
            config.NAME_FEATURES_WAVE: self.load_audio_with_librosa,
            config.NAME_FEATURES_WAVELET: self.get_wavelet_features
        }[config.NAME_FEATURES_PROJECT]

    def pre_processing(self, file_path):
        files = glob.glob(os.path.join(file_path,'*.mp3'))
        labels = list(map(lambda x:os.path.split(x)[-1].split('_'),files))
        Y = pd.DataFrame(labels).iloc[:,1:6]   #提取雨量，温度，湿度，气压，风速，
        Y.columns=['RAINFALL INTENSITY','TEMPERATURE','HUMIDITY','ATMOSPHERE PRESSURE','WIND SPEED']

        features = []
        for index, file_path in enumerate(tqdm(files)):

            extracted_feature = self.extract_type_fuc(file_path)
            features.append(extracted_feature)

        X = np.array(features)
        return X, Y

    def get_mel_spectrogram(self, file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512):
        """ 非新模块生成方式 """
        y, sr = librosa.load(file_path)

        # 计算梅尔谱
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mel)

        # 计算 log 梅尔谱
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return log_mel_spectrogram

    def get_mfcc(self,  file_path, mfcc_max_padding=0):
        try:
            y, sr = librosa.load(file_path)
            normalized_y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=self.n_mfcc) # Compute MFCC coefficients
            normalized_mfcc = librosa.util.normalize(mfcc)

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None

        return normalized_mfcc

    def get_new_mfcc(self, file_path):
        # noinspection PyBroadException
        try:
            y, sr = librosa.load(file_path)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=self.n_mfcc)
            normalized_mfcc = librosa.util.normalize(mfcc)

        except Exception as e:
            error(f"extract features from audio error: {e} from {file_path}")
            return None

        return normalized_mfcc

    def get_pncc(self, file_path):
        try:
            y, sr = librosa.load(file_path)
            normalized_y = librosa.util.normalize(y)
            pncc_feature = pncc.pncc(normalized_y, nfilts=50, fs=sr, num_ceps=self.n_pncc)
            normalized_pncc = librosa.util.normalize(pncc_feature)

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None

        return normalized_pncc

    @staticmethod
    def generate_stft_spectrogram(file_path, sr=22050, n_fft=2048, hop_length=512):                                  
        y, sr = librosa.load(file_path)
        normalized_y = librosa.util.normalize(y)
        spec = librosa.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)
        spec_db = librosa.amplitude_to_db(abs(spec))
        s_db_normalized = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db))

        # channel_spec = np.stack([normalized_spec] * 3, axis=0)

        return s_db_normalized

    def load_audio_with_librosa(self, file_path, target_sr=16000):

        waveform, sr = librosa.load(file_path, sr=target_sr)  # 重采样到 target_sr
        waveform = librosa.util.normalize(waveform)
        return waveform

    def get_wavelet_features(self, file_path):
        y, sr = librosa.load(file_path)

        # wavelets = mo_dwt(y, 'db4', level=1)
        wavelets = mo_dwt(y, 'haar', level=self.n_level_wavelet)

        wavelets_mfcc_feature = get_wavelet_layer_mfcc(wavelets, sr, 40)
        # wavelets = np.concatenate(wavelets)

        return librosa.util.normalize(wavelets_mfcc_feature)


def trim_mute(y, top_db=15):
    sr = 22050
    target_length = 3 * sr  # 3秒的样本数
    min_top_db = 1  # 允许的最小阈值

    # 逐步降低静音阈值直到音频长度 ≤ 3秒
    while True:
        intervals = librosa.effects.split(y, top_db=top_db)
        y_trimmed = librosa.effects.remix(y, intervals)
        current_length = len(y_trimmed)

        # 当满足长度要求或阈值触底时退出循环
        if current_length <= target_length or top_db <= min_top_db:
            break

        top_db -= 1  # 逐步减小阈值

    # 补零到目标长度（仅在不足时）
    if len(y_trimmed) < target_length:
        pad_length = target_length - len(y_trimmed)
        pad_before = pad_length // 2
        pad_after = pad_length - pad_before
        y_trimmed = np.pad(y_trimmed, (pad_before, pad_after), mode='constant')
    else:
        start = (current_length - target_length) // 2
        y_trimmed = y_trimmed[start:start + target_length]

    return y_trimmed
def split_data(data, seq_length=128):
    remainder = len(data) % seq_length
    if remainder != 0:
        # 填充到最近的 128 的倍数
        data = np.pad(data, (0, seq_length - remainder), mode='constant')

    return data.reshape(seq_length, -1)

def get_wavelet_layer_mfcc(wavelets, sr, n_mfcc):
    all_layer_mfcc_list = []
    for i, wavelet in enumerate(wavelets):
        layer_mfcc = librosa.feature.mfcc(y=wavelet, sr=sr, n_mfcc=n_mfcc)
        # if layer_mfcc.shape[1] < 173:
        #     # 如果列数不足，进行填充
        #     layer_mfcc = np.pad(layer_mfcc, ((0, 0), (0, 173 - layer_mfcc.shape[1])), mode='constant')
        # elif layer_mfcc.shape[1] > 173:
        #     # 如果列数过多，进行截断
        #     layer_mfcc = layer_mfcc[:, :173]
        all_layer_mfcc_list.append(layer_mfcc)
    return np.vstack(all_layer_mfcc_list) # shape: [(level + 1) * n_mfcc , 173]


def circular_convolve_d(h_t, v_j_1, j):
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    indices = np.mod(np.arange(N)[:, None] - 2 ** (j - 1) * l, N)
    v_p = v_j_1[indices]
    w_j = np.sum(h_t * v_p, axis=1)
    return w_j


def mo_dwt(x, filters, level):
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)

    wave_coeff = []
    v_j_1 = x

    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wave_coeff.append(w)

    wave_coeff.append(v_j_1)
    return np.vstack(wave_coeff)

if __name__ == '__main__':


    audio_utils = FeaturesExtract()

    # train
    X_train, Y_train = audio_utils.pre_processing(audio_utils.dataset_path_train)
    Y_train.to_csv(f'../data/train_labels.csv')
    np.save(f"../data/{config.NAME_TRAIN_FEATURES_FILE}", X_train)

    # test
    X_test, Y_test = audio_utils.pre_processing(audio_utils.dataset_path_test)
    Y_test.to_csv(f'../data/test_labels.csv')
    np.save(f"../data/{config.NAME_TEST_FEATURES_FILE}", X_test)