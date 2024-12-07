import os.path

import config
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
import spafe.features.pncc as pncc
from moviepy.editor import *



class FeaturesExtract:
    def __init__(self):
        self.dataset_path_train = '../../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_train'
        self.dataset_path_test = '../../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_test'
        self.dataset_path = '../../SARIDDATA/SARID/audio'
        self.n_mfcc = config.N_MFCC
        self.n_mel = config.N_MEL
        self.n_pncc = config.N_PNCC

        self.extract_type_fuc = {
            config.NAME_FEATURES_MFCC: self.get_mfcc,
            config.NAME_FEATURES_MEL: self.get_mel_spectrogram,
            config.NAME_FEATURES_PNCC: self.get_pncc,
            config.NAME_FEATURES_SPEC: self.generate_stft_spectrogram,
            config.NAME_FEATURES_WAVE: self.load_audio_with_librosa
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

        # 设置窗长和 hop size
        frame_length = int(0.025 * sr)  # 25 ms
        hop_length = int(0.010 * sr)  # 10 ms

        # 计算梅尔谱
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mel,
                                                         n_fft=frame_length, hop_length=hop_length)

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

    def generate_stft_spectrogram(self, file_path, sr=22050, n_fft=2048, hop_length=512):
        y, sr = librosa.load(file_path, sr=sr)

        spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        spec_db = spec_db[:200]
        normalized_spec = librosa.util.normalize(spec_db)

        channel_spec = np.stack([normalized_spec] * 3, axis=0)

        return channel_spec

    def load_audio_with_librosa(self, file_path, target_sr=16000):

        waveform, sr = librosa.load(file_path, sr=target_sr)  # 重采样到 target_sr
        waveform = librosa.util.normalize(waveform)
        return waveform


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