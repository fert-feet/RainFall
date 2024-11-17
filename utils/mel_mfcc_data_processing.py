
import config

import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from moviepy.editor import *
import librosa
import librosa.display



class utils_audio():
    def __init__(self):
        self.dataset_path_train = '../../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_train'
        self.dataset_path_test = '../../SARIDDATA/SARID/split/audio_without_background_split_nocoverage_test'
        self.dataset_path = '../../SARIDDATA/SARID/audio'
        self.n_mfcc = config.N_MFCC
        self.n_mel = config.N_MEL
        self.feature_type = config.NAME_FEATURES_PROJECT

    def get_mfcc(self,file_path, mfcc_max_padding=0):
        try:
            y, sr = librosa.load(file_path)
            normalized_y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=self.n_mfcc) # Compute MFCC coefficients
            normalized_mfcc = librosa.util.normalize(mfcc)

        except Exception as e:
            print("Error parsing wavefile: ", e)
            return None
        return normalized_mfcc

    def pre_processing(self, file_path):
        files = glob.glob(os.path.join(file_path,'*.mp3'))
        labels = list(map(lambda x:os.path.split(x)[-1].split('_'),files))
        Y = pd.DataFrame(labels).iloc[:,1:6]   #提取雨量，温度，湿度，气压，风速，
        Y.columns=['RAINFALL INTENSITY','TEMPERATURE','HUMIDITY','ATMOSPHERE PRESSURE','WIND SPEED']

        features = []
        for index, file_path in enumerate(tqdm(files)):
            extracted_feature = self.get_mfcc(file_path)
            features.append(extracted_feature)

        X = np.array(features)
        return X, Y
    def get_mel_spectrogram(self,file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512):
        pass


if __name__ == '__main__':


    audio_utils = utils_audio()

    # train
    X_train, Y_train = audio_utils.pre_processing(audio_utils.dataset_path_train)
    Y_train.to_csv(f'../data/{config.NAME_TRAIN_LABEL_FILE}.csv')
    np.save(f"../data/{config.NAME_TRAIN_FEATURES_FILE}", X_train)

    # test
    X_test, Y_test = audio_utils.pre_processing(audio_utils.dataset_path_test)
    Y_test.to_csv(f'../data/{config.NAME_TEST_LABEL_FILE}.csv')
    np.save(f"../data/{config.NAME_TEST_FEATURES_FILE}", X_test)