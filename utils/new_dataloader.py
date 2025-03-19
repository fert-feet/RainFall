import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import librosa
import glob


class DataSetFromFiledFeatures(Dataset):
    def __init__(self, feature_path, label_path):
        super(DataSetFromFiledFeatures, self).__init__()
        self.label = pd.read_csv(label_path)
        self.feature = np.load(feature_path)
        self.length = self.label.shape[0]
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature_item = self.feature[index]
        rainfall_intensity = self.label.iloc[index]['RAINFALL INTENSITY']
        return feature_item,rainfall_intensity


class DataSetFromOriginData(Dataset):
    def __init__(self, data_path):
        super(DataSetFromOriginData, self).__init__()

        self.files = glob.glob(os.path.join(data_path, '*.mp3'))
        self.labels = list(map(lambda x: os.path.split(x)[-1].split('_'), self.files))

        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        audio_file_paths = self.files[index]
        rainfall_intensity = self.labels[index][5]

        # 加载音频文件并提取特征
        y, sr = librosa.load(audio_file_paths)

        return y, rainfall_intensity


def create_dataloader(data_path, label_path, batch_size, shuffle, collate_fn):
    dataset = DataSetFromFiledFeatures(data_path, label_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader, dataset

def create_origin_dataloader(data_paths, batch_size, shuffle):
    dataset = DataSetFromOriginData(data_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

def get_train_data_loaders(data_paths, is_use_origin_data=False, batch_size=32):
    """
    根据is_use_origin_data参数选择从源数据提取特征还是从已提取的特征文件中生成训练数据加载器

    Args:
        data_paths: 数据路径字典
        is_use_origin_data: 是否使用原始数据，默认为False
        batch_size: 批次大小，默认为32

    Returns:
        train_loaders: 训练数据加载器字典
    """
    if is_use_origin_data:
        return origin_train_data_loaders(data_paths, batch_size)
    else:
        return train_data_loaders(data_paths, batch_size)

def get_test_data_loaders(data_paths, is_use_origin_data=False, batch_size=32):
    """
    根据is_use_origin_data参数选择从源数据提取特征还是从已提取的特征文件中生成测试数据加载器

    Args:
        data_paths: 数据路径字典
        is_use_origin_data: 是否使用原始数据，默认为False
        batch_size: 批次大小，默认为32

    Returns:
        test_loaders: 测试数据加载器字典
    """
    if is_use_origin_data:
        return origin_test_data_loaders(data_paths, batch_size)
    else:
        return test_data_loaders(data_paths, batch_size)

def train_data_loaders(data_paths, batch_size=32):
    """
    data_paths = {
    'train_data_paths': {
        'spec': 'path/to/spec_train_data',
        'mfcc': 'path/to/mfcc_train_data',
        'wave': 'path/to/wave_train_data',
        'mel': 'path/to/mel_train_data',
    },
    'train_label_path': 'path/to/shared_train_labels',
    'test_data_paths': {
        'spec': 'path/to/spec_test_data',
        'mfcc': 'path/to/mfcc_test_data',
        'wave': 'path/to/wave_test_data',
        'mel': 'path/to/mel_test_data'
    },
    'test_label_path': 'path/to/shared_test_labels'
}
    """
    train_data_paths = data_paths['train_data_paths']
    train_label_path = data_paths['train_label_path']

    train_loaders = {}
    for feature_name, feature_path in train_data_paths.items():
        if feature_path:  # Check if the feature path is not empty
            train_loader, _ = create_dataloader(feature_path, train_label_path, batch_size, True, dataset_collect)
            train_loaders[feature_name] = train_loader

    return train_loaders

def test_data_loaders(data_paths, batch_size=32):
    test_data_paths = data_paths['test_data_paths']
    test_label_path = data_paths['test_label_path']

    test_loaders = {}
    for feature_name, feature_path in test_data_paths.items():
        if feature_path:  # Check if the feature path is not empty
            test_loader, test_dataset = create_dataloader(feature_path, test_label_path, batch_size, False, dataset_collect)
            test_loaders[feature_name] = test_loader
            test_loaders[f'{feature_name}_dataset'] = test_dataset

    return test_loaders

def origin_train_data_loaders(data_paths, batch_size=32):
    """
    从原始数据中提取特征并生成训练数据加载器

    Args:
        data_paths: 数据路径字典，包含 'train_path' 键
        batch_size: 批次大小，默认为32

    Returns:
        train_loaders: 训练数据加载器字典
    """
    train_path = data_paths.get('train_data_paths')

    train_loaders = {}

    for feature_name, feature_path in train_path.items():
        if feature_path:
            train_loader, _ = create_origin_dataloader(feature_path, batch_size, True)
            train_loaders[feature_name] = train_loader

    return train_loaders

def origin_test_data_loaders(data_paths, batch_size=32):
    """
    从原始数据中提取特征并生成测试数据加载器

    Args:
        data_paths: 数据路径字典，包含 'test_path' 键
        batch_size: 批次大小，默认为32

    Returns:
        test_loaders: 测试数据加载器字典
    """
    test_path = data_paths.get('test_data_paths')

    test_loaders = {}

    for feature_name, feature_path in test_path.items():
        if feature_path:
            test_loader, test_dataset = create_origin_dataloader(feature_path, batch_size, False)
            test_loaders[feature_name] = test_loader
            test_loaders[f'{feature_name}_dataset'] = test_dataset

    return test_loaders

def dataset_collect(batch):
    features,batch_rainfall_intensities = [],[]

    for feature,batch_rainfall_intensity in batch:
        features.append(feature)
        batch_rainfall_intensities.append(batch_rainfall_intensity)
    features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
    batch_rainfall_intensities = torch.from_numpy(np.array(batch_rainfall_intensities)).type(torch.FloatTensor)

    return features,batch_rainfall_intensities
