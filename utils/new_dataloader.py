from .config import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MyDataSet(Dataset):
    def __init__(self, feature_path, label_path):
        super(MyDataSet, self).__init__()
        self.label = pd.read_csv(label_path)
        self.feature = np.load(feature_path)
        self.length = self.label.shape[0]
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature_item = self.feature[index]
        rainfall_intensity = self.label.iloc[index]['RAINFALL INTENSITY']
        return feature_item,rainfall_intensity

def create_dataloader(data_path, label_path, batch_size, shuffle, collate_fn):
    dataset = MyDataSet(data_path, label_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader, dataset

def get_train_data_loaders(data_paths, batch_size=32):
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

def get_test_data_loaders(data_paths, batch_size=32):
    test_data_paths = data_paths['test_data_paths']
    test_label_path = data_paths['test_label_path']

    test_loaders = {}
    for feature_name, feature_path in test_data_paths.items():
        if feature_path:  # Check if the feature path is not empty
            test_loader, test_dataset = create_dataloader(feature_path, test_label_path, batch_size, False, dataset_collect)
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
