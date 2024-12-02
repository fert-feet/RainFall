from .config import *
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class MyDataSet(Dataset):
    def __init__(self,label_path,feature_path):
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
        'wave': 'path/to/wave_train_data'
    },
    'train_label_path': 'path/to/shared_train_labels',
    'test_data_paths': {
        'spec': 'path/to/spec_test_data',
        'mfcc': 'path/to/mfcc_test_data',
        'wave': 'path/to/wave_test_data'
    },
    'test_label_path': 'path/to/shared_test_labels'
}
    """
    train_data_paths = data_paths['train_data_paths']
    train_label_path = data_paths['train_label_path']

    spec_train_loader = create_dataloader(train_data_paths['spec'], train_label_path, batch_size, True, dataset_collect)
    mfcc_train_loader = create_dataloader(train_data_paths['mfcc'], train_label_path, batch_size, True, dataset_collect)
    wave_train_loader = create_dataloader(train_data_paths['wave'], train_label_path, batch_size, True, dataset_collect)

    return {
        'spec': spec_train_loader,
        'mfcc': mfcc_train_loader,
        'wave': wave_train_loader
    }

def get_test_data_loaders(data_paths, batch_size=32):
    test_data_paths = data_paths['test_data_paths']
    test_label_path = data_paths['test_label_path']

    spec_test_loader, spec_test_dataset = create_dataloader(test_data_paths['spec'], test_label_path, batch_size, False, dataset_collect)
    mfcc_test_loader, mfcc_test_dataset = create_dataloader(test_data_paths['mfcc'], test_label_path, batch_size, False, dataset_collect)
    wave_test_loader, wave_test_dataset = create_dataloader(test_data_paths['wave'], test_label_path, batch_size, False, dataset_collect)

    return {
        'spec': spec_test_loader,
        'mfcc': mfcc_test_loader,
        'wave': wave_test_loader,
        'spec_dataset': spec_test_dataset,
        'mfcc_dataset': mfcc_test_dataset,
        'wave_dataset': wave_test_dataset
    }

def dataset_collect(batch):
    features,batch_rainfall_intensities = [],[]

    for feature,batch_rainfall_intensity in batch:
        features.append(feature)
        batch_rainfall_intensities.append(batch_rainfall_intensity)
    features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
    batch_rainfall_intensities = torch.from_numpy(np.array(batch_rainfall_intensities)).type(torch.FloatTensor)

    return features,batch_rainfall_intensities
