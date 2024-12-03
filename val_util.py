from utils import config
import torch
from utils.new_dataloader import get_train_data_loaders, get_test_data_loaders
from model.base_model import CoAttentionModel
from nets.baseline_training import get_lr_scheduler, set_optimizer_lr
import torch.optim as optim
from tqdm import tqdm
from model.general_net import *
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae
from sklearn.metrics import r2_score  # R square
from utils.draw import Plotter
import numpy as np

batch_size = config.BATCH_SIZE

R2_list = []
MAE_list = []
MSE_list = []
RMSE_list = []
train_loss_list = []
test_loss_list = []


data_paths = {
    'train_data_paths': {
        'spec': f'./data/{config.NAME_SPEC_TRAIN_FEATURES_FILE}.npy',
        'mfcc': f'./data/{config.NAME_MFCC_TRAIN_FEATURES_FILE}.npy',
        'wave': f'./data/{config.NAME_WAVE_TRAIN_FEATURES_FILE}.npy'
    },
    'train_label_path': './data/train_labels.csv',
    'test_data_paths': {
        'spec': f'./data/{config.NAME_SPEC_TEST_FEATURES_FILE}.npy',
        'mfcc': f'./data/{config.NAME_MFCC_TEST_FEATURES_FILE}.npy',
        'wave': f'./data/{config.NAME_WAVE_TEST_FEATURES_FILE}.npy'
    },
    'test_label_path': './data/test_labels.csv'
}

train_data_loaders = get_train_data_loaders(data_paths, batch_size)
test_data_loaders = get_test_data_loaders(data_paths, batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = CoAttentionModel().to(device)

model.load_state_dict(torch.load(f'./logs/{config.NAME_MODEL_PERFORMANCE_FILE}.ckpt'))
model.eval()

mfcc_acoustic_feature = test_data_loaders['mfcc_dataset'].feature
spec_acoustic_feature = test_data_loaders['spec_dataset'].feature
wave_acoustic_feature = test_data_loaders['wave_dataset'].feature

outputs = []
step = 16
with torch.no_grad():
    mfcc_acoustic_feature = torch.tensor(mfcc_acoustic_feature).cuda()
    spec_acoustic_feature = torch.tensor(spec_acoustic_feature).cuda()
    wave_acoustic_feature = torch.tensor(wave_acoustic_feature).cuda()

    for index in tqdm(range(0, mfcc_acoustic_feature.shape[0], step)):
        step_mfcc_acoustic_feature = mfcc_acoustic_feature[index:index+step].to(torch.float)
        step_spec_acoustic_feature = spec_acoustic_feature[index:index+step].to(torch.float)
        step_wave_acoustic_feature = wave_acoustic_feature[index:index+step].to(torch.float)
        rainfall_intensity = model(step_mfcc_acoustic_feature, step_spec_acoustic_feature, step_wave_acoustic_feature)
        outputs = torch.cat((outputs, rainfall_intensity)) if index > 0 else rainfall_intensity

outputs = np.array(outputs.squeeze().cpu(), dtype=float)
labels = test_data_loaders['mfcc_dataset'].label['RAINFALL INTENSITY'].to_numpy()
MSE = mean_squared_error(labels, outputs)
RMSE = np.sqrt(mean_squared_error(labels, outputs))
MAE = mean_absolute_error(labels, outputs)
R2 = r2_score(labels, outputs)
print('R2_Value = {}'.format(R2))
# R2_list.append(R2)
# RMSE_list.append(RMSE)
# MSE_list.append(MSE)
# MAE_list.append(MAE)
#
# draw_tool = result_show(labels, outputs, R2, RMSE, MSE, MAE)
# draw_tool.simply_draw()
