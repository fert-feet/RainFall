import numpy as np

from utils import config
import torch
from utils.new_dataloader import train_data_loaders, test_data_loaders
from model.base_model import CoAttentionModel
from nets.baseline_training import get_lr_scheduler, set_optimizer_lr
import torch.optim as optim
from tqdm import tqdm
from model.general_net import *
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae
from sklearn.metrics import r2_score  # R square
from utils.draw import Plotter

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = config.NUM_EPOCHES
batch_size = config.BATCH_SIZE
learning_rate = config.LEARNING_RATE

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

train_data_loaders = train_data_loaders(data_paths, batch_size)
test_data_loaders = test_data_loaders(data_paths, batch_size)


model = CoAttentionModel().to(device)

Init_lr = 5e-4
Min_lr = Init_lr * 0.01
optimizer_type = "adam"
momentum = 0.9
weight_decay = 0.0003
lr_decay_type = 'cos'
nbs = 64
lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# Loss and optimizer
criterion_r = nn.SmoothL1Loss()

# Optimizer selection based on optimizer_type
optimizer = {
    'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                     weight_decay=weight_decay)
}[optimizer_type]

# Formula for obtaining a decrease in the learning rate
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, num_epochs)

R2_list = []
MAE_list = []
MSE_list = []
RMSE_list = []
train_loss_list = []
test_loss_list = []


for epoch in range(num_epochs):
    total_loss = 0
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    model.train()
    total_step = len(train_data_loaders['mfcc'])

    for i, (mfcc_data, spec_data, wave_data) in enumerate(zip(train_data_loaders['mfcc'], train_data_loaders['spec'], train_data_loaders['wave'])):
        mfcc_images = mfcc_data[0].to(device)
        mfcc_labels = mfcc_data[1].to(torch.float32).to(device)
        spec_images = spec_data[0].to(device)
        spec_labels = spec_data[1].to(torch.float32).to(device)
        wave_images = wave_data[0].to(device)
        wave_labels = wave_data[1].to(torch.float32).to(device)

        # Forward pass
        rainfall_intensity = model(mfcc_images, spec_images, wave_images)
        r_loss = criterion_r(rainfall_intensity, mfcc_labels.view([-1, 1]))
        loss = r_loss
        total_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), f'./logs/{config.NAME_MODEL_PERFORMANCE_FILE}.ckpt')

    # Test
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
    R2_list.append(R2)
    RMSE_list.append(RMSE)
    MSE_list.append(MSE)
    MAE_list.append(MAE)

    draw_tool = Plotter(labels, outputs, R2, RMSE, MSE, MAE)
    draw_tool.simply_draw()
