import numpy as np

from utils import config
import torch
from utils.new_dataloader import get_train_data_loaders, get_test_data_loaders
from model.base_model import SingleLSTMModel, SingleTransformerModel, CoAENetTransformerModel, SingleITransformerModel
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
single_feature = config.NAME_FEATURES_PROJECT

data_paths = {
    'train_data_paths': {
        'spec': f'./data/{config.NAME_SPEC_TRAIN_FEATURES_FILE}.npy',
        'mfcc': f'./data/{config.NAME_MFCC_TRAIN_FEATURES_FILE}.npy',
        'wave': f'./data/{config.NAME_WAVE_TRAIN_FEATURES_FILE}.npy',
        'mel': f'./data/{config.NAME_MEL_TRAIN_FEATURES_FILE}.npy',
        'wavelet': f'./data/{config.NAME_WAVELET_TRAIN_FEATURES_FILE}.npy'
    },
    'train_label_path': './data/train_labels.csv',
    'test_data_paths': {
        'spec': f'./data/{config.NAME_SPEC_TEST_FEATURES_FILE}.npy',
        'mfcc': f'./data/{config.NAME_MFCC_TEST_FEATURES_FILE}.npy',
        'wave': f'./data/{config.NAME_WAVE_TEST_FEATURES_FILE}.npy',
        'mel': f'./data/{config.NAME_MEL_TEST_FEATURES_FILE}.npy',
        'wavelet': f'./data/{config.NAME_WAVELET_TEST_FEATURES_FILE}.npy'
    },
    'test_label_path': './data/test_labels.csv'
}

train_data_loaders = get_train_data_loaders(data_paths, batch_size)
test_data_loaders = get_test_data_loaders(data_paths, batch_size)

feature_train_loader = train_data_loaders[single_feature]
feature_test_dataset= test_data_loaders[single_feature + '_dataset']

model = SingleITransformerModel(turn_to_d_model=256).to(device)




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
    total_step = len(feature_train_loader)

    for i, (train_loader) in enumerate(feature_train_loader):
        images = train_loader[0].to(device)
        targets = train_loader[1].to(torch.float32).to(device)


        # Forward pass
        rainfall_intensity = model(images)
        r_loss = criterion_r(rainfall_intensity, targets.view([-1, 1]))
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

    acoustic_feature = feature_test_dataset.feature

    outputs = []
    step = 16
    with torch.no_grad():
        acoustic_feature = torch.tensor(acoustic_feature).cuda()

        for index in tqdm(range(0, acoustic_feature.shape[0], step)):
            step_acoustic_feature = acoustic_feature[index:index + step].to(torch.float)
            rainfall_intensity = model(step_acoustic_feature)
            outputs = torch.cat((outputs, rainfall_intensity)) if index > 0 else rainfall_intensity

    outputs = np.array(outputs.squeeze().cpu(), dtype=float)
    labels = feature_test_dataset.label['RAINFALL INTENSITY'].to_numpy()
    MSE = mean_squared_error(labels, outputs)
    R_MSE = np.sqrt(mean_squared_error(labels, outputs))
    MAE = mean_absolute_error(labels, outputs)
    R2 = r2_score(labels, outputs)
    print('R2_Value = {}'.format(R2))
    print('MSE_Value = {}'.format(MSE))
    print('R_MSE = {}'.format(R_MSE))
    R2_list.append(R2)
    RMSE_list.append(R_MSE)
    MSE_list.append(MSE)
    MAE_list.append(MAE)

    if epoch + 1 == num_epochs:
        draw_tool = Plotter(labels, outputs, R2_list, RMSE_list, MSE_list, MAE_list, single_feature)
        draw_tool.simply_draw()
        draw_tool.plot_outputs_scatter()
        draw_tool.plot_outputs_vs_labels()
