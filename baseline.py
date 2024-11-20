import numpy as np

from model.Base_Model import BaseModel
from utils.dataloader import USRADataset,USRADataset_collate
from torch.utils.data import DataLoader
from nets.baseline_training import get_lr_scheduler, set_optimizer_lr
import torch.optim as optim
from tqdm import tqdm
from SARID.model.general_net import *
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae
from sklearn.metrics import r2_score  # R square
from utils.draw import result_show
from utils import config


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Hyper parameters
num_epochs = config.NUM_EPOCHES
batch_size = config.BATCH_SIZE
learning_rate = config.LEARNING_RATE

# train/test data path
mfcc_train_features_path = f'./data/{config.NAME_MFCC_TRAIN_FEATURES_FILE}.npy'
mfcc_test_features_path = f'./data/{config.NAME_MFCC_TEST_FEATURES_FILE}.npy'
mel_train_features_path = f'./data/{config.NAME_MEL_TRAIN_FEATURES_FILE}.npy'
mel_test_features_path = f'./data/{config.NAME_MEL_TEST_FEATURES_FILE}.npy'

train_labels_path = f'./data/{config.NAME_TRAIN_LABEL_FILE}.csv'
test_labels_path = f'./data/{config.NAME_TEST_LABEL_FILE}.csv'

mfcc_train_dataset = USRADataset(train_labels_path, mfcc_train_features_path)
mfcc_val_dataset = USRADataset(test_labels_path, mfcc_test_features_path)
mel_train_dataset = USRADataset(train_labels_path, mel_train_features_path)
mel_val_dataset = USRADataset(test_labels_path, mel_test_features_path)

mel_train_loader = DataLoader(mel_train_dataset, shuffle=True, batch_size=batch_size,
                              collate_fn=USRADataset_collate)
mfcc_train_loader = DataLoader(mfcc_train_dataset, shuffle=True, batch_size=batch_size,
                              collate_fn=USRADataset_collate)
mel_test_loader = DataLoader(mel_val_dataset, shuffle=False, batch_size=batch_size,
                         collate_fn=USRADataset_collate)
mfcc_test_loader = DataLoader(mfcc_val_dataset, shuffle=False, batch_size=batch_size,
                         collate_fn=USRADataset_collate)

# For the data setting and model training:
# Please notice that the current code is for the paper settings, but due to the different features dimensions and model structure,
# you need to adjust the feature dimension to make sure that the code can be run correctly
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            BaseCNN_Conv(173, 128, kernel_size=3, padding=2, dilation=1))
        self.layer2 = nn.Sequential(
            BaseCNN_Conv(128, 256, kernel_size=3, padding=2, dilation=1))
        self.layer3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).squeeze()
        out = self.fc1(out)
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.LSTM(input_size=1025, hidden_size=256))
        self.layer2 = nn.Sequential(
            nn.LSTM(input_size=256, hidden_size=256))
        self.layer3 = nn.Sequential(
            nn.Linear(256,512),nn.ReLU(),nn.BatchNorm1d(173))
        self.linear = nn.Sequential(
            nn.Linear(512, 128),nn.AdaptiveAvgPool1d(1))
        self.fc1 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(173, 1)

    def forward(self, x):
        out1, state1 = self.layer1(x)
        out2, state2 = self.layer2(out1)
        out = self.layer3(out2)
        out = self.linear(out).squeeze()
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.n_feature = config.N_MFCC if config.NAME_FEATURES_PROJECT == config.NAME_FEATURES_MFCC else config.N_MEL
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feature, nhead=8, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.layer1 = nn.Sequential(
            self.transformer_encoder)
        self.layer4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(16)) # 2d pooling input (batch_size, seq, time)
        self.fc1 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(2, 0, 1) # (B, F, T) -> (T, B, F)
        out = self.layer1(x)
        out = out.permute(1, 0, 2) # (T, B, F) -> (B, T, F) or (B, F, T)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        rainfall_intensity = self.fc3(out)
        return rainfall_intensity


model = BaseModel().to(device)
# -------------------------------------------------------------------#
#   Determine the current batch_size and adaptively adjust the learning rate
# -------------------------------------------------------------------#
Init_lr             = 5e-4
Min_lr              = Init_lr * 0.01
optimizer_type      = "adam"
momentum            = 0.9
weight_decay        = 0.0003
lr_decay_type       = 'cos'
nbs = 64
lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# Loss and optimizer
criterion_r = nn.SmoothL1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
R2_max = 0.6
R2_list = []
MAE_list = []
MSE_list = []
RMSE_list = []
total_step = len(mel_train_loader)
# # ---------------------------------------#
# #   Optimizer selection based on optimizer_type
# # ---------------------------------------#
optimizer = {
    'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                     weight_decay=weight_decay)
}[optimizer_type]

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs*batch_size)

#
# # ---------------------------------------#
# #   Formula for obtaining a decrease in the learning rate
# # ---------------------------------------#
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, num_epochs)

train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    total_loss = 0
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    model.train()
    # size = len(mel_train_loader)
    for i, (mfcc_data, mel_data) in enumerate(zip(mfcc_train_loader, mel_train_loader)):
        mfcc_images = mfcc_data[0].to(device)
        mfcc_labels = mfcc_data[1].to(torch.float32).to(device)
        mel_images = mel_data[0].to(device)
        mel_labels = mel_data[1].to(torch.float32).to(device)

        # Forward pass
        rainfall_intensity = model(mfcc_images, mel_images)
        r_loss = criterion_r(rainfall_intensity, mfcc_labels.view([-1, 1]))
        loss = r_loss
        total_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    # train_loss_list.append(total_loss / size)
    torch.save(model.state_dict(), f'./logs/{config.NAME_MODEL_PERFORMANCE_FILE}.ckpt')

    # Test
    model.load_state_dict(torch.load(f'./logs/{config.NAME_MODEL_PERFORMANCE_FILE}.ckpt'))
    model.eval()

    mfcc_acoustic_feature = mfcc_val_dataset.feature
    mel_acoustic_feature = mel_val_dataset.feature
    outputs = []
    step = 16
    with torch.no_grad():
        # split input into slice
        mfcc_acoustic_feature = torch.tensor(mfcc_acoustic_feature).cuda()
        mel_acoustic_feature = torch.tensor(mel_acoustic_feature).cuda()
        # TODO change "acoustic_feature.shape[0]" to an global number -> length of test features
        for index in tqdm(range(0,mfcc_acoustic_feature.shape[0],step)):
            if index ==0:
                step_mfcc_acoustic_feature = mfcc_acoustic_feature[index:step].to(torch.float)
                step_mel_acoustic_feature = mel_acoustic_feature[index:step].to(torch.float)
                rainfall_intensity = model(step_mfcc_acoustic_feature, step_mel_acoustic_feature)
                outputs = rainfall_intensity
            elif index>0 and index != mfcc_acoustic_feature.shape[0]-mfcc_acoustic_feature.shape[0]%step:
                step_mfcc_acoustic_feature = mfcc_acoustic_feature[index:index+step].to(torch.float)
                step_mel_acoustic_feature = mel_acoustic_feature[index:index+step].to(torch.float)
                rainfall_intensity = model(step_mfcc_acoustic_feature, step_mel_acoustic_feature)
                outputs = torch.cat((outputs,rainfall_intensity))
            elif index == mfcc_acoustic_feature.shape[0]-mfcc_acoustic_feature.shape[0]%step:
                step_mfcc_acoustic_feature = mfcc_acoustic_feature[index:mfcc_acoustic_feature.shape[0]].to(torch.float)
                step_mel_acoustic_feature = mel_acoustic_feature[index:mel_acoustic_feature.shape[0]].to(torch.float)
                rainfall_intensity = model(step_mfcc_acoustic_feature, step_mel_acoustic_feature)
                outputs = torch.cat((outputs, rainfall_intensity))
    outputs = np.array(outputs.squeeze().cpu(),dtype=float)
    labels = mfcc_val_dataset.label['RAINFALL INTENSITY'].to_numpy()
    MSE = mean_squared_error(labels, outputs)
    RMSE = np.sqrt(mean_squared_error(labels, outputs))
    MAE = mean_absolute_error(labels, outputs)
    R2 = r2_score(labels, outputs)
    print('R2_Value = {}'.format(R2))
    R2_list.append(R2)
    RMSE_list.append(RMSE)
    MSE_list.append(MSE)
    MAE_list.append(MAE)

    draw_tool = result_show(labels, outputs, R2, RMSE, MSE, MAE)
    draw_tool.simply_draw()
