"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model
# from torchsummary import summary
from .general_net import *
from .paper_model.iTransformer import Model as iTransformer


# __all__ = ['Ser_Model']
class CoAttentionModel(nn.Module):
    def __init__(self):
        super(CoAttentionModel, self).__init__()

        # CNN for Spectrogram
        self.alexnet_model = ModifiedAlexNet(num_classes=1, in_ch=3, pretrained=True)

        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(9216, 128)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l

        # LSTM for MFCC
        self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,
                                 bidirectional=True)  # bidirectional = True

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        self.post_mfcc_layer = nn.Linear(88576,
                                         128)  # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm

        # Spectrogram + MFCC
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, 275)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l

        # WAV2VEC 2.0
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(
            "./model/pre_trained_model/wav2vec2")

        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, 128)  # 512 for 1 and 768 for 2

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 1)

    def forward(self, audio_mfcc, audio_spec, audio_wav):
        # audio_spec: [batch, 3, 200, 173]
        # audio_mfcc: [batch, 40, 173]
        # audio_wav: [32, 88200]

        # audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)

        # transpose
        audio_spec = audio_spec.permute(0, 1, 3, 2)
        audio_mfcc = audio_mfcc.permute(0, 2, 1)

        # spectrogram - SER_CNN
        audio_spec, output_spec_t = self.alexnet_model(audio_spec)  # [batch, 256, 6, 6], []
        audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1)  # [batch, 256, 36]

        # audio -- MFCC with BiLSTM
        audio_mfcc, _ = self.lstm_mfcc(audio_mfcc)  # [batch, 173, 512]

        audio_spec_ = torch.flatten(audio_spec, 1)  # [batch, 9216]
        audio_spec_d = self.post_spec_dropout(audio_spec_)  # [batch, 9216]
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False)  # [batch, 128]

        # + audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1)  # [batch, 88576]
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_)  # [batch, 88576]
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False)  # [batch, 128]

        # FOR WAV2VEC2.0 WEIGHTS
        spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1)  # [batch, 256]
        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)  # [batch, 256]
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d),
                                       inplace=False)  # [batch, 275]
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)  # [batch, 1, 275]
        # + audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)

        # wav2vec 2.0
        audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state  # [batch, 275, 768]
        audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav)  # [batch, 1, 768], [batch, 1, dimension of wave_model]
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1)  # [batch, 768]
        # audio_wav = torch.mean(audio_wav, dim=1)

        audio_wav_d = self.post_wav_dropout(audio_wav)  # [batch, 768]
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False)  # [batch, 768]

        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384]
        audio_att_d_1 = self.post_att_dropout(audio_att)  # [batch, 384]
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)  # [batch, 128]
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128]
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 1]


        return output_att

class CoAENetTransformerModel(nn.Module):
    def __init__(self):
        super(CoAENetTransformerModel, self).__init__()
        self.ae_net_model = AENet()
        self.transformer_model = ModifiedTransformer(1280, 5)

    def forward(self, audio_mfcc):
        audio_mfcc = audio_mfcc.permute(0, 2, 1)
        audio_mfcc = audio_mfcc.unsqueeze(1)
        audio_mfcc = audio_mfcc.repeat(1, 3, 1, 1)
        ae_output, _ = self.ae_net_model(audio_mfcc)
        output = self.transformer_model(ae_output)
        return output


class SingleTransformerModel(nn.Module):
    def __init__(self, n_features=40, n_head=5):
        super(SingleTransformerModel, self).__init__()
        self.transformer = ModifiedTransformer(n_features, n_head)

    def forward(self, audio_mfcc):
        audio_mfcc = audio_mfcc.permute(0, 2, 1)
        output = self.transformer(audio_mfcc)
        return output

class SingleWaveVec2Model(nn.Module):
    def __init__(self):
        super(SingleWaveVec2Model, self).__init__()
        self.wave_model = ModifiedWaveVec2()

    def forward(self, audio_wav):
        output = self.wave_model(audio_wav)
        return output

class SingleCNNModel(nn.Module):
    def __init__(self):
        super(SingleCNNModel, self).__init__()
        self.cnn_model = ModifiedCNN()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.cnn_model(x)
        return out

class SingleLSTMModel(nn.Module):
    def __init__(self, n_features=40):
        super(SingleLSTMModel, self).__init__()
        self.lstm_model = ModifiedLSTM(n_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.lstm_model(x)
        return out

class CoLSTMTransformerModel(nn.Module):
    def __init__(self, hidden_layer_sizes=None, n_features=40, n_head=5):
        super(CoLSTMTransformerModel, self).__init__()
        self.transformer_model = ModifiedEncoderOnlyTransformer(n_features, n_head)

        self.bi_lstm_layers = ModifiedBiLSTM(n_features, hidden_layer_sizes)

        self.avg_pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(16))  # 2d pooling input (batch_size, seq, time)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, audio_mfcc):
        audio_mfcc = audio_mfcc.permute(0, 2, 1)

        transformer_output = self.transformer_model(audio_mfcc)
        bi_lstm_output = self.bi_lstm_layers(transformer_output)

        out = self.avg_pool_layer(bi_lstm_output)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CoLSTMTransformerResidualModel(nn.Module):
    def __init__(self, hidden_layer_sizes=None, n_features=40, n_head=5):
        super(CoLSTMTransformerResidualModel, self).__init__()
        # 原始组件
        self.transformer_model = ModifiedEncoderOnlyTransformer(n_features, n_head)
        self.bi_lstm_layers = ModifiedBiLSTM(n_features, hidden_layer_sizes)

        # 新增残差相关组件
        self.residual_norm = nn.LayerNorm(128)  # 层归一化
        self.dimension_align = nn.Linear(n_features, 128)  # 维度对齐（可选）

        self.avg_pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(16))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, audio_mfcc):
        audio_mfcc = audio_mfcc.permute(0, 2, 1)

        # Transformer输出
        transformer_output = self.transformer_model(audio_mfcc)

        # LSTM处理
        bi_lstm_output = self.bi_lstm_layers(transformer_output)

        # 残差连接（核心修改）
        residual = self.dimension_align(transformer_output)  # 维度对齐
        residual_output = bi_lstm_output + residual  # 残差相加
        residual_output = self.residual_norm(residual_output)  # 层归一化

        # 后续处理
        out = self.avg_pool_layer(residual_output)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class SingleITransformerModel(nn.Module):
    def __init__(self, seq_len=173, turn_to_d_model=40, n_heads=5):
        super(SingleITransformerModel, self).__init__()
        self.transformer_model = iTransformer(seq_len, turn_to_d_model, n_heads)

    def forward(self, audio_mfcc):
        audio_mfcc = audio_mfcc.permute(0, 2, 1)[:, :, :200]
        output = self.transformer_model(audio_mfcc)
        return output



# input_m = torch.randn(1, 40, 173).cuda()
# model = CoLSTMTransformerModel(n_features=40, n_head=40).to('cuda')
# summary(model, input_m)
  