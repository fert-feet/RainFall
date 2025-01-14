import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted
# from torchsummary import summary
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, seq_len=173, turn_to_d_model=40, n_heads=5):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = 1
        self.output_attention = False
        self.use_norm = False
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, turn_to_d_model, "fixed", "h",
                                                    0.1)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1, output_attention=False), d_model=turn_to_d_model, n_heads=n_heads),
                    d_model=turn_to_d_model,
                    d_ff=512,
                    dropout=0.1,
                ) for l in range(4)
            ],
            norm_layer=torch.nn.LayerNorm(turn_to_d_model)
        )
        self.avg_pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(16)
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.projector = nn.Linear(turn_to_d_model, 1, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        out = self.avg_pool_layer(enc_out.permute(0, 2, 1))
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        # B N E -> B N S -> B S N
        # dec_out = self.projector(enc_out).permute(0, 2, 1)  # filter the covariates

        return out

# X = torch.randn(1, 173, 40).cuda()
# model = Model(turn_to_d_model=173).cuda()
# summary(model, X)
# print(model(X).shape)