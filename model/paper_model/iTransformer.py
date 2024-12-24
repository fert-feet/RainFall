import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self):
        super(Model, self).__init__()
        # Embedding
        # self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor=5, attention_dropout=0.1,
                                      output_attention=False), d_model=40, n_heads=5),
                    d_model=40,
                ) for l in range(4)
            ],
            norm_layer=torch.nn.LayerNorm(40)
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(20, 1)


    def forward(self, x_enc):
        # enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(x_enc, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output