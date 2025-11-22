import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, config, data):
        super(FEDformer, self).__init__()
        self.version = config['version']
        self.mode_select = config['mode_select']
        self.modes = config['modes']
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']
        self.L = config['L']
        self.feat_dim = data.feature_df.shape[1]


        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.feat_dim, self.d_model, config['embed'], config['freq'],
                                                  config['dropout'])
        

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=self.L, base=config['base'])
        else:
            encoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            n_heads=config['n_heads'],
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)

        # Encoder

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model, config['n_heads']),

                    self.d_model,
                    config['d_ff'],
                    moving_avg=config['moving_avg'],
                    dropout=config['dropout'],
                    activation=config['activation']
                ) for l in range(config['e_layers'])
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(config['dropout'])
        self.projection = nn.Linear(config['d_model'] * self.seq_len, self.num_classes, dtype=torch.float32)
    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]




    
