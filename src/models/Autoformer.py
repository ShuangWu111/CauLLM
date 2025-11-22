# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, config, data):
        super(Autoformer, self).__init__()
        
        self.pred_len = 0
        self.output_attention = config['output_attention']
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.num_classes = len(data.class_names)
        


        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                           config['dropout'])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, config['factor'], attention_dropout=config['dropout'],
                                        output_attention=config['output_attention']),
                        config['d_model'], config['n_heads']),
                    config['d_model'],
                    config['d_ff'],
                    moving_avg=config['moving_avg'],
                    dropout=config['dropout'],
                    activation=config['activation']
                ) for l in range(config['e_layers'])
            ],
            norm_layer=my_Layernorm(config['d_model'])
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(config['dropout'])
        self.projection = nn.Linear(
                config['d_model'] * self.seq_len, self.num_classes, dtype=torch.float32)

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    def forward(self, x_enc, x_mark_enc):
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out
        

        