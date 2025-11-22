import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, config,data):
        super(Transformer, self).__init__()
        self.pred_len = 0
        self.output_attention = config['output_attention']
        self.num_classes = len(data.class_names)
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        # Embedding
        self.enc_embedding = DataEmbedding(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                           config['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config['factor'], attention_dropout=config['dropout'],
                                      output_attention=config['output_attention']), config['d_model'], config['n_heads']),
                    config['d_model'],
                    config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation']
                ) for l in range(config['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(config['d_model'],dtype=torch.float32)
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(config['dropout'])
        self.projection = nn.Linear(config['d_model'] * self.seq_len, self.num_classes, dtype=torch.float32)
    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    def forward(self, x_enc, x_mark_enc):

        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]
        
