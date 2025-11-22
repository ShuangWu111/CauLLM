import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, config, data):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.pred_len = 0
        padding = config['stride']
        stride = config['stride']
        patch_len = config['patch_size']
        self.num_classes = len(data.class_names)

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            config['d_model'], patch_len, stride, padding, config['dropout'])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config['factor'], attention_dropout=config['dropout'],
                                      output_attention=False), config['d_model'], config['n_heads']),
                    config['d_model'],
                    config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation']
                ) for l in range(config['e_layers'])
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config['d_model'],dtype = torch.float32), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = config['d_model'] * \
                       int((self.seq_len - patch_len) / stride + 2)
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(config['dropout'])
        self.projection = nn.Linear(
                self.head_nf * config['enc_in'], self.num_classes, dtype=torch.float32)


    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc):
        
        dec_out = self.classification(x_enc.float(), x_mark_enc)
        return dec_out  # [B, N]