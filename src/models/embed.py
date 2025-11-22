import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        y = self.pe[:, :x.size(1)]
        return y


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.relu = nn.ReLU()
        self.batchnormalize = nn.BatchNorm1d(d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.relu(self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2))
        return x
# ISRUC dataset configuration
# class CustomLinearTransform(nn.Module):
#     def __init__(self):
#         super(CustomLinearTransform, self).__init__()
#         # Define four different linear transformation layers
#         self.linear1 = nn.Linear(48, 30)  # 48 -> 30
#         self.linear2 = nn.Linear(16, 30)  # 16 -> 30
#         self.linear3 = nn.Linear(8, 30)   # 8  -> 30
#         self.linear4 = nn.Linear(8, 30)   # 8  -> 30
#         device = torch.device('cuda:{}'.format(0))
#         self.linear1.to(device = device)
#         self.linear2.to(device = device)
#         self.linear3.to(device = device)
#         self.linear4.to(device = device)


#     def forward(self, x):
#         # x: (16, 376, 80)

#         # Slice by specified indices
#         x1 = x[:, :, 0:48]   # (16, 376, 48)
#         x2 = x[:, :, 48:64]  # (16, 376, 16)
#         x3 = x[:, :, 64:72]  # (16, 376, 8)
#         x4 = x[:, :, 72:80]  # (16, 376, 8)

#         # Apply linear transformations
#         x1 = self.linear1(x1)  # (16, 376, 30)
#         x2 = self.linear2(x2)  # (16, 376, 30)
#         x3 = self.linear3(x3)  # (16, 376, 30)
#         x4 = self.linear4(x4)  # (16, 376, 30)

#         # Stack into (16, 376, 4, 30)
#         x_out = torch.stack([x1, x2, x3, x4], dim=2)  # (16, 376, 4, 30)

#         return x_out

# Racket, BasicMotions configuration
class CustomLinearTransform(nn.Module):
   def __init__(self):
       super(CustomLinearTransform, self).__init__()
       # Define 2 different linear transformation layers
       self.linear1 = nn.Linear(24, 24)
       self.linear2 = nn.Linear(24, 24)
       device = torch.device('cuda:{}'.format(1))
       self.linear1.to(device = device)
       self.linear2.to(device = device)


   def forward(self, x):
       # x: (16, 376, 48)

       # Slice by specified indices
       x1 = x[:, :, 0:24]   # (16, 376, 24)
       x2 = x[:, :, 24:48]  # (16, 376, 24)


       # Apply linear transformations
       x1 = self.linear1(x1)  # (16, 376, 24)
       x2 = self.linear2(x2)  # (16, 376, 24)


       # Stack into (16, 376, 2, 24)
       x_out = torch.stack([x1, x2], dim=2)  # (16, 376, 2, 24)

       return x_out
    
       
class CausalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(CausalEmbedding, self).__init__()
        self.transform = CustomLinearTransform()
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_model = d_model
        self.conv1d = nn.Conv1d(in_channels=48, out_channels=self.d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='circular', bias=False)
        self.conv1d_isruc = nn.Conv1d(in_channels=120, out_channels=self.d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='circular', bias=False)
        self.device = torch.device('cuda:{}'.format(1))
        self.conv1d = self.conv1d.to(self.device)
        self.conv1d_isruc = self.conv1d_isruc.to(self.device)
        self.ismul = True  # Whether multimodal or not
    def forward(self, x, weighted_matrix):
        device = torch.device('cuda:{}'.format(1))
        batchsize, l, m = x.shape
        if self.ismul:  # Multimodal case
            y = self.transform(x)  # Shape: (16, 376, 4, 30) or (16, 376, 2, 24)
            #matrix_tensor = nn.Parameter(torch.randn(batchsize, 376, 30, 30))
            matrix_tensor = nn.Parameter(torch.randn(batchsize, l, 24, 24))
            y = torch.einsum('bqkc,bqcl->bqkl', y.to(device), matrix_tensor.to(device))
            
            expanded_weighted_matrix = weighted_matrix.unsqueeze(1) 
            transposed_weighted_matrix = expanded_weighted_matrix.permute(0,1,3,2)
            z = torch.matmul(transposed_weighted_matrix.double().to(device), y.double().to(device))
            # if ind != -2:   
            #     weighted_matrix_save = weighted_matrix[ind].cpu().detach().numpy()
            #     np.save("matrix_isruc.npy", weighted_matrix_save)
            #     z_attention = z[ind].cpu().detach().numpy()
            #     np.save("attention_isruc.npy", z_attention)
            a = z.shape[0]
            b = z.shape[1]
            z = z.view(a,b,-1)
            
            z = self.conv1d(z.permute(0, 2, 1)).transpose(1, 2)


        else:  # Multivariate case
            y = x.reshape(batchsize,l,-1,8)
            matrix_tensor = nn.Parameter(torch.randn(batchsize, l, 8, 8))
            y = torch.einsum('bqkc,bqcl->bqkl', y.to(device), matrix_tensor.to(device))

            expanded_weighted_matrix = weighted_matrix.unsqueeze(1)
            transposed_weighted_matrix = expanded_weighted_matrix.permute(0,1,3,2)
            z = torch.matmul(transposed_weighted_matrix.double().to(device), y.double().to(device))
            # if ind != -2:   
            #     weighted_matrix_save = weighted_matrix[ind].cpu().detach().numpy()
            #     np.save("matrix_self.npy", weighted_matrix_save)
            #     z_attention = z[ind].cpu().detach().numpy()
            #     np.save("attention_self.npy", z_attention)
            a = z.shape[0]
            b = z.shape[1]
            z = z.view(a,b,-1)
            conv1d_mulva = nn.Conv1d(in_channels=m, out_channels=self.d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='circular', bias=False)
            conv1d_mulva = conv1d_mulva.to(self.device)
            z = conv1d_mulva(z.permute(0, 2, 1)).transpose(1, 2)
        return z


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.causal_embedding = CausalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, weighted_matrix, x_mark):
    
        x = self.value_embedding(x) + self.position_embedding(x)+self.causal_embedding(x, weighted_matrix)
        
        return self.dropout(x)

