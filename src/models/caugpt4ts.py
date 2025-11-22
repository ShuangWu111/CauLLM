from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from models.embed import DataEmbedding, DataEmbedding_wo_time
from models.causal_M import causal_M


class caugpt4ts(nn.Module):
    
    def __init__(self, config, data):
        super(caugpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = config['gpt_layer']
        self.feat_dim = data.feature_df.shape[1] #通道数
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])
        
        self.gpt2 = AutoModel.from_pretrained('/home/wushuang23s/Classification/src/models/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        # gpt2config = GPT2Config.from_pretrained('/media/hdd/wushuang/One_fits_all/Classification/src/models/gpt2/config.json')
        # self.gpt2 = GPT2Model(gpt2config)
#        self.bert = AutoModel.from_pretrained('/root/Classification/src/models/bert', output_attentions=True, output_hidden_states=True)
#        self.bert.encoder.layer = self.bert.encoder.layer[:self.gpt_layers]
        
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(1))
        self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
        
    def average(self, x):#将多模态数据的通道数均变成1
        avg_0_6 = x[:, :, 0:6].mean(dim=2, keepdim=True)  # (16, 3000, 1)
        avg_6_8 = x[:,:,6:8].mean(dim = 2, keepdim=True)
        avg_8 = x[:,:,8].unsqueeze(-1)
        avg_9 = x[:,:,9].unsqueeze(-1)
        output = torch.cat([avg_0_6, avg_6_8, avg_8, avg_9], dim=2)
        output = output.cpu().numpy()
        return output
    
    def forward(self, x_enc, x_mark_enc, weighted_matrix):
        B, L, M = x_enc.shape#(16,3000,10)
        #x_ave = self.average(x_enc)
        device = torch.device('cuda:{}'.format(1))
        #weighted_matrix = causal_M(x_ave).to(device)#因果关联权重矩阵
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')#(16,376,80)
        
        outputs = self.enc_embedding(input_x, weighted_matrix, None)
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs

    
