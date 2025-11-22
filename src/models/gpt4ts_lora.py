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
from peft import LoraConfig, get_peft_model



class gpt4ts_lora(nn.Module):
    
    def __init__(self, config, data):
        super(gpt4ts_lora, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        self.gpt2 = AutoModel.from_pretrained('/media/hdd/wushuang/One_fits_all/Classification/src/models/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        #LoRA 配置
        lora_config = LoraConfig(
            r=8,  # 低秩参数维度
            lora_alpha=32,  # LoRA 缩放因子
            lora_dropout=0.1,  # Dropout
            bias="none",  # 不训练 bias
            target_modules=["c_attn"]  # 仅对 GPT2 的 Self-Attention 层应用 LoRA
        )

        self.gpt2 = get_peft_model(self.gpt2, lora_config)


        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True  # 仅解冻 LayerNorm 和 Position Embedding
            elif 'lora' in name:
                param.requires_grad = True  # 仅解冻 LoRA 适配器
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
        
    def forward(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape #(16,3000,10)
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)#对最后一维使用滑动窗口
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')#(16,376,80)
        
        outputs = self.enc_embedding(input_x, None)
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs

    