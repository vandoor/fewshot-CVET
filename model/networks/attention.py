import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def produce_qkv(self, q, k, v):
        d_k, d_v = self.d_k, self.d_v
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, d_k)
        k = self.w_ks(k).view(sz_b, len_k, d_k)
        v = self.w_vs(v).view(sz_b, len_v, d_v)
        return q, k, v

    def forward(self, q, k, v, residual):
        output, attn, log_attn = self.attention(q, k, v)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output
