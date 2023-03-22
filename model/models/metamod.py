import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from model.models.base import FewShotModel
from model.networks.attention import Attention


class MetaMod(FewShotModel):
    def __init__(self, args):
        super(MetaMod, self).__init__(args)
        self.args = args
        if args.backbone_class == 'Res12':
            hdim = 640
        else:
            raise ValueError('')
        self.slf_attn = Attention(hdim, hdim, hdim, dropout=0.5)

    def _m_forward(self, x):
        # (n_way*n_shot, embed)
        embed_dim = x.shape[-1]
        ckr = x.reshape(self.args.way, -1, embed_dim)
        ckr = torch.mean(ckr, dim=1, keepdim=True)
        # ckr: (way, 1, emb)
        q, k, v = self.slf_attn.produce_qkv(ckr, ckr, ckr)
        print(f'q,k,v shape: {q.shape},{k.shape},{v.shape},{ckr.shape}')
        new_ckr = self.slf_attn(q, k, v, ckr)
        # new_ckr : (way, 1, emb)
        new_ckr = new_ckr.squeeze(1)
        return new_ckr


