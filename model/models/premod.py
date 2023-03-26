import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.base import FewShotModel
from model.networks.attention import Attention
from model.networks.classifier import Classifier
from model.networks.proj import Proj


class PreMod(FewShotModel):
    def __init__(self, args):
        super(PreMod, self).__init__(args)
        if args.backbone_class == 'Res12':
            hdim = 640
        else:
            raise ValueError('')
        self.classify = Classifier(args)
        self.slf_attn = Attention(hdim, hdim, hdim, dropout=0.5)
        self.pru = nn.Sequential(nn.Linear(hdim, args.D), nn.ReLU())

    def get_qkvu(self, x):
        bsz, cha, hei, wei = x.shape
        x = x.permute(0, 2, 3, 1).view(bsz, hei*wei, cha)
        return *self.slf_attn.produce_qkv(x, x, x), self.pru(x)

    def _forward(self, x):
        return self.classify(x)

