import torch
import torch.nn as nn
import numpy as np
from model.networks.proj import Proj

class FewShotModel(nn.Module):
    def __init__(self, args):
        super(FewShotModel, self).__init__()
        self.args = args
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')
        self.proj = Proj(hdim, args.D, args)

    def split_instance(self):
        args = self.args
        if self.training:
            return (
                torch.arange(args.way*args.shot).long()
                .view(1, args.shot, args.way),
                torch.arange(args.way*args.shot, args.way*(args.shot+args.query)).long()
                .view(1, args.query, args.way)
            )
        else:
            return (
                torch.arange(args.eval_way*args.eval_shot).long()
                .view(1, args.eval_shot, args.eval_way),
                torch.arange(args.eval_way*args.eval_shot, args.eval_way*(args.eval_shot+args.eval_query)).long()
                .view(1, args.eval_query,args.eval_way)
            )

    def forward(self, x, get_feature=False, require_losses=False, get_prototype=False):
        if get_feature:
            x = self.encoder(x)
            x = self.gap(x).view(x.shape[0], -1)
            ans = {'h': x}
            if get_prototype:
                ans['proto'] = self._m_forward(x)
            if require_losses:
                ans['z'] = self.get_z(x)
            return ans
        else:
            x = self.encoder(x)
            embed = self.gap(x).view(x.shape[0], -1)
            ans = {'logits': self._forward(embed)}
            if require_losses:
                ans['q'], ans['k'], ans['v'], ans['u'] = self.get_qkvu(x)
                ans['z'] = self.get_z(embed)
            return ans

    # def _forward(self, x, support_idx, query_idx):
    #     raise NotImplementedError('Suppose to be implemented by subclass')

    def _m_forward(self, x):
        raise NotImplementedError('Suppose to be implemented by subclass')
    def _forward(self, x):
        raise NotImplementedError('Suppose to be implemented by subclass')

    def get_qkvu(self, x):
        raise NotImplementedError('Suppose to be implemented by subclass')

    def get_z(self, x):
        return self.proj(x)

