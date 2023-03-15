import torch
import torch.nn as nn
import numpy as np
from model.networks.proj import proj

class FewShotModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = proj(640, args.D, args)
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')

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

    def forward(self, x, get_feature=False, require_qkv=False):
        if get_feature:
            return self.encoder(x)
        else:
            x = x.squeeze(0)
            x = self.encoder(x)
            print('base forward: conv result:', x.shape)
            num_inst = x.shape[0]
            instance_emb = self.gap(x).view(num_inst, -1)
            ans = {'z': self.fc(instance_emb)}
            # split support query set for few shot data
            support_idx, query_idx = self.split_instance()

            if self.training:
                ans['logits'] = self._forward(instance_emb, support_idx, query_idx)
            else:
                ans['logits'] = self._forward(instance_emb, support_idx, query_idx)

            if require_qkv:
                ans['q'], ans['k'], ans['v'] = self.get_qkv(x)
            return ans

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')

    def get_qkv(self, x):
        raise NotImplementedError('Suppose to be implemented by subclass')