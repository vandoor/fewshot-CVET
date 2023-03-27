import torch
import torch.nn as nn
import numpy as np
from model.networks.proj import Proj
import torchvision


class FewShotModel(nn.Module):
    def __init__(self, args):
        super(FewShotModel, self).__init__()
        self.args = args
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            from model.networks.myresnet18 import MyResNet18
            hdim = 512
            self.encoder = MyResNet18(torchvision.models.resnet18(pretrained=True))
        else:
            raise ValueError('')
        self.proj = Proj(hdim, args.D, args)

    def forward(self, x, get_feature=False, require_losses=False, get_prototype=False):
        if get_feature:
            # 用在meta training上，返回的有embedding, 各类的prototype, z
            x = self.encoder(x)
            x = self.gap(x).view(x.shape[0], -1)
            ans = {'h': x}
            if get_prototype:
                ans['proto'] = self._m_forward(x)
            if require_losses:
                ans['z'] = self.get_z(x)
                ans['proto_z'] = self._m_foward(ans['z'])  # TODO z's embed dim should equal x's
            return ans
        else:
            # 用于pretraining, q，k，v，u是用来算map-map、vec-map的loss, logits就是每一类的概率值
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

