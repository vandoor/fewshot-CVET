import torch
import torch.nn as nn
import torchvision.models as models


class MyResNet18(nn.Module):
    def __init__(self, resnet):
        super(MyResNet18, self).__init__()
        # 去掉最后两个，用来保留H, W，算loss用
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x
