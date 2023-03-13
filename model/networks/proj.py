import torch.nn as nn
import torch
import torch.nn.functional as F


class proj(nn.Module):
    def __init__(self, args):
        if args.backbone_class == 'Res12':
            hdim = args.D
            # TODO : determine the size
        else:
            raise ValueError("no other modules are for the backbone")

        self.fc = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU(), )
