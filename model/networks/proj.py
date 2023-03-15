import torch.nn as nn
import torch
import torch.nn.functional as F


class proj(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        if args.backbone_class == 'Res12':
            hdim = args.D
            # TODO : determine the size
        else:
            raise ValueError("no other modules are for the backbone")

        self.fc = nn.Sequential(nn.Linear(in_dim, hdim), nn.ReLU(), nn.Linear(hdim, out_dim), nn.ReLU())
