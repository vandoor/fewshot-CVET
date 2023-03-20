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

    # def _forward(self, instance_embs, support_idx, query_idx):
    #     emb_dim = instance_embs.size(-1)
    #     support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
    #     query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (-1,)))
    #     # support : (1, shot , way , emb) ?
    #     # query : (1, query , way , emb) ?
    #     print(support.shape)
    #
    #     # 这里应该是 假设support的size是( batch, shot, way, emb)，这样求平均值才合理
    #     proto = support.mean(dim=1)  # batch, way, emb
    #     num_batch = proto.size(0)
    #     num_proto = support.size(1)
    #     num_query = np.prod(query_idx.shape[-2:])
    #
    #     q, k, v = self.slf_attn.produce_qkv(proto, proto, proto)
    #     proto = self.slf_attn(q, k, v, proto)
    #
    #     if self.args.use_euclidean:
    #         query = query.view(-1, emb_dim).unsqueeze(1)  # (batch * nq * way, 1, emb)
    #         proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
    #         # (batch, way, emb) -> (batch, 1, way, emb) -> (batch, query, way, emb)
    #         proto = proto.view(num_batch * num_query, num_proto, emb_dim)
    #         logits = -torch.sum((proto - query) ** 2, 2)  # (batch*query, way)
    #     else:
    #         proto = F.normalize(proto, dim=-1)
    #         query = query.view(num_batch, -1, emb_dim)
    #         logits = torch.bmm(query, proto.permute([0, 2, 1]))  # (batch, query, way)
    #         logits = logits.view(-1, num_proto)
    #
    #     return logits

