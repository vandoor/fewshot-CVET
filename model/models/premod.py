import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.base import FewShotModel


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, atten_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

# TODO : 魔改MultiHeadAttention， 变成不multihead，而且用于生成v_{a|b} \in (BS, H*W, D)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        # (n, bs, lenq, dk)->(n*bs, lenq, dk)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn, log_attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output


class fs_class_model(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'Res12':
            hdim = 640
        else:
            raise ValueError('')
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (-1,)))
        # support : (1, shot , way , emb) ?
        # query : (1, query , way , emb) ?
        print(support.shape)
        # 这里应该是 假设support的size是( batch, shot, way, emb)，这样求平均值才合理
        proto = support.mean(dim=1)  # batch, way, emb
        num_batch = proto.size(0)
        num_proto = support.size(1)
        num_query = np.prod(query_idx.shape[-2:])

        proto = self.slf_attn(proto, proto, proto)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (batch * nq * way, 1, emb)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            # (batch, way, emb) -> (batch, 1, way, emb) -> (batch, query, way, emb)
            proto = proto.view(num_batch * num_query, num_proto, emb_dim)
            logits = -torch.sum((proto - query) ** 2, 2)  # (batch*query, way)
        else:
            proto = F.normalize(proto, dim=-1)
            query = query.view(num_batch, -1, emb_dim)
            logits = torch.bmm(query, proto.permute([0, 2, 1]))  # (batch, query, way)
            logits = logits.view(-1, num_proto)

        return logits

