from typing import (
    Optional,
)
import math

import torch as T
from torch import nn
from torch.nn import functional as F

import opt_einsum as oe

from torch import Tensor

einsum = oe.contract


def masked_softmax(xs: Tensor, mask: Tensor, dim: int = -1, eps=1e-12):
    xs = xs.masked_fill(~mask, -1e9)
    xs = F.softmax(xs, dim=dim)
    return xs

class Attention(nn.Module):
    def __init__(
            self,
            kind: str,
            query_dim: int,
            input_dim: int,
            output_dim: int = None,
            activation: str = 'auto',
            scaled = True,
    ):
        super().__init__()
        assert kind in [
            'dot',
            'linear',
        ]

        self.kind = kind
        self.Dq = query_dim
        self.Din = input_dim
        self.Dout = output_dim or self.Din
        self.activation = 'auto'
        self.scaled = scaled

        self.Wq_ = nn.Linear(self.Dq, self.Din)
        self.Wk_ = nn.Linear(self.Din, self.Din)
        self.Wv_ = nn.Linear(self.Din, self.Dout)
        self.Wz_ = nn.Linear(self.Din, self.Dout)

    def forward(
            self,
            query: Tensor,
            data: Tensor,
            content_mask: Optional[Tensor] = None,
            prejudice_mask: Optional[Tensor] = None,
    ):
        #^ query: [b, ts, tw, dq]
        #^ data: [b, ts, di]
        #^ content_mask: [b, ts, tw]
        #^ prejudice_mask: [b, ts, ts]
        #^ => output: [b, ts, tw, dz]

        dimB, dimS, dimW, dimI = query.shape

        # TODO: Optimize out the [ts, ts, *] intermediate
        qs = self.Wq_(query)
        ks = self.Wk_(data)
        vs = self.Wv_(data)

        if content_mask is not None:
            words_mask = content_mask.any(2)
            #^ words_mask : [b, ts]
        else:
            words_mask = qs.new_ones((dimB, dimS))

        if self.kind == 'linear':
            # Ref: https://twitter.com/francoisfleuret/status/1267455240007188486
            assert prejudice_mask is None, "Linear mode does not support prejudice_mask."
            assert content_mask is not None, "Linear mode requires a content_mask."
            qs = T.relu(qs) * content_mask.unsqueeze(3)
            #^ qs: [bswi]
            ks = T.relu(ks) * words_mask.unsqueeze(2)
            #^ ks: [bsi]
            vks = einsum("bsi, bsz -> bzi", ks, vs)
            #^ vks : [b, dz, di]
            zs = einsum("bswi, bzi -> bswz", qs, vks)
            #^ zs : [b, ts, tw, dz]
            if self.scaled:
                ks = ks.sum(1)
                #^ ks: [bi]
                denom = einsum("bswi, bi -> bsw", qs, ks) + 1e-9
                zs = zs / denom

        elif self.kind == 'dot':
            # Ref: https://arxiv.org/abs/1706.03762
            # s=ts in q
            # S=ts in ks,vs
            att_map = einsum("bqwi, bki -> bqkw", qs, ks)
            #^ [b, ts:q, ts:k, tw]
            if self.scaled == 'seqlen':
                att_map_ndim = len(att_map.shape) - 1
                norm_coeff = words_mask.sum(1).view(-1, *([1] * att_map_ndim))
                #^ [b, _, _, _]
                att_map = att_map / T.sqrt(norm_coeff.float())
            else:
                att_map = att_map / math.sqrt(self.Din)

            if content_mask is None and prejudice_mask is None:
                att_map = F.softmax(att_map, dim=2)
            else:
                if content_mask is None:
                    assert prejudice_mask is not None # !for mypy
                    qk_mask = prejudice_mask.unsqueeze(3)
                    #^ qk_mask : [b, ts:q, ts:k, tw^]
                elif prejudice_mask is None:
                    qk_mask = words_mask.unsqueeze(1).unsqueeze(3) * content_mask.unsqueeze(2)
                    #^ qk_mask : [b, ts:q, ts:k^, tw]
                else:
                    qk_mask = words_mask.unsqueeze(1).unsqueeze(3)
                    # qk_mask = words_mask.unsqueeze(1).unsqueeze(3) * content_mask.unsqueeze(2)
                    qk_mask = qk_mask * prejudice_mask.unsqueeze(3)
                    #^ qk_mask : [b, ts:q^, ts:k, tw]

                att_map = masked_softmax(att_map, qk_mask.bool(), dim=2)

            #^ att_map : [b, ts:q, ts:k, tw]
            zs = einsum("bqkw, bkz -> bqwz", att_map, vs)

        zs = self.Wz_(zs)
        return zs, att_map