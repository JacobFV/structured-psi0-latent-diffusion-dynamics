"""Multi-head attention shared by every net family, with the relation-factor hooks (D-144, docs/relations.md 3.5).

logits = q.k / sqrt(d) + bias + <q_aug, k_aug>  (+ key-mask -inf). `bias` [B,H|1,Q,K] and the augmentation
([B,H,Q,A], [B,H,K,A]) come from a `rrp.policies.relations.ops.FactorSite`; the augmentation is concatenated to q / k
(kernel-compatible: no dense [Q,K] tensor unless a bias or mask is present). Without factors the call path is the
plain SDPA call it always was.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, dim: int, heads: int, kv_dim: int | None = None):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim or dim, dim)
        self.v = nn.Linear(kv_dim or dim, dim)
        self.o = nn.Linear(dim, dim)

    def kv(self, x):
        B, T, _ = x.shape
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        return k, v

    def forward(self, x, kv=None, kv_cache=None, key_mask=None, bias=None, need_weights=False, q_aug=None, k_aug=None):
        """x [B, Q, D]; key_mask [B, K] True=valid; bias [B, H|1, Q, K] additive; q_aug / k_aug [B, H, Q|K, A]."""
        B, Q, _ = x.shape
        q = self.q(x).view(B, Q, self.h, self.dk).transpose(1, 2)
        k, v = kv_cache if kv_cache is not None else self.kv(kv if kv is not None else x)
        mask = None
        if key_mask is not None:
            mask = torch.zeros(B, 1, 1, key_mask.shape[1], dtype=q.dtype, device=q.device)
            mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if bias is not None:
            mask = bias.to(q.dtype) if mask is None else mask + bias.to(q.dtype)
        scale = None
        if q_aug is not None:
            q = torch.cat([q, q_aug.to(q.dtype) * math.sqrt(self.dk)], -1)
            k = torch.cat([k, k_aug.to(k.dtype).expand(B, -1, -1, -1)], -1)
            scale = 1.0 / math.sqrt(self.dk)
        if need_weights:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            if mask is not None:
                att = att + mask
            att = att.softmax(-1)
            out = att @ v
            return self.o(out.transpose(1, 2).reshape(B, Q, -1)), att
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale) if scale is not None else \
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Q, -1))


class RelBlock(nn.Module):
    """Pre-LN (cross-attention, self-attention, MLP) block shared by system 0 / legged / Ψ₀ / pointer nets; module keys
    n1, x, n2, s, n3, m match the per-family copies it replaces. `bias_x` / `bias_s` / aug pairs come from FactorSites."""

    def __init__(self, D: int, heads: int, mlp=None):
        super().__init__()
        from rrp.policies.nets.flow import MLP
        self.n1, self.x, self.n2, self.s, self.n3 = nn.LayerNorm(D), MHA(D, heads), nn.LayerNorm(D), MHA(D, heads), \
            nn.LayerNorm(D)
        self.m = mlp if mlp is not None else MLP(D, D, 4 * D)

    def forward(self, q, kv, kv_mask=None, q_mask=None, bias_x=None, bias_s=None, aug_x=(None, None), aug_s=(None, None)):
        q = q + self.x(self.n1(q), kv=kv, key_mask=kv_mask, bias=bias_x, q_aug=aug_x[0], k_aug=aug_x[1])
        q = q + self.s(self.n2(q), key_mask=q_mask, bias=bias_s, q_aug=aug_s[0], k_aug=aug_s[1])
        return q + self.m(self.n3(q))
