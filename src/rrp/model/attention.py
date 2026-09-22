"""Attention with typed structural bias (query-row / key-column orientation).

bias[b, h, q, k] = sum_r w[r, h] * A_r[b, q, k]     (w zero-initialized)
With w == 0 the layer is numerically identical to plain attention (zero-bias equivalence).
Bias modes: 'true' | 'none' (no parameters used) | 'zero' (force 0) | 'reversed'
(transpose context relations) | 'rewired' (key-permuted relations, degree preserving).
Content attention is always kept, so missing edges never forbid attention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.data.features import N_REL


class StructuralBias(nn.Module):
    def __init__(self, heads: int, n_rel: int = N_REL, init: float = 0.0):
        super().__init__()
        self.w = nn.Parameter(torch.full((n_rel, heads), float(init)))

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        # rel [B, Q, K, R] bool -> [B, H, Q, K]
        return torch.einsum("bqkr,rh->bhqk", rel.to(self.w.dtype), self.w)


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

    def forward(self, x, kv=None, kv_cache=None, key_mask=None, bias=None, need_weights=False):
        """x [B, Q, D]; key_mask [B, K] True=valid; bias [B, H, Q, K] additive."""
        B, Q, _ = x.shape
        q = self.q(x).view(B, Q, self.h, self.dk).transpose(1, 2)
        k, v = kv_cache if kv_cache is not None else self.kv(kv if kv is not None else x)
        mask = None
        if key_mask is not None:
            mask = torch.zeros(B, 1, 1, key_mask.shape[1], dtype=q.dtype, device=q.device)
            mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if bias is not None:
            mask = bias.to(q.dtype) if mask is None else mask + bias.to(q.dtype)
        if need_weights:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            if mask is not None:
                att = att + mask
            att = att.softmax(-1)
            out = att @ v
            return self.o(out.transpose(1, 2).reshape(B, Q, -1)), att
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Q, -1))


def transform_relations(rel: torch.Tensor, mode: str, generator: torch.Generator | None = None,
                        key_mask: torch.Tensor | None = None) -> torch.Tensor | None:
    """Apply the bias-mode control to a [B,Q,K,R] relation tensor."""
    if mode in ("none", "zero"):
        return None if mode == "none" else torch.zeros_like(rel)
    if mode == "true":
        return rel
    if mode == "reversed":
        if rel.shape[1] == rel.shape[2]:
            return rel.transpose(1, 2)
        return rel
    if mode == "rewired":
        B, Q, K, R = rel.shape
        out = torch.zeros_like(rel)
        for b in range(B):
            nvalid = int(key_mask[b].sum()) if key_mask is not None else K
            perm = torch.randperm(nvalid, generator=generator)
            idx = torch.arange(K)
            idx[:nvalid] = perm
            out[b] = rel[b][:, idx]
        return out
    raise ValueError(f"unknown bias mode {mode}")
