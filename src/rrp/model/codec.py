"""Graph-conditioned action codec: a[b,h,n,u] <-> z[b,h,n,d].

Encoder sees demonstrated native (normalized) actions + public body/state node features.
Decoder sees ONLY z + body/state node features (never demonstrated actions or future truth).
Losses: masked reconstruction + small KL bottleneck + physical-effect reconstruction
(TCP displacement per step, from the decoder side) so latents are effect-shaped.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MHA
from .batch import NODE_DIM
from .flow import MLP


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                               empty: str = "error") -> torch.Tensor:
    m = mask.to(pred.dtype)
    while m.dim() < pred.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(pred)
    denom = m.sum()
    if denom == 0:
        if empty == "zero":
            return (pred * 0).sum()
        raise ValueError("empty mask")
    return ((pred - target) ** 2 * m).sum() / denom


@dataclass
class CodecConfig:
    width: int = 128
    heads: int = 4
    layers: int = 2
    latent_dim: int = 4
    horizon: int = 16
    beta: float = 1e-3
    effect_weight: float = 0.5
    version: str = "codec-v1"


class NodeTimeStack(nn.Module):
    def __init__(self, D, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict(dict(n1=nn.LayerNorm(D), t=MHA(D, heads), n2=nn.LayerNorm(D),
                                                        e=MHA(D, heads), n3=nn.LayerNorm(D), m=MLP(D, D, 4 * D)))
                                     for _ in range(layers)])

    def forward(self, x, node_mask):
        B, H, N, D = x.shape
        for L in self.layers:
            y = L["n1"](x).permute(0, 2, 1, 3).reshape(B * N, H, D)
            x = x + L["t"](y).reshape(B, N, H, D).permute(0, 2, 1, 3)
            y = L["n2"](x).reshape(B * H, N, D)
            km = node_mask[:, None, :].expand(B, H, N).reshape(B * H, N)
            x = x + L["e"](y, key_mask=km).reshape(B, H, N, D)
            x = x + L["m"](L["n3"](x))
        return x


class ActionCodec(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        D = cfg.width
        self.cfg = cfg
        self.node = MLP(NODE_DIM, D)
        self.h_emb = nn.Embedding(cfg.horizon, D)
        self.a_in = nn.Linear(1, D)
        self.enc = NodeTimeStack(D, cfg.heads, cfg.layers)
        self.to_z = nn.Linear(D, 2 * cfg.latent_dim)
        self.z_in = nn.Linear(cfg.latent_dim, D)
        self.dec = NodeTimeStack(D, cfg.heads, cfg.layers)
        self.to_a = nn.Linear(D, 1)
        self.effect = nn.Linear(D, 4)   # per timestep: dTCP xyz (x10) + gripper closure change

    def _cond(self, node_feats, H):
        return self.node(node_feats)[:, None] + self.h_emb.weight[None, :H, None]

    def encode(self, a, node_feats, node_mask, sample: bool = False):
        """a [B,H,N] -> z mean [B,H,N,d] (and logvar)"""
        x = self.a_in(a[..., None]) + self._cond(node_feats, a.shape[1])
        x = self.enc(x, node_mask)
        mu, logvar = self.to_z(x).chunk(2, -1)
        logvar = logvar.clamp(-8, 4)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return z * node_mask[:, None, :, None], mu, logvar

    def decode(self, z, node_feats, node_mask):
        x = self.z_in(z) + self._cond(node_feats, z.shape[1])
        x = self.dec(x, node_mask)
        a = self.to_a(x).squeeze(-1) * node_mask[:, None, :]
        pooled = (x * node_mask[:, None, :, None]).sum(2) / node_mask.sum(1).clamp(min=1)[:, None, None]
        return a, self.effect(pooled)

    def loss(self, a, valid, node_feats, node_mask, effect_target=None):
        z, mu, logvar = self.encode(a, node_feats, node_mask, sample=True)
        a_hat, eff = self.decode(z, node_feats, node_mask)
        m = valid & node_mask[:, None, :]
        rec = masked_reconstruction_loss(a_hat, a, m)
        mm = m.to(mu.dtype)[..., None].expand_as(mu)
        kl = (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar) * mm).sum() / mm.sum().clamp(min=1)
        loss = rec + self.cfg.beta * kl
        logs = dict(rec=float(rec), kl=float(kl))
        if effect_target is not None:
            tm = valid.any(-1).to(eff.dtype)[..., None]
            el = (((eff - effect_target) ** 2) * tm).sum() / tm.sum().clamp(min=1) / 4
            loss = loss + self.cfg.effect_weight * el
            logs["effect"] = float(el)
        return loss, logs
