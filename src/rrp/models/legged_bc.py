"""Legged behaviour-cloning positive-control network (W4: moved unchanged from rrp.learning.legged_bc, which keeps
the training loop and re-exports these names): LeggedBC, build(cfg), load_bc(path, dev)."""
from __future__ import annotations

import torch
import torch.nn as nn

from rrp.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM, H
from rrp.models.flow import MLP, sinusoidal
from rrp.models.legged_latent import block, run_block


class LeggedBC(nn.Module):
    def __init__(self, D=192, heads=4, enc_layers=2, dec_layers=3, H=H):
        super().__init__()
        self.H, self.D = H, D
        self.node = MLP(NODE_STATIC_DIM + 2, D)
        self.glob = MLP(GLOBAL_DIM + 6 + 2, D)
        self.asm = MLP(ASM_DIM + 1, D)
        self.enc = nn.ModuleList([nn.ModuleDict(dict(n=nn.LayerNorm(D), a=_mha(D, heads), n2=nn.LayerNorm(D),
                                                     m=MLP(D, D, 4 * D))) for _ in range(enc_layers)])
        self.x_in = nn.Linear(H, D)
        self.t_in = MLP(D, D)
        self.blocks = nn.ModuleList([block(D, heads) for _ in range(dec_layers)])
        self.out = nn.Linear(D, H)

    def prepare(self, b):
        import math
        x = self.node(torch.cat([b["node_static"], b["q"][..., None], b["qd"][..., None]], -1))
        osc = torch.stack([torch.sin(2 * math.pi * b["osc"]), torch.cos(2 * math.pi * b["osc"])], -1)
        g = self.glob(torch.cat([b["ctx"], b["imu"], osc], -1))[:, None]
        a = self.asm(torch.cat([b["asm_static"], b["asm_touch"][..., None]], -1))
        t = torch.cat([g, x, a], 1)
        m = torch.cat([torch.ones_like(b["node_mask"][:, :1]), b["node_mask"], b["asm_mask"]], 1)
        for L in self.enc:
            t = t + L["a"](L["n"](t), key_mask=m)
            t = t + L["m"](L["n2"](t))
        N = b["node_static"].shape[1]
        return t, m, t[:, 1:1 + N]

    def velocity(self, xt, tau, cache, b):
        t, m, nodes = cache
        h = nodes + self.x_in(xt) + self.t_in(sinusoidal(tau, self.D))[:, None]
        for L in self.blocks:
            h = run_block(L, h, t, m, b["node_mask"])
        return self.out(h)

    def loss(self, b, a, amask):
        cache = self.prepare(b)
        eps = torch.randn_like(a)
        tau = torch.rand(a.shape[0], device=a.device)
        t_ = tau[:, None, None]
        xt = (1 - t_) * eps + t_ * a
        v = self.velocity(xt, tau, cache, b)
        m = amask[..., None].float()
        return (((v - (a - eps)) ** 2) * m).sum() / (m.sum() * a.shape[-1])

    @torch.no_grad()
    def sample(self, b, nfe=8, generator=None):
        cache = self.prepare(b)
        B, N = b["node_mask"].shape
        x = torch.randn(B, N, self.H, device=b["q"].device, generator=generator)
        for k in range(nfe):
            tau = torch.full((B,), k / nfe, device=x.device)
            x = x + (1.0 / nfe) * self.velocity(x, tau, cache, b)
        return x * b["node_mask"][..., None]


def _mha(D, heads):
    from rrp.models.attention import MHA
    return MHA(D, heads)


def build(cfg):
    m = cfg.get("model", {})
    return LeggedBC(D=m.get("width", 192), enc_layers=m.get("enc_layers", 2), dec_layers=m.get("dec_layers", 3))


def load_bc(path, dev):
    st = torch.load(str(path), map_location=dev, weights_only=False)
    m = build(st["cfg"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, st
