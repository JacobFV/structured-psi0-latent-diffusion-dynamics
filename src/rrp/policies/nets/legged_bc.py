"""Legged behaviour-cloning positive-control network (W4: moved unchanged from rrp.learning.legged_bc, which keeps
the training loop and re-exports these names): LeggedBC, build(cfg), load_bc(path, dev).

Relations (unit HL, `legged-rel-v1`): `cfg["model"]["factors"]` (default `legged-none`: no parameters, the state-dict
keys and outputs of the pre-relations net). With a relational list (`preset:legged`) the context grows from
[glob, joints, limbs] to [glob, joints, limbs, feet, terrain cells] (the layout of `legged_latent.LeggedTokens`), each
encoder layer carries a `ctx>ctx` FactorSite and each decoder block an `act>ctx` / `act>act` one; the queries are the
joint tokens, so those edges are the joint rows of the ctx graph. The graph is built from public batch tensors only."""
from __future__ import annotations

import torch
import torch.nn as nn

from rrp.policies.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM, H
from rrp.policies.nets.attention import RelBlock
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.legged_latent import (_CARRIES, _leg_sites, _run_block, legged_graph, legged_specs,
                                             relational_specs, scan_xy)
from rrp.policies.relations.ops import FactorSite
from rrp.policies.relations.base import EdgeSet, TokenSet, assert_deployable, estimates_loss


class LeggedBC(nn.Module):
    def __init__(self, D=192, heads=4, enc_layers=2, dec_layers=3, H=H, factors=None):
        super().__init__()
        self.H, self.D = H, D
        self.specs = legged_specs(factors)
        self.rel = relational_specs(self.specs)
        self.extended, self.deploy = bool(self.rel), False
        self.node = MLP(NODE_STATIC_DIM + 2, D)
        self.glob = MLP(GLOBAL_DIM + 6 + 2, D)
        self.asm = MLP(ASM_DIM + 1, D)
        if self.extended:
            self.foot = MLP(4, D)
            self.cell = MLP(3, D)
            self.rsites = _leg_sites(self.rel, heads, D, dec_layers)
        self.enc = nn.ModuleList([nn.ModuleDict(dict(
            n=nn.LayerNorm(D), a=_mha(D, heads), n2=nn.LayerNorm(D), m=MLP(D, D, 4 * D),
            **({"b": FactorSite(heads, D, "ctx>ctx", self.rel, _CARRIES["ctx>ctx"])} if self.extended else {})))
            for _ in range(enc_layers)])
        self.x_in = nn.Linear(H, D)
        self.t_in = MLP(D, D)
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(dec_layers)])
        self.out = nn.Linear(D, H)

    def set_deploy(self, deploy: bool = True):
        if deploy:
            assert_deployable(self.specs)
        self.deploy = deploy
        return self

    def _graph(self, b, rewire_gen=None):
        """(LeggedTokens, RelCtx) whose act set is the N joint tokens (the decoder queries): act>ctx / act>act are the
        joint rows / joint-joint block of the ctx>ctx edges."""
        tok, rc = legged_graph(b, 1, self.deploy, rewire_gen)
        E, J = rc.edges["ctx>ctx"], tok.sl("joint")
        rc.sets["act"] = TokenSet("act", b["node_mask"], None, {"assembly_id": b["node_asm"][..., None]}, {}, self.deploy)
        rc.edges["act>ctx"] = EdgeSet(E.vocab, E.data[:, J])
        rc.edges["act>act"] = EdgeSet(E.vocab, E.data[:, J][:, :, J])
        return tok, rc

    def prepare(self, b, rewire_gen=None, attn=None):
        """(tokens, mask, joint tokens[, RelCtx, per-block biases when extended]); `attn` collects the encoder layers'
        [B,H,T,T] attention (diagnostics only)."""
        import math
        x = self.node(torch.cat([b["node_static"], b["q"][..., None], b["qd"][..., None]], -1))
        osc = torch.stack([torch.sin(2 * math.pi * b["osc"]), torch.cos(2 * math.pi * b["osc"])], -1)
        g = self.glob(torch.cat([b["ctx"], b["imu"], osc], -1))[:, None]
        a = self.asm(torch.cat([b["asm_static"], b["asm_touch"][..., None]], -1))
        parts = [g, x, a]
        if self.extended:
            tok, rc = self._graph(b, rewire_gen)
            st = b["asm_static"]
            parts.append(self.foot(torch.cat([st[..., 3:6], b["asm_touch"][..., None]], -1)))
            if tok.C:
                parts.append(self.cell(torch.cat([scan_xy(x.device)[None].expand(x.shape[0], -1, -1),
                                                  b["terrain"][..., None] / 0.3], -1)))
            m = tok.mask
        else:
            m = torch.cat([torch.ones_like(b["node_mask"][:, :1]), b["node_mask"], b["asm_mask"]], 1)
        t = torch.cat(parts, 1)
        for L in self.enc:
            if self.extended:
                xn = L["n"](t)
                qa, ka = L["b"].augment(rc, xn, xn)
                r = L["a"](xn, key_mask=m, bias=L["b"].bias(rc), q_aug=qa, k_aug=ka, need_weights=attn is not None)
                if attn is not None:
                    r, w = r
                    attn.append(w)
                t = t + r
            else:
                t = t + L["a"](L["n"](t), key_mask=m)
            t = t + L["m"](L["n2"](t))
        N = b["node_static"].shape[1]
        if not self.extended:
            return t, m, t[:, 1:1 + N]
        return t, m, t[:, 1:1 + N], rc, [(S["c"].bias(rc), S["s"].bias(rc)) for S in self.rsites]

    def velocity(self, xt, tau, cache, b, attn=None):
        t, m, nodes = cache[:3]
        h = nodes + self.x_in(xt) + self.t_in(sinusoidal(tau, self.D))[:, None]
        if not self.extended:
            for L in self.blocks:
                h = L(h, t, kv_mask=m, q_mask=b["node_mask"])
        else:
            for L, (bx, bs) in zip(self.blocks, cache[4]):
                h = _run_block(L, h, t, m, b["node_mask"], bx, bs, attn)
        return self.out(h)

    def loss(self, b, a, amask):
        cache = self.prepare(b)
        eps = torch.randn_like(a)
        tau = torch.rand(a.shape[0], device=a.device)
        t_ = tau[:, None, None]
        xt = (1 - t_) * eps + t_ * a
        v = self.velocity(xt, tau, cache, b)
        m = amask[..., None].float()
        loss = (((v - (a - eps)) ** 2) * m).sum() / (m.sum() * a.shape[-1])
        if self.extended and cache[3].estimates and not cache[3].deploy:
            loss = loss + estimates_loss(cache[3], self.specs)[0]     # the factors' pair estimates vs their labels
        return loss

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
    from rrp.policies.nets.attention import MHA
    return MHA(D, heads)


def build(cfg):
    m = cfg.get("model", {})
    return LeggedBC(D=m.get("width", 192), enc_layers=m.get("enc_layers", 2), dec_layers=m.get("dec_layers", 3),
                    factors=m.get("factors"))


def load_bc(path, dev):
    st = torch.load(str(path), map_location=dev, weights_only=False)
    m = build(st["cfg"]).to(dev)
    m.load_state_dict(st["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, st
