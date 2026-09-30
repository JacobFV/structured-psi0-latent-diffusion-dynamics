"""ReadoutProbe: the one packet / token probe (D-144, docs/relations.md 4). Probe queries are registry entries
(`FactorDef(form="readout", readout=ReadoutDef(...))`, presets `probes:<family>`); the probe answers every query of its
spec list from ONLY the received tensor z [B,K,M,dz], the query type and fixed random handle codes (addresses, not
features). `metadata_only=True` is the no-z control probe.

Layout contract: with the preset `probes:arm-packet-v1` (+ `probe.arm.goal_effect`) the module names, parameter
creation order and outputs equal the former `nets.latent_probes.PacketProbe` (tests/unit/test_relations.py), so arm /
dual checkpoints load strictly. Other families' checkpoints load through data-level key maps (units R4-R6).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.nets.attention import MHA
from rrp.policies.relations.base import FactorSpec, get_factor, resolve


def readout_defs(specs):
    out = []
    for s in specs:
        d = get_factor(s.name)
        if d.form == "readout" and s.control != "off":
            out.append((s, d.readout))
    return out


class ReadoutProbe(nn.Module):
    def __init__(self, dz: int, knots: int, specs=None, width: int = 128, heads: int = 4, max_entities: int = 8,
                 max_assemblies: int = 2, max_pairs: int = 16, metadata_only: bool = False, seed: int = 1234,
                 preset: str = "probes:arm-packet-v1"):
        super().__init__()
        from rrp.policies.nets.flow import MLP
        self.specs = resolve(specs, default=preset) if specs is None or not all(isinstance(s, FactorSpec) for s in specs) \
            else tuple(specs)
        self.queries = readout_defs(self.specs)
        addr = {r.address for _, r in self.queries}
        self.metadata_only = metadata_only
        g = torch.Generator().manual_seed(seed)
        if addr & {"entity", "entity×asm"}:
            self.register_buffer("ent_code", F.normalize(torch.randn(max_entities, 16, generator=g), dim=-1))
        self.register_buffer("asm_code", F.normalize(torch.randn(max_assemblies, 16, generator=g), dim=-1))
        if addr & {"knot×asm", "knot×pair"}:
            self.register_buffer("knot_code", F.normalize(torch.randn(knots, 16, generator=g), dim=-1))
        if "knot×pair" in addr:
            self.register_buffer("pair_code", F.normalize(torch.randn(max_pairs, 16, generator=g), dim=-1))
        D = width
        self.z_in = nn.Linear(dz, D)
        self.knot = nn.Embedding(knots, D)
        self.asm_in = nn.Linear(16, D)
        if addr & {"entity", "entity×asm"}:
            self.ent_in = nn.Linear(16, D)
        if addr & {"knot×asm", "knot×pair"}:
            self.kq_in = nn.Linear(16, D)
        if "knot×pair" in addr:
            self.pair_in = nn.Linear(16, D)
        self.qtype = nn.Embedding(len(self.queries), D)
        self.const = nn.Parameter(torch.zeros(1, 1, D))
        self.att = MHA(D, heads)
        self.att2 = MHA(D, heads)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.heads = nn.ModuleDict({r.query: nn.Linear(D, int(s.p.get("out", r.out))) for s, r in self.queries})
        self.mlp = MLP(D, D, 2 * D)

    def _tokens(self, z, zmask):
        B, K, M, _ = z.shape
        if self.metadata_only:
            pos = self.knot.weight[None, :, None] + self.asm_in(self.asm_code[:M])[None, None]
            t = self.const.expand(B, K * M, -1) + pos.reshape(1, K * M, -1)
        else:                                   # summation order kept bit-identical to the former PacketProbe
            t = self.z_in(z) + self.knot.weight[None, :, None] + self.asm_in(self.asm_code[:M])[None, None]
            t = t.reshape(B, K * M, -1)
        km = zmask[:, None, :].expand(B, K, M).reshape(B, K * M)
        return t, km

    def _read(self, q, t, km):
        r = q + self.att(self.n1(q), kv=t, key_mask=km)
        r = r + self.att2(self.n2(r), kv=t, key_mask=km)
        return r + self.mlp(r)

    def forward(self, z: torch.Tensor, zmask: torch.Tensor, n_entities: int = 0, n_pairs: int = 0) -> dict:
        """z [B,K,M,dz] (the received packet), zmask [B,M] -> {query: prediction} shaped by the query's address:
        entity [B,S,out], entity×asm [B,S,M,out], asm [B,M,out], knot×asm [B,K,M,out], knot×pair [B,K,C,out],
        body [B,out]."""
        B, K, M, _ = z.shape
        t, km = self._tokens(z, zmask)
        S = n_entities
        a = self.asm_in(self.asm_code[:M])
        out = {}
        for i, (s, r) in enumerate(self.queries):
            qt = self.qtype.weight[i]
            if r.address == "entity":
                e = self.ent_in(self.ent_code[:S])
                out[r.query] = self.heads[r.query](self._read((qt + e)[None].expand(B, S, -1), t, km))
            elif r.address == "entity×asm":
                e = self.ent_in(self.ent_code[:S])
                qq = (qt + e[:, None] + a[None]).reshape(1, S * M, -1).expand(B, -1, -1)
                out[r.query] = self.heads[r.query](self._read(qq, t, km)).reshape(B, S, M, -1)
            elif r.address == "asm":
                out[r.query] = self.heads[r.query](self._read((qt + a)[None].expand(B, M, -1), t, km))
            elif r.address == "knot×asm":
                kq = (self.kq_in(self.knot_code[:K])[:, None] + a[None]).reshape(K * M, -1)
                out[r.query] = self.heads[r.query](self._read((qt + kq)[None].expand(B, -1, -1), t, km)).reshape(B, K, M, -1)
            elif r.address == "knot×pair":
                if not n_pairs:
                    continue
                pc = self.pair_in(self.pair_code[:n_pairs])
                kp = (self.kq_in(self.knot_code[:K])[:, None] + pc[None]).reshape(K * n_pairs, -1)
                out[r.query] = self.heads[r.query](self._read((qt + kp)[None].expand(B, -1, -1), t, km)).reshape(
                    B, K, n_pairs, -1)
            elif r.address == "body":
                out[r.query] = self.heads[r.query](self._read(qt[None, None].expand(B, 1, -1), t, km))[:, 0]
            else:
                raise ValueError(f"{s.name}: address {r.address!r} is not a packet address")
        return out


# ------------------------------------------------------------------ losses (one table for every readout)
def gaussian_nll(pred, target, mask, lv_min: float = -8.0, lv_max: float = 6.0):
    d = target.shape[-1]
    mu, lv = pred[..., :d], pred[..., d:2 * d].clamp(lv_min, lv_max)
    nll = 0.5 * (((target - mu) ** 2) / lv.exp() + lv + math.log(2 * math.pi)).sum(-1)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def readout_loss(out: dict, labels: dict, specs, masks: dict | None = None) -> tuple[torch.Tensor, dict]:
    """labels[ReadoutDef.label] (target), masks[query] (bool over the prediction's leading dims; default all).
    Weight = FactorSpec.weight (default 1). Returns (total, {probe_<query>: float})."""
    L, total = {}, 0.0
    for s, r in readout_defs(specs):
        if r.query not in out or r.label not in labels:
            continue
        p, y = out[r.query], labels[r.label]
        m = (masks or {}).get(r.query)
        if m is None:
            m = torch.ones(p.shape[:-1] if r.loss not in ("ce", "soft_ce") else p.shape[:-1], dtype=torch.bool,
                           device=p.device)
        mf = m.float()
        den = mf.sum().clamp(min=1)
        if r.loss == "bce":
            v = (F.binary_cross_entropy_with_logits(p.squeeze(-1), y.float(), reduction="none") * mf).sum() / den
        elif r.loss == "mse":
            v = (((p.squeeze(-1) if y.dim() < p.dim() else p) - y * r.scale).pow(2).reshape(*m.shape, -1).sum(-1)
                 * mf).sum() / den
        elif r.loss == "gauss":
            v = gaussian_nll(p, y * r.scale, m, s.p.get("lv_min", r.lv_min))
        elif r.loss == "ce":
            v = (F.cross_entropy(p.reshape(-1, p.shape[-1]), y.reshape(-1).long(), reduction="none")
                 * mf.reshape(-1)).sum() / den
        elif r.loss == "soft_ce":
            v = (-(y * F.log_softmax(p, -1)).sum(-1) * mf).sum() / den
        elif r.loss == "cos":
            v = ((1 - (F.normalize(p, dim=-1) * y).sum(-1)) * mf).sum() / den
        else:
            raise ValueError(r.loss)
        L[r.query] = v
        total = total + (1.0 if s.weight is None else s.weight) * v
    return total, {f"probe_{k}": float(v.detach()) for k, v in L.items()}
