"""Cached multi-bank flow policy (system i).

Conventions (archived handoff doc 01_architecture.md):
  eps ~ N(0, I) on valid coords;  z_tau = (1 - tau) eps + tau z ;  v_target = z - eps
  loss = masked mean ||v_theta(z_tau, tau, ctx) - v_target||^2 ; sample: integrate tau 0 -> 1.
Only the action stream is noised. Context (4 typed clean banks) is encoded once per
observation, independent of tau and of the noisy actions, so per-layer cross-attention K/V
can be cached across all sampler steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.features.featurizer import BANKS, HASH_DIM
from rrp.policies.nets.attention import MHA
from rrp.policies.nets.batch import Batch, BANK_DIMS, NODE_DIM, relation_token_sets
from rrp.policies.relations.base import (EdgeSet, FAMILIES, RelCtx, assert_deployable, control_of, estimates_loss,
                                         resolve)
from rrp.policies.relations.catalog import ARM_REL_VOCAB
from rrp.policies.relations.ops import FactorSite, FieldReadouts, needs_token_fields

# what the arm sites offer factors: declared once in `catalog.FAMILIES["arm"]` (edge vocab, token hiddens (R12),
# the R13 geometry fields); a factor applies only where `FactorSite._applies` finds its field in these carries.
CTX_CARRIES = FAMILIES["arm"].sites["ctx>ctx"]
ACT_CARRIES = FAMILIES["arm"].sites["act>ctx"]


# ------------------------------------------------------------------ flow math
def interpolate_target(noise: torch.Tensor, data: torch.Tensor, tau):
    tau = torch.as_tensor(tau, dtype=noise.dtype, device=noise.device)
    while tau.dim() < noise.dim():
        tau = tau.unsqueeze(-1)
    z = (1 - tau) * noise + tau * data
    return z, data - noise


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype)
    while m.dim() < pred.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(pred)
    denom = m.sum()
    if denom == 0:
        raise ValueError("empty mask: no valid coordinates")
    return ((pred - target) ** 2 * m).sum() / denom


# ------------------------------------------------------------------ config
@dataclass
class PolicyConfig:
    width: int = 256
    heads: int = 4
    ctx_layers: int = 3
    blocks: int = 6
    horizon: int = 16
    latent_dim: int = 1               # action coords per node (1 = direct normalized action)
    factors: list | None = None       # relation factors (docs/relations.md); None = preset "arm" (the 17 typed edges
                                      # + incidence messages). Former flags: bias_mode -> control of edge.*,
                                      # structured=False -> msg.incidence serialized, slot_handles -> id.slot_handle
    dropout: float = 0.0
    attention: str = "factorized"     # factorized | dense (all-token ablation of the action stream)
    image_tokens: int = 0             # VLM resampled tokens appended to the scene bank
    image_dim: int = 0
    max_slots: int = 8
    family: str = "arm"               # net family (catalog.FAMILIES): what `factors` are checked against
    name: str = "policy"

    @property
    def D(self):
        return self.width

    def specs(self):
        return resolve(self.factors, default="arm", family=self.family)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyConfig":
        """Config dict -> PolicyConfig. Maps the pre-D-144 flags still stored in checkpoint configs (on-disk data):
        bias_mode, structured, slot_handles -> `factors`; `aux` (D-146 item 4: the hidden-state `SemanticReadout` is
        gone, a recorded `aux: true` config only names parameters `FlowPolicy` drops on load) is discarded."""
        d = dict(d)
        d.pop("aux", None)
        bm, st, sh = d.pop("bias_mode", None), d.pop("structured", None), d.pop("slot_handles", None)
        if bm is None and st is None and sh is None:
            return cls(**d)
        if d.get("factors") is not None:
            raise ValueError("config mixes `factors` with the legacy bias_mode / structured / slot_handles keys")
        items = ["preset:arm"]
        if bm not in (None, "true"):
            items.append({"name": "edge.*", "control": {"none": "off"}.get(bm, bm)})
        if st is False:
            items.append({"name": "msg.incidence", "control": "serialized"})
        if sh:
            items.append("id.slot_handle")
        return cls(**d, factors=None if items == ["preset:arm"] else items)


def sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    dt = x.dtype if x.is_floating_point() else torch.float32
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=dt) / half)
    a = x.to(dt)[..., None] * freqs * 1000.0
    return torch.cat([a.sin(), a.cos()], -1)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=None):
        super().__init__()
        h = hidden or 2 * d_out
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ context (system i clean banks)
class ContextEncoder(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        D = cfg.D
        self.cfg = cfg
        self.proj = nn.ModuleDict({b: MLP(BANK_DIMS[b], D) for b in BANKS})
        self.bank_emb = nn.Embedding(len(BANKS), D)
        self.kind_emb = nn.Embedding(8, D)
        self.text = nn.Linear(HASH_DIM, D)          # serialized pointer text (unstructured baseline)
        self.ptr = nn.Linear(D, D)                  # incidence message (structured)
        self.img = nn.Linear(cfg.image_dim, D) if cfg.image_tokens else None
        specs = cfg.specs()
        self.msg = control_of(specs, "msg.incidence")            # on | serialized | off
        self.slot_emb = nn.Embedding(cfg.max_slots, D) if any(s.name == "id.slot_handle" for s in specs) else None
        self.use_slots = control_of(specs, "id.slot_handle") == "on"
        self.layers = nn.ModuleList()
        for _ in range(cfg.ctx_layers):
            self.layers.append(nn.ModuleDict(dict(n1=nn.LayerNorm(D), att=MHA(D, cfg.heads), n2=nn.LayerNorm(D),
                                                  mlp=MLP(D, D, 4 * D),
                                                  bias=FactorSite(cfg.heads, D, "ctx>ctx", specs, CTX_CARRIES))))
        self.readouts = FieldReadouts(D, specs)
        self.fields = needs_token_fields(specs)
        self.deploy = False

    def forward(self, batch: Batch, rewire_gen=None):
        parts, masks = [], []
        for i, b in enumerate(BANKS):
            x = self.proj[b](batch.bank_tokens[b]) + self.bank_emb.weight[i] + self.kind_emb(batch.bank_kind[b].clamp(max=7))
            if self.msg == "serialized":                        # unstructured baseline: the facts as hashed text
                x = x + self.text(batch.bank_text[b])
            if b == "scene" and self.use_slots:                 # public slot address (tracker slot id)
                x = x + self.slot_emb.weight[:x.shape[1]][None]
            parts.append(x)
            masks.append(batch.bank_mask[b])
        h = torch.cat(parts, 1)
        mask = torch.cat(masks, 1)
        if self.msg == "on" and batch.pointers.shape[1] > 0:
            src, dst = batch.pointers[..., 0], batch.pointers[..., 1]
            valid = src >= 0
            gathered = torch.gather(h, 1, dst.clamp(min=0)[..., None].expand(-1, -1, h.shape[-1]))
            msg = self.ptr(gathered) * valid[..., None].to(h.dtype)
            h = h.scatter_add(1, src.clamp(min=0)[..., None].expand(-1, -1, h.shape[-1]), msg)
        rel, pad = batch.ctx_rel, 0
        if self.img is not None and "image_tokens" in batch.extra:
            it = self.img(batch.extra["image_tokens"])            # [B, I, D]
            h = torch.cat([h, it], 1)
            pad = it.shape[1]
            mask = torch.cat([mask, torch.ones(it.shape[:2], dtype=torch.bool, device=mask.device)], 1)
            rel = F.pad(rel, (0, 0, 0, pad, 0, pad))
        # the ctx / act token sets: fields the active factors read (only then), the training labels, the image pad
        edges = {"ctx>ctx": EdgeSet(ARM_REL_VOCAB, rel)}
        sets = relation_token_sets(self.cfg.family, batch, batch.extra.get("relation_labels"), self.deploy,
                                   fields=None if self.fields else (), pad_ctx=pad, edges=edges)
        rc = RelCtx(sets=sets, edges=edges, generator=rewire_gen, deploy=self.deploy)
        for li, L in enumerate(self.layers):
            self.readouts.observe(li, "ctx", h, rc)
            xn = L["n1"](h)
            qa, ka = L["bias"].augment(rc, xn, xn)
            h = h + L["att"](xn, key_mask=mask, bias=L["bias"].bias(rc), q_aug=qa, k_aug=ka)
            h = h + L["mlp"](L["n2"](h))
        return h, mask, rc


@dataclass
class ContextCache:
    """Per-layer cross-attention K/V. Valid only for the exact key it was built with."""
    key: tuple
    ctx: torch.Tensor
    ctx_mask: torch.Tensor
    kv: list
    node_emb: torch.Tensor
    act_bias: list
    node_bias: list
    node_mask: torch.Tensor
    hits: int = 0
    rc: Any = None                   # the forward's RelCtx (pair / field estimates for `estimates_loss`)


class AdaLN(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.norm = nn.LayerNorm(D, elementwise_affine=False)
        self.mod = nn.Linear(D, 2 * D)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        s, b = self.mod(c).chunk(2, -1)
        return self.norm(x) * (1 + s) + b


class ActionBlock(nn.Module):
    """Temporal self-attention per node, entity self-attention per timestep, cross-attention to
    cached clean context, MLP. All conditioned on flow time via adaLN."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        D, H = cfg.D, cfg.heads
        self.cfg = cfg
        self.n_t, self.a_t = AdaLN(D), MHA(D, H)
        self.n_e, self.a_e = AdaLN(D), MHA(D, H)
        self.n_c, self.a_c = AdaLN(D), MHA(D, H)
        self.n_m, self.mlp = AdaLN(D), MLP(D, D, 4 * D)
        specs = cfg.specs()
        self.bias_c = FactorSite(H, D, "act>ctx", specs, ACT_CARRIES)
        self.bias_e = FactorSite(H, D, "act>act", specs, ACT_CARRIES)

    def forward(self, x, tcond, cache: ContextCache, li: int, dense: bool = False):
        B, Hh, N, D = x.shape
        c = tcond[:, None, None, :]
        if dense:   # all-token ablation over (h, n)
            y = self.n_t(x, c).reshape(B, Hh * N, D)
            km = cache.node_mask[:, None, :].expand(B, Hh, N).reshape(B, Hh * N)
            x = x + self.a_t(y, key_mask=km).reshape(B, Hh, N, D)
        else:
            y = self.n_t(x, c).permute(0, 2, 1, 3).reshape(B * N, Hh, D)
            x = x + self.a_t(y).reshape(B, N, Hh, D).permute(0, 2, 1, 3)
            y = self.n_e(x, c).reshape(B * Hh, N, D)
            nm = cache.node_mask[:, None, :].expand(B, Hh, N).reshape(B * Hh, N)
            nb = cache.node_bias[li]
            nb = nb[:, None].expand(B, Hh, *nb.shape[1:]).reshape(B * Hh, *nb.shape[1:]) if nb is not None else None
            x = x + self.a_e(y, key_mask=nm, bias=nb).reshape(B, Hh, N, D)
        y = self.n_c(x, c).reshape(B, Hh * N, D)
        ab = cache.act_bias[li]
        if ab is not None:   # [B, H, N, C] -> repeat over horizon
            ab = ab[:, :, None].expand(B, ab.shape[1], Hh, N, ab.shape[-1]).reshape(B, ab.shape[1], Hh * N, -1)
        x = x + self.a_c(y, kv_cache=cache.kv[li], key_mask=cache.ctx_mask, bias=ab).reshape(B, Hh, N, D)
        x = x + self.mlp(self.n_m(x, c))
        return x


class FlowPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.D
        self.context = ContextEncoder(cfg)
        self.node = MLP(NODE_DIM, D)
        self.node_from_ctx = nn.Linear(D, D)       # node identity also read from its morph token
        self.z_in = nn.Linear(cfg.latent_dim, D)
        self.h_emb = nn.Embedding(cfg.horizon, D)
        self.tau = MLP(D, D)
        self.blocks = nn.ModuleList([ActionBlock(cfg) for _ in range(cfg.blocks)])
        self.out_norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, cfg.latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # Per-dim target standardization: the flow runs in (z - z_mean) / z_std; loss()/sample() speak raw z.
        # Identity by default (action-space flows, checkpoints saved before these buffers existed).
        self.register_buffer("z_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("z_std", torch.ones(cfg.latent_dim))
        self._register_load_state_dict_pre_hook(self._default_norm_buffers)
        self._register_load_state_dict_pre_hook(self._drop_legacy_readout)

    def _drop_legacy_readout(self, state_dict, prefix, *args):
        """On-disk data remap (D-144 addendum b): checkpoints trained with the removed `SemanticReadout` (`aux: true`, the
        v6 / v7div BC experts) carry `readout.*` tensors no module owns any more; inference never read them."""
        for k in [k for k in state_dict if k.startswith(prefix + "readout.")]:
            del state_dict[k]

    def _default_norm_buffers(self, state_dict, prefix, *args):
        for k, v in (("z_mean", self.z_mean), ("z_std", self.z_std)):
            state_dict.setdefault(prefix + k, v.detach().clone())

    def set_target_norm(self, mean: torch.Tensor, std: torch.Tensor):
        self.z_mean.copy_(mean)
        self.z_std.copy_(std.clamp(min=1e-3))

    def normalize(self, z):
        return (z - self.z_mean) / self.z_std

    def denormalize(self, z):
        return z * self.z_std + self.z_mean

    # ---------------- context / cache
    def factor_specs(self):
        return self.cfg.specs()

    def set_deploy(self, deploy: bool = True) -> "FlowPolicy":
        """Deployable inference: refuse privileged (gt) factor sources and label reads (docs/relations.md 7)."""
        if deploy:
            assert_deployable(self.cfg.specs())
        self.context.deploy = deploy
        return self

    def prepare(self, batch: Batch, key: tuple = ("uncached",), rewire_gen=None) -> ContextCache:
        ctx, cmask, rc = self.context(batch, rewire_gen)
        B, N = batch.node_feats.shape[:2]
        morph_off = batch.bank_offset["morph"]
        if "node_ctx_index" in batch.extra:              # latent path: generated entities are assemblies
            idx = batch.extra["node_ctx_index"]
            node_ctx = torch.gather(ctx, 1, idx[..., None].expand(-1, -1, ctx.shape[-1]))
        else:
            node_ctx = ctx[:, morph_off:morph_off + N]   # action node tokens are the first morph tokens
        node_emb = self.node(batch.node_feats) + self.node_from_ctx(node_ctx)
        act_rel = batch.act_rel
        if ctx.shape[1] > act_rel.shape[2]:
            act_rel = F.pad(act_rel, (0, 0, 0, ctx.shape[1] - act_rel.shape[2]))
        rc.edges["act>ctx"] = EdgeSet(ARM_REL_VOCAB, act_rel)
        rc.edges["act>act"] = EdgeSet(ARM_REL_VOCAB, batch.node_rel)
        kv, ab, nb = [], [], []
        for blk in self.blocks:
            kv.append(blk.a_c.kv(ctx))
            ab.append(blk.bias_c.bias(rc))
            nb.append(blk.bias_e.bias(rc))
        return ContextCache(key, ctx, cmask, kv, node_emb, ab, nb, batch.node_mask, rc=rc)

    # ---------------- velocity field
    def velocity(self, z: torch.Tensor, tau: torch.Tensor, cache: ContextCache, return_hidden: bool = False):
        """z [B, H, N, d], tau [B] -> v [B, H, N, d]"""
        B, Hh, N, _ = z.shape
        x = self.z_in(z) + cache.node_emb[:, None] + self.h_emb.weight[None, :Hh, None]
        tcond = self.tau(sinusoidal(tau, self.cfg.D))
        hidden = []
        for li, blk in enumerate(self.blocks):
            x = blk(x, tcond, cache, li, dense=self.cfg.attention == "dense")
            hidden.append(x)
        v = self.out(self.out_norm(x))
        v = v * cache.node_mask[:, None, :, None].to(v.dtype)
        return (v, hidden) if return_hidden else v

    def loss(self, batch: Batch, target: torch.Tensor, valid: torch.Tensor, generator=None, packet_loss_fn=None,
             packet_weight: float = 0.0, packet_tau_min: float = 0.0) -> tuple[torch.Tensor, dict]:
        """target [B,H,N,d] clean latent/action (raw space); valid [B,H,N] mask. The flow MSE is computed in
        standardized space; the packet objective sees the de-standardized estimate; the factors' own probes / pair
        estimates are supervised through `estimates_loss` (labels ride in `batch.extra["relation_labels"]`)."""
        cache = self.prepare(batch)
        target = self.normalize(target)
        B = target.shape[0]
        eps = torch.randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
        tau = torch.rand(B, generator=generator, device=target.device, dtype=target.dtype)
        z_tau, v_t = interpolate_target(eps, target, tau)
        m = (valid & batch.node_mask[:, None, :]).to(target.dtype)
        z_tau = z_tau * m[..., None]
        v = self.velocity(z_tau, tau, cache)
        fl = masked_mse(v, v_t, m)
        logs = {"flow": float(fl.detach())}
        loss = fl
        if packet_loss_fn is not None and packet_weight > 0:
            # R38: semantic objective on the predicted CLEAN LATENT (the tensor system 0 will receive):
            # z_hat_clean = z_tau + (1 - tau) * v_theta
            t_ = tau[:, None, None, None]
            z_hat_clean = self.denormalize(z_tau + (1 - t_) * v)
            if packet_tau_min > 0:
                # near-noise estimates cannot be semantically confident without distorting the velocity field:
                # only samples with tau >= packet_tau_min pass semantic gradient into v
                keep = (tau >= packet_tau_min)[:, None, None, None]
                z_hat_clean = torch.where(keep, z_hat_clean, z_hat_clean.detach())
            pl, plogs = packet_loss_fn(z_hat_clean)
            loss = loss + packet_weight * pl
            logs.update({f"zhat_{k}": x for k, x in plogs.items()})
        if cache.rc.estimates:       # the factors' own probes / pair estimates, supervised by the relation labels
            el, elogs, _ = estimates_loss(cache.rc, self.cfg.specs())    # weighted per factor (FactorSpec.weight)
            loss = loss + el
            logs.update(elogs)
        return loss, logs

    @torch.no_grad()
    def sample(self, cache: ContextCache, horizon: int, nfe: int = 8, generator=None, noise=None):
        B, N = cache.node_mask.shape
        z = noise if noise is not None else torch.randn((B, horizon, N, self.cfg.latent_dim), generator=generator,
                                                        device=cache.ctx.device, dtype=cache.ctx.dtype)
        z = z * cache.node_mask[:, None, :, None].to(z.dtype)
        dt = 1.0 / nfe
        for k in range(nfe):
            tau = torch.full((B,), k * dt, device=z.device, dtype=z.dtype)
            z = z + dt * self.velocity(z, tau, cache)
        return self.denormalize(z) * cache.node_mask[:, None, :, None].to(z.dtype)
