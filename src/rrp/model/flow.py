"""Cached multi-bank flow policy (system i).

Conventions (docs/handoff/docs/01_architecture.md):
  eps ~ N(0, I) on valid coords;  z_tau = (1 - tau) eps + tau z ;  v_target = z - eps
  loss = masked mean ||v_theta(z_tau, tau, ctx) - v_target||^2 ; sample: integrate tau 0 -> 1.
Only the action stream is noised. Context (4 typed clean banks) is encoded once per
observation, independent of tau and of the noisy actions, so per-layer cross-attention K/V
can be cached across all sampler steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.data.features import BANKS, HASH_DIM, N_REL
from .attention import MHA, StructuralBias, transform_relations
from .batch import Batch, BANK_DIMS, NODE_DIM


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
    structured: bool = True           # pointers/incidence messages
    bias_mode: str = "true"           # true|none|zero|reversed|rewired
    aux: bool = True                  # auxiliary semantic readouts from action hidden states
    dropout: float = 0.0
    attention: str = "factorized"     # factorized | dense (all-token ablation of the action stream)
    image_tokens: int = 0             # VLM resampled tokens appended to the scene bank
    image_dim: int = 0
    max_slots: int = 8
    slot_handles: bool = False        # add a learned embedding of the PUBLIC tracker slot id (entity address) to
                                      # scene tokens; needed when slot order is not canonical (paired binding data)
    name: str = "policy"

    @property
    def D(self):
        return self.width


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
        self.slot_emb = nn.Embedding(cfg.max_slots, D) if cfg.slot_handles else None
        self.layers = nn.ModuleList()
        for _ in range(cfg.ctx_layers):
            self.layers.append(nn.ModuleDict(dict(n1=nn.LayerNorm(D), att=MHA(D, cfg.heads), n2=nn.LayerNorm(D),
                                                  mlp=MLP(D, D, 4 * D), bias=StructuralBias(cfg.heads))))

    def forward(self, batch: Batch, rewire_gen=None):
        parts, masks = [], []
        for i, b in enumerate(BANKS):
            x = self.proj[b](batch.bank_tokens[b]) + self.bank_emb.weight[i] + self.kind_emb(batch.bank_kind[b].clamp(max=7))
            if not self.cfg.structured:
                x = x + self.text(batch.bank_text[b])
            if b == "scene" and self.slot_emb is not None:     # public slot address (tracker slot id)
                x = x + self.slot_emb.weight[:x.shape[1]][None]
            parts.append(x)
            masks.append(batch.bank_mask[b])
        h = torch.cat(parts, 1)
        mask = torch.cat(masks, 1)
        if self.cfg.structured and batch.pointers.shape[1] > 0:
            src, dst = batch.pointers[..., 0], batch.pointers[..., 1]
            valid = src >= 0
            gathered = torch.gather(h, 1, dst.clamp(min=0)[..., None].expand(-1, -1, h.shape[-1]))
            msg = self.ptr(gathered) * valid[..., None].to(h.dtype)
            h = h.scatter_add(1, src.clamp(min=0)[..., None].expand(-1, -1, h.shape[-1]), msg)
        if self.img is not None and "image_tokens" in batch.extra:
            it = self.img(batch.extra["image_tokens"])            # [B, I, D]
            h = torch.cat([h, it], 1)
            mask = torch.cat([mask, torch.ones(it.shape[:2], dtype=torch.bool, device=mask.device)], 1)
        rel = batch.ctx_rel
        if self.img is not None and "image_tokens" in batch.extra:
            pad = batch.extra["image_tokens"].shape[1]
            rel = F.pad(rel, (0, 0, 0, pad, 0, pad))
        rel = transform_relations(rel, self.cfg.bias_mode, rewire_gen, mask) if self.cfg.structured or \
            self.cfg.bias_mode != "none" else None
        for L in self.layers:
            bias = L["bias"](rel) if rel is not None else None
            h = h + L["att"](L["n1"](h), key_mask=mask, bias=bias)
            h = h + L["mlp"](L["n2"](h))
        return h, mask, rel


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
        self.bias_c = StructuralBias(H)
        self.bias_e = StructuralBias(H)

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
        self.readout = SemanticReadout(cfg) if cfg.aux else None
        # Per-dim target standardization: the flow runs in (z - z_mean) / z_std; loss()/sample() speak raw z.
        # Identity by default (action-space flows, checkpoints saved before these buffers existed).
        self.register_buffer("z_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("z_std", torch.ones(cfg.latent_dim))
        self._register_load_state_dict_pre_hook(self._default_norm_buffers)

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
    def prepare(self, batch: Batch, key: tuple = ("uncached",), rewire_gen=None) -> ContextCache:
        ctx, cmask, _ = self.context(batch, rewire_gen)
        B, N = batch.node_feats.shape[:2]
        morph_off = batch.bank_offset["morph"]
        if "node_ctx_index" in batch.extra:              # latent path: generated entities are assemblies
            idx = batch.extra["node_ctx_index"]
            node_ctx = torch.gather(ctx, 1, idx[..., None].expand(-1, -1, ctx.shape[-1]))
        else:
            node_ctx = ctx[:, morph_off:morph_off + N]   # action node tokens are the first morph tokens
        node_emb = self.node(batch.node_feats) + self.node_from_ctx(node_ctx)
        use_bias = self.cfg.bias_mode != "none"
        act_rel = batch.act_rel
        if ctx.shape[1] > act_rel.shape[2]:
            act_rel = F.pad(act_rel, (0, 0, 0, ctx.shape[1] - act_rel.shape[2]))
        if use_bias:
            act_rel = transform_relations(act_rel, "rewired" if self.cfg.bias_mode == "rewired" else
                                          ("zero" if self.cfg.bias_mode == "zero" else "true"), rewire_gen, cmask)
        kv, ab, nb = [], [], []
        for blk in self.blocks:
            kv.append(blk.a_c.kv(ctx))
            ab.append(blk.bias_c(act_rel) if use_bias else None)
            nb.append(blk.bias_e(batch.node_rel) if use_bias and self.cfg.bias_mode != "zero" else None)
        return ContextCache(key, ctx, cmask, kv, node_emb, ab, nb, batch.node_mask)

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

    def loss(self, batch: Batch, target: torch.Tensor, valid: torch.Tensor, labels: dict | None = None,
             aux_weight: float = 0.1, generator=None, packet_loss_fn=None,
             packet_weight: float = 0.0, packet_tau_min: float = 0.0) -> tuple[torch.Tensor, dict]:
        """target [B,H,N,d] clean latent/action (raw space); valid [B,H,N] mask. The flow MSE is computed in
        standardized space; packet/readout objectives see the de-standardized estimate."""
        cache = self.prepare(batch)
        target = self.normalize(target)
        B = target.shape[0]
        eps = torch.randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
        tau = torch.rand(B, generator=generator, device=target.device, dtype=target.dtype)
        z_tau, v_t = interpolate_target(eps, target, tau)
        m = (valid & batch.node_mask[:, None, :]).to(target.dtype)
        z_tau = z_tau * m[..., None]
        v, hidden = self.velocity(z_tau, tau, cache, return_hidden=True)
        fl = masked_mse(v, v_t, m)
        logs = {"flow": float(fl.detach())}
        loss = fl
        if packet_loss_fn is not None and packet_weight > 0:
            # R38: semantic objective on the predicted CLEAN LATENT (the tensor system 0 will receive), not on
            # hidden states: z_hat_clean = z_tau + (1 - tau) * v_theta
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
        if self.readout is not None and labels is not None:
            # predicted clean action from the current estimate (future-effect readouts use it)
            t_ = tau[:, None, None, None]
            z_hat = self.denormalize(z_tau + (1 - t_) * v)
            aux, alog = self.readout(hidden, cache, batch, labels, z_hat)
            loss = loss + aux_weight * aux
            logs.update(alog)
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


# ------------------------------------------------------------------ auxiliary semantic readouts
class SemanticReadout(nn.Module):
    """Query-conditioned readouts from ACTION-EXPERT hidden states (system i), bound to canonical
    scene slots. Gradients flow into the denoising blocks (R37)."""

    def __init__(self, cfg: PolicyConfig, layers=(-1, None)):
        super().__init__()
        D = cfg.D
        self.cfg = cfg
        self.q = nn.Linear(D, D)
        self.att = MHA(D, cfg.heads)
        self.zproj = nn.Linear(cfg.latent_dim, D)
        self.heads = nn.ModuleDict(dict(held=nn.Linear(D, 1), contact=nn.Linear(D, 1), visible=nn.Linear(D, 1),
                                        rel=nn.Linear(D, 3), focus=nn.Linear(D, 1), fut=nn.Linear(D, 3),
                                        gaze=nn.Linear(D, 1)))

    def read(self, hidden, cache, batch, z_hat=None, layer: int = None):
        layer = len(hidden) // 2 if layer is None else layer
        h = hidden[layer]                                        # [B, H, N, D]
        B, Hh, N, D = h.shape
        if z_hat is not None:
            h = h + self.zproj(z_hat)
        keys = h.reshape(B, Hh * N, D)
        km = cache.node_mask[:, None, :].expand(B, Hh, N).reshape(B, Hh * N)
        so = batch.bank_offset["scene"]
        S = batch.bank_tokens["scene"].shape[1]
        slot_q = self.q(cache.ctx[:, so:so + S])                 # canonical slot handles
        r = self.att(slot_q, kv=keys, key_mask=km)                # [B, S, D]
        return {k: v(r) for k, v in self.heads.items()}

    def forward(self, hidden, cache, batch, labels, z_hat):
        out = self.read(hidden, cache, batch, z_hat)
        smask = batch.bank_mask["scene"] & labels["slot_valid"]
        m = smask.float()
        den = m.sum().clamp(min=1)
        bce = lambda logit, y: (F.binary_cross_entropy_with_logits(logit.squeeze(-1), y.float(),
                                                                   reduction="none") * m).sum() / den
        l_held = bce(out["held"], labels["held"])
        l_contact = bce(out["contact"], labels["contact"])
        l_vis = bce(out["visible"], labels["visible"])
        l_focus = bce(out["focus"], labels["focus"])
        l_rel = ((out["rel"] - labels["rel_tcp"] * 5).pow(2).sum(-1) * m).sum() / den
        l_fut = ((out["fut"] - labels["future_disp"] * 10).pow(2).sum(-1) * m).sum() / den
        l_gaze = ((out["gaze"].squeeze(-1) - labels["gaze"] / 30).pow(2) * m).sum() / den
        total = l_held + l_contact + l_vis + l_focus + l_rel + l_fut + 0.5 * l_gaze
        return total, dict(aux_held=float(l_held), aux_contact=float(l_contact), aux_visible=float(l_vis),
                           aux_focus=float(l_focus), aux_rel=float(l_rel), aux_future=float(l_fut),
                           aux_gaze=float(l_gaze))
