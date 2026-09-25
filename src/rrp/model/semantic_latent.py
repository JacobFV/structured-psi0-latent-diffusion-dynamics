"""Contextual semantic/action target encoder (R38, correction section 6).

    z_target = E(observed context at t [4 banks: morphology/state, scene/entities, task/events, interaction],
                 supplied task context (task bank), morphology (node + assembly tokens),
                 demonstrated behavior a[t : t+H] (normalized native actions))

Input contract (audited): E receives ONLY public featurized observations, the supplied task graph/runtime view,
public morphology and the demonstrated actions. It receives NO privileged simulator inputs (privileged labels only
supervise packet probes). E is therefore a posterior/teacher-side encoder because it sees demonstrated future
behavior; it is never used at deployment — system i must generate z from permissible observations only.

Output: z[b, K, M, dz] with a Gaussian posterior (mu, logvar); KL-to-N(0,I) regularizes scale so z is directly a
well-conditioned flow target. latent_space_version identifies a frozen (encoder, realizer, probe) triple.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .attention import MHA
from .batch import Batch, NODE_DIM, MORPH_DIM
from .flow import ContextEncoder, PolicyConfig, MLP

ASM_GRIPPER_ONEHOT = (1, 2)   # ASM_KINDS index of 'hand', 'gripper' in the morph assembly token


@dataclass
class LatentConfig:
    width: int = 256
    heads: int = 4
    ctx_layers: int = 2
    enc_layers: int = 3
    knots: int = 4
    knot_times: tuple = (0.1, 0.3, 0.5, 0.7)      # seconds after packet valid_from
    dz: int = 64
    horizon: int = 16                              # demonstrated action steps seen by E
    control_dt: float = 0.05
    beta_kl: float = 1e-3
    semantic_weight: float = 1.0                   # 0 => latent_nosem (capacity-matched control)
    realizer_layers: int = 2
    max_phase_ticks: int = 12                      # realizer trained on phases 0..max (0.55 s)
    name: str = "latent_sem_v1"
    binding_cf: float = 0.0                        # fraction of each batch appended as counterfactual-binding copies
    binding_contrast: float = 0.0                  # optional weight: push E(cf) away from E(factual) (hinge)
    slot_handles: bool = False                     # public slot-address embedding on scene tokens (see PolicyConfig)

    def version(self) -> str:
        d = asdict(self)
        for k, dflt in (("binding_cf", 0.0), ("binding_contrast", 0.0), ("slot_handles", False)):
            if d[k] == dflt:                           # added later: omit at default so v1 versions are unchanged
                d.pop(k)
        return "ls-" + hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def assembly_tokens(batch: Batch, max_m: int = 2):
    """Controllable manipulator assemblies = morph-bank assembly tokens (kind 2) whose kind one-hot is hand/gripper.
    Returns (feats [B,M,MORPH_DIM], mask [B,M], ctx_index [B,M])."""
    toks, kinds, mask = batch.bank_tokens["morph"], batch.bank_kind["morph"], batch.bank_mask["morph"]
    B = toks.shape[0]
    feats = toks.new_zeros(B, max_m, toks.shape[-1])
    am = torch.zeros(B, max_m, dtype=torch.bool, device=toks.device)
    idx = torch.zeros(B, max_m, dtype=torch.long, device=toks.device)
    is_grip = (kinds == 2) & mask & (toks[..., list(ASM_GRIPPER_ONEHOT)].sum(-1) > 0.5)
    for b in range(B):
        pos = torch.nonzero(is_grip[b]).flatten()[:max_m]
        n = len(pos)
        if n:
            feats[b, :n] = toks[b, pos]
            am[b, :n] = True
            idx[b, :n] = pos + batch.bank_offset["morph"]
    return feats, am, idx


class TargetEncoder(nn.Module):
    def __init__(self, cfg: LatentConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.width
        pc = PolicyConfig(width=D, heads=cfg.heads, ctx_layers=cfg.ctx_layers, blocks=1, horizon=cfg.horizon,
                          structured=True, bias_mode="true", aux=False, slot_handles=cfg.slot_handles)
        self.context = ContextEncoder(pc)
        self.node = MLP(NODE_DIM, D)
        self.a_in = nn.Linear(1, D)
        self.h_emb = nn.Embedding(cfg.horizon, D)
        self.q_knot = nn.Embedding(cfg.knots, D)
        self.asm = MLP(MORPH_DIM, D)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(n1=nn.LayerNorm(D), x=MHA(D, cfg.heads), n2=nn.LayerNorm(D),
                                                        s=MHA(D, cfg.heads), n3=nn.LayerNorm(D), m=MLP(D, D, 4 * D)))
                                     for _ in range(cfg.enc_layers)])
        self.out = nn.Linear(D, 2 * cfg.dz)

    def forward(self, batch: Batch, a: torch.Tensor, valid: torch.Tensor, asm_feats, asm_mask, asm_ctx_idx):
        """a [B,H,N] normalized demonstrated actions; returns mu, logvar [B,K,M,dz]."""
        ctx, cmask, _ = self.context(batch)
        B, H, N = a.shape
        beh = self.a_in(a[..., None]) + self.node(batch.node_feats)[:, None] + self.h_emb.weight[None, :H, None]
        bmask = (valid & batch.node_mask[:, None, :]).reshape(B, H * N)
        kv = torch.cat([ctx, beh.reshape(B, H * N, -1)], 1)
        kvm = torch.cat([cmask, bmask], 1)
        K, M = self.cfg.knots, asm_feats.shape[1]
        asm_ctx = torch.gather(ctx, 1, asm_ctx_idx[..., None].expand(-1, -1, ctx.shape[-1]))
        q = (self.asm(asm_feats) + asm_ctx)[:, None] + self.q_knot.weight[None, :, None]    # [B,K,M,D]
        q = q.reshape(B, K * M, -1)
        qm = asm_mask[:, None, :].expand(B, K, M).reshape(B, K * M)
        for L in self.layers:
            q = q + L["x"](L["n1"](q), kv=kv, key_mask=kvm)
            q = q + L["s"](L["n2"](q), key_mask=qm)
            q = q + L["m"](L["n3"](q))
        mu, logvar = self.out(q).reshape(B, K, M, 2, self.cfg.dz).unbind(3)
        return mu * asm_mask[:, None, :, None], logvar.clamp(-8, 4)
