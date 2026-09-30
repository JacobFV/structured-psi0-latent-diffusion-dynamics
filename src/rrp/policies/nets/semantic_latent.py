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
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from rrp.policies.nets.attention import MHA
from rrp.policies.nets.batch import Batch, NODE_DIM, MORPH_DIM
from rrp.policies.nets.flow import ContextEncoder, PolicyConfig, MLP
from rrp.policies.relations.base import resolve

ASM_GRIPPER_ONEHOT = (1, 2)   # ASM_KINDS index of 'hand', 'gripper' in the morph assembly token

# D-144 R2: the arm/dual probe queries of preset "probes:arm-packet-v1" (catalog.py), in the order LatentConfig's
# legacy scalar knobs used to apply uniformly. `goal_effect` is excluded (R1: it is a SEPARATE, opt-in spec toggled
# by the run's `probe` config block, not by this scalar).
_ARM_PROBE_QUERIES = ("visible", "looking_at", "focused_on", "held_by", "acting_on", "rel_pos", "observed_effect",
                      "subtask")


# D-144 addendum (2026-09-30, decision (b)): the ONE legacy-key mapping table for on-disk LatentConfig
# configs/checkpoints written before unit R2 (docs/relations.md 10). Applies to LOADING an existing legacy dict
# into `LatentConfig(**kw)` only -- it is not a construction surface for new configs (see `__init__` below), the
# same status as `LEGACY_PICKLE_MODULES` in `harness.data.collect`. Extend it only if another flat key is found to
# have fed `LatentConfig` historically.
LATENT_LEGACY_KEYS = ("semantic_weight", "probe_lv_min", "binding_cf", "binding_contrast", "slot_handles")


def _probe_factors(weight: float, lv_min: float, slot_handles: bool) -> tuple:
    """`factors:` entries equivalent to the pre-R2 (semantic_weight, probe_lv_min, slot_handles) triple: one
    per-query readout weight/lv_min override (docs/relations.md 4: "Weight, lv_min ... become spec weight/params")
    plus `id.slot_handle` when slot_handles was set (PolicyConfig already reads this same factor name)."""
    items = [{"name": f"probe.arm.{q}", "weight": weight, "params": {"lv_min": lv_min}} for q in _ARM_PROBE_QUERIES]
    if slot_handles:
        items.append("id.slot_handle")
    return tuple(items)


@dataclass(init=False)
class LatentConfig:
    """R38 target-encoder config. `factors` (docs/relations.md 10, unit R2) is the one declarative knob for what used
    to be four separate scalars (semantic_weight, probe_lv_min, slot_handles were LatentConfig fields;
    packet_semantic_weight / aux_weight were separate top-level run-config keys read by the flow/BC trainers): each
    arm probe query's loss weight and Gaussian log-variance floor is now a `probe.arm.<query>` FactorSpec override
    (`weight`, `params.lv_min`), and the public slot-address embedding is the `id.slot_handle` factor -- the SAME
    registry entries `nets.probes.ReadoutProbe` / `PolicyConfig.factors` already read (F2-F4, R1). `cf_mix` /
    `cf_contrast` (ex `binding_cf` / `binding_contrast`) stay plain scalars: the counterfactual-binding swap
    (`nets.latent_batch`, ported onto `harness.data.relgen.transforms.cf_swap`) has no catalog entry of its own
    (out of this unit's owned files: catalog.py belongs to no single fanout row for a non-relational data
    augmentation), so there is nothing in `factors:` for it to reference.

    Legacy construction: every stored checkpoint / config on disk was written as
    `LatentConfig(**cfg_json["latent"])` with the flat pre-R2 keys (`semantic_weight`, `probe_lv_min`, `binding_cf`,
    `binding_contrast`, `slot_handles`); that call keeps working unchanged (`bundles.load_representation`,
    `policies.latent`, this module's own `train_representation` -- two of those three files are owned by other
    fanout units and must not be edited here). `__init__` accepts either the flat legacy keys OR `factors=`
    (mixing both is an error, same rule as `PolicyConfig.from_dict`)."""
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
    realizer_layers: int = 2
    max_phase_ticks: int = 12                      # realizer trained on phases 0..max (0.55 s)
    name: str = "latent_sem_v1"
    factors: tuple | None = None                   # None = the former defaults (weight 1.0, lv_min -8.0, no handles)
    cf_mix: float = 0.0                             # fraction of each batch appended as counterfactual-binding copies
    cf_contrast: float = 0.0                        # optional weight: push E(cf) away from E(factual) (hinge)

    def __init__(self, **kw):
        legacy = {k: kw.pop(k) for k in LATENT_LEGACY_KEYS if k in kw}
        if legacy and kw.get("factors") is not None:
            raise ValueError("LatentConfig: mixes `factors` with the legacy semantic_weight/probe_lv_min/"
                             "binding_cf/binding_contrast/slot_handles keys")
        if legacy:
            kw["factors"] = _probe_factors(legacy.get("semantic_weight", 1.0), legacy.get("probe_lv_min", -8.0),
                                           legacy.get("slot_handles", False))
            kw["cf_mix"] = legacy.get("binding_cf", 0.0)
            kw["cf_contrast"] = legacy.get("binding_contrast", 0.0)
        names = {f.name for f in fields(self)}
        unknown = set(kw) - names
        if unknown:
            raise TypeError(f"LatentConfig: unknown field(s) {sorted(unknown)}")
        for f in fields(self):
            setattr(self, f.name, kw.get(f.name, f.default))

    # ------------------------------------------------------------------ factor-derived (docs/relations.md 4, R2)
    @property
    def specs(self):
        return resolve(self.factors if self.factors is not None else _probe_factors(1.0, -8.0, False))

    @property
    def weight(self) -> float:
        """The (uniform, pre-R2-equivalent) probe loss weight: every `probe.arm.*` spec carries the same value."""
        ws = {s.weight for s in self.specs if s.name.startswith("probe.arm.") and s.control != "off"}
        return next(iter(ws)) if len(ws) == 1 else (1.0 if not ws else sorted(ws)[-1])

    @property
    def lv_min(self) -> float:
        lvs = {s.p.get("lv_min", -8.0) for s in self.specs if s.name.startswith("probe.arm.") and s.control != "off"}
        return next(iter(lvs)) if len(lvs) == 1 else (-8.0 if not lvs else sorted(lvs)[-1])

    @property
    def slot_handles(self) -> bool:
        return any(s.name == "id.slot_handle" and s.control != "off" for s in self.specs)

    def version(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "factors"}
        # --- back-compat only: reproduces the pre-R2 hash shape bit-for-bit so existing checkpoints / lineage
        # `latent_space_version` strings stay valid (the acceptance criterion "LatentConfig.version() identical for
        # every config under configs/"). This is the one place in this unit's owned files where the retired flat key
        # NAMES ("semantic_weight", "probe_lv_min", "binding_cf") must still appear: they are literally the JSON
        # object keys the ALREADY-COMPUTED, already-shipped hashes were taken over, and a SHA-256 preimage cannot be
        # reproduced under different key names. See research/tracks/rel-r2.md ("open question for the lead").
        d.pop("cf_mix", None); d.pop("cf_contrast", None)
        d["semantic_weight"], d["probe_lv_min"] = self.weight, self.lv_min
        d["binding_cf"], d["binding_contrast"], d["slot_handles"] = self.cf_mix, self.cf_contrast, self.slot_handles
        for k, dflt in (("binding_cf", 0.0), ("binding_contrast", 0.0), ("slot_handles", False),
                         ("probe_lv_min", -8.0)):
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
                          aux=False, factors=["preset:arm", "id.slot_handle"] if cfg.slot_handles else None)
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
