"""Legged latent packet models (R38 architecture instantiated for legs/body/arms assemblies).

    E  (target encoder, training only): public context at t + morphology + demonstrated native joint targets
       a[t : t+H]  ->  z[K, M, dz]  (Gaussian posterior; KL to N(0, I))
    R  (system 0, online every 20 ms): received z + morphology node tokens + CURRENT joint encoders, IMU,
       foot touch, osc-v1 phase + elapsed phase since packet.valid_from  ->  native joint targets (action units)
       Each joint attends only to the knots of its own assembly and of the body assembly.
    P  (packet probe): P(z, query type, opaque assembly/knot codes) -> locomotion semantics:
         leg m:  contact(m, k)     true foot contact at knot time k (future physical outcome; LABEL only)
         body :  goal_rel          active waypoint in the true body frame (m, Gaussian)
                 displacement      base (dx, dy, dyaw) over the packet horizon 0.8 s (body frame, Gaussian)
                 subtask           active event (walk_to_a / walk_to_b / halt / done)
                 fall              fall within the horizon
       metadata_only=True: control probe without z.
    S  (system i flow, deployable): public context only (no demonstrated actions, no privileged state)
       -> z via rectified flow on the standardized E mean; semantic loss through the frozen P on
       z_hat_clean for tau >= tau_min.

Relations (unit HL, docs/relations.md section 11, `legged-rel-v1`): with a relational factor list (`preset:legged`)
the public context grows from [glob, joints] to [glob, joints, limbs, feet, terrain cells] and E / S run
`FactorSite`s at `ctx>ctx` (each context layer), `act>ctx` and `act>act` (each packet block). The graph is derived
from PUBLIC batch tensors only (`legged_graph`). Without one (`legged-none`, the default) every class runs the
pre-HL code path: no extra parameters, the same state-dict keys and outputs, so old checkpoints load unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM, KNOT_TIMES, MAX_M, H
from rrp.policies.nets.attention import MHA, RelBlock
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.probes import ReadoutProbe
from rrp.policies.relations.base import (EdgeSet, FactorSpec, FAMILIES, RelCtx, TOKEN_KINDS, TokenSet,
                                         assert_deployable, estimates_loss, get_factor, resolve)
from rrp.policies.relations.catalog import LEGGED_REL_VOCAB
from rrp.policies.relations.ops import FactorSite, applicable_sites

N_SUBTASK = 4
LEGGED_SITES = ("ctx>ctx", "act>ctx", "act>act")
_CARRIES = {s: FAMILIES["legged"].sites[s] for s in LEGGED_SITES}     # what the leg sites offer factors (catalog)

# Public terrain scan (D-146; `envs.mujoco.legged_core.SCAN_*`, pinned equal by tests/unit/test_legged_relations.py):
# cell index = ix * SCAN_NY + iy, body yaw frame, x forward. Nets never import the env.
SCAN_NX, SCAN_NY, SCAN_DX, SCAN_X0, SCAN_Y0 = 11, 7, 0.1, -0.2, -0.3
SCAN_DIM = SCAN_NX * SCAN_NY
OVER_CELL_DX, OVER_CELL_DY = (-0.05, 0.35), 0.15          # `edge.over_cell`: landing window of a foot (m, body frame)


def scan_xy(device=None) -> torch.Tensor:
    """[SCAN_DIM, 2] body-frame (x, y) of every terrain-scan cell."""
    gx, gy = torch.meshgrid(SCAN_X0 + SCAN_DX * torch.arange(SCAN_NX), SCAN_Y0 + SCAN_DX * torch.arange(SCAN_NY),
                            indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).to(device=device, dtype=torch.float32)


def legged_specs(factors) -> tuple:
    """The factor list a legged net is built from: FactorSpecs pass through, anything else resolves against the legged
    family with the default `legged-none` (no relational factor)."""
    if factors is not None and all(isinstance(x, FactorSpec) for x in factors):
        return tuple(factors)
    return resolve(factors, default="legged-none", family="legged")


def relational_specs(specs) -> tuple:
    """The specs that attach at a ctx>ctx / act>ctx / act>act site (off-control ones included: an ablation keeps the
    token layout and the parameters, it only switches the term off)."""
    ft = FAMILIES["legged"]
    return tuple(s for s in specs if set(applicable_sites(get_factor(s.name), s, specs, ft)) & set(LEGGED_SITES))


# ------------------------------------------------------------------ the legged relation graph (public tensors only)
@dataclass
class LeggedTokens:
    """Layout of the extended ctx token set [glob | N joints | M limbs | M feet | C terrain cells]."""
    N: int
    M: int
    C: int
    mask: Any                      # [B,T] bool
    foot_mask: Any                 # [B,M]
    foot_xy: Any                   # [B,M,2] default-stance foot xy (m, body frame)

    @property
    def T(self):
        return 1 + self.N + 2 * self.M + self.C

    def sl(self, part: str) -> slice:
        N, M = self.N, self.M
        return {"glob": slice(0, 1), "joint": slice(1, 1 + N), "limb": slice(1 + N, 1 + N + M),
                "foot": slice(1 + N + M, 1 + N + 2 * M), "cell": slice(1 + N + 2 * M, self.T)}[part]


def legged_tokens(b) -> LeggedTokens:
    B, N = b["node_mask"].shape
    M = b["asm_mask"].shape[1]
    ter = b.get("terrain")
    C = 0 if ter is None else ter.shape[1]
    fm = b["asm_mask"] & b["asm_is_leg"]
    parts = [torch.ones_like(b["node_mask"][:, :1]), b["node_mask"], b["asm_mask"], fm]
    if C:
        parts.append(b["terrain_valid"] if "terrain_valid" in b else torch.ones(B, C, dtype=torch.bool, device=fm.device))
    a = b["asm_static"]
    return LeggedTokens(N, M, C, torch.cat(parts, 1), fm, a[..., 3:5] * a[..., 8:9])


def _mirror_matrix(b) -> torch.Tensor:
    """[B,M,M] bool: assembly m <-> its left/right mirror (same kind, opposite side, nearest by |dx| + ||y|-|y'|| + |dz|
    on the assembly position, kept only when mutual). Body / sideless assemblies have none."""
    a, am = b["asm_static"], b["asm_mask"]
    M = a.shape[1]
    kind, side, pos = a[..., 0:3].argmax(-1), a[..., 7], a[..., 3:6]
    ok = ((kind[:, :, None] == kind[:, None, :]) & (side[:, :, None] * side[:, None, :] < 0)
          & am[:, :, None] & am[:, None, :])
    cost = ((pos[:, :, None, 0] - pos[:, None, :, 0]).abs() + (pos[:, :, None, 1].abs() - pos[:, None, :, 1].abs()).abs()
            + (pos[:, :, None, 2] - pos[:, None, :, 2]).abs()).masked_fill(~ok, float("inf"))
    best = cost.argmin(-1)
    has = torch.isfinite(cost.gather(-1, best[..., None])[..., 0])
    mutual = best.gather(1, best) == torch.arange(M, device=best.device)
    return F.one_hot(best, M).bool() & (has & mutual)[..., None]


def legged_edges(b, tok: LeggedTokens, K: int) -> dict:
    """The `legged-rel-v1` EdgeSets: {"ctx>ctx": [B,T,T,9], "act>ctx": [B,K*M,T,9], "act>act": [B,K*M,K*M,9]} (bool, channel
    order `catalog.LEGGED_REL_VOCAB`). act tokens are (knot, assembly), k-major. Joint kinematics come from the per-
    assembly depth counter (a chain), limb adjacency from kind grouping (see the catalog doc of each edge)."""
    B, N, M, C, T = b["node_mask"].shape[0], tok.N, tok.M, tok.C, tok.T
    dev = b["node_mask"].device
    ch = {n: i for i, n in enumerate(LEGGED_REL_VOCAB)}
    na, nm, am, fm = b["node_asm"], b["node_mask"], b["asm_mask"], tok.foot_mask
    depth = (b["node_static"][..., 9] * 6).round().long()
    oh = F.one_hot(na, M).bool() & nm[..., None]                                   # [B,N,M] joint -> its assembly
    same = (na[:, :, None] == na[:, None, :]) & nm[:, :, None] & nm[:, None, :]
    eyeN, eyeM = torch.eye(N, dtype=torch.bool, device=dev), torch.eye(M, dtype=torch.bool, device=dev)
    dj = depth[:, :, None] - depth[:, None, :]                                     # depth(i) - depth(j)
    mir = _mirror_matrix(b)                                                        # [B,M,M]
    ohf, mirf = oh.float(), mir.float()
    mir_jj = ((ohf @ mirf @ ohf.transpose(1, 2)) > 0) & (dj == 0)                  # homologous joint of the mirror limb
    kind = b["asm_static"][..., 0:3].argmax(-1)
    adj = ((kind[:, :, None] == kind[:, None, :]) & (kind[:, :, None] != 1) & am[:, :, None] & am[:, None, :]
           & ~eyeM)                                                                # same-kind non-body limbs
    dfoot = torch.diag_embed(fm)                                                   # [B,M,M] limb m -> foot m
    if C:
        cxy = scan_xy(dev)
        d = cxy[None, None] - tok.foot_xy[:, :, None]                              # [B,M,C,2]
        over = ((d[..., 0] >= OVER_CELL_DX[0]) & (d[..., 0] <= OVER_CELL_DX[1]) & (d[..., 1].abs() <= OVER_CELL_DY)
                & fm[..., None])
    J, L, Fo, Ce = tok.sl("joint"), tok.sl("limb"), tok.sl("foot"), tok.sl("cell")
    E = torch.zeros(B, T, T, len(LEGGED_REL_VOCAB), dtype=torch.bool, device=dev)
    E[:, :, :, ch["same_node"]] = torch.eye(T, dtype=torch.bool, device=dev)
    E[:, J, J, ch["kin_parent"]] = same & (dj == 1)
    E[:, J, J, ch["kin_child"]] = same & (dj == -1)
    E[:, J, J, ch["same_assembly"]] = same & ~eyeN
    E[:, J, J, ch["mirror"]] = mir_jj
    E[:, L, L, ch["mirror"]] = mir
    E[:, Fo, Fo, ch["mirror"]] = mir & fm[:, :, None] & fm[:, None, :]
    E[:, J, L, ch["node_in_assembly"]] = oh
    E[:, L, L, ch["limb_adjacent"]] = adj
    E[:, L, Fo, ch["foot_of"]] = dfoot
    if C:
        E[:, Fo, Ce, ch["over_cell"]] = over
    A = torch.zeros(B, M, T, len(LEGGED_REL_VOCAB), dtype=torch.bool, device=dev)     # per-assembly act -> ctx
    A[:, :, L, ch["same_node"]] = eyeM & am[:, :, None]
    A[:, :, J, ch["node_in_assembly"]] = oh.transpose(1, 2)
    A[:, :, J, ch["same_assembly"]] = oh.transpose(1, 2)
    A[:, :, L, ch["same_assembly"]] = eyeM & am[:, :, None]
    A[:, :, Fo, ch["same_assembly"]] = dfoot
    A[:, :, J, ch["mirror"]] = (mirf @ ohf.transpose(1, 2)) > 0
    A[:, :, L, ch["mirror"]] = mir
    A[:, :, Fo, ch["mirror"]] = mir & fm[:, None, :]
    A[:, :, L, ch["limb_adjacent"]] = adj
    A[:, :, Fo, ch["foot_of"]] = dfoot
    if C:
        A[:, :, Ce, ch["over_cell"]] = over
    eK = torch.eye(K, dtype=torch.bool, device=dev)[None, :, None, :, None]        # [1,K,1,K,1]
    mm = eyeM[None, None, :, None, :]
    AA = torch.zeros(B, K, M, K, M, len(LEGGED_REL_VOCAB), dtype=torch.bool, device=dev)
    AA[..., ch["same_node"]] = eK & mm & (am[:, None, :, None, None] & am[:, None, None, None, :])
    AA[..., ch["same_assembly"]] = mm & ~eK & (am[:, None, :, None, None] & am[:, None, None, None, :])
    AA[..., ch["mirror"]] = eK & mir[:, None, :, None, :]
    AA[..., ch["limb_adjacent"]] = eK & adj[:, None, :, None, :]
    return {"ctx>ctx": E, "act>ctx": A.repeat(1, K, 1, 1), "act>act": AA.reshape(B, K * M, K * M, -1)}


def legged_graph(b, K: int, deploy: bool = False, generator=None):
    """(LeggedTokens, RelCtx) for one batch: the ctx / act token sets (the fields `pos3d` and `assembly_id`, the public
    scan / morphology geometry only), the three `legged-rel-v1` EdgeSets and, in training batches only (never when
    `deploy`), the supervision label `foothold_next` [B,T,T] built from the batch's `foothold_cell` [B,M] (-2 absent,
    -1 planted, >= 0 the scan cell of the next landing). A missing key is a masked label, not an error."""
    tok = legged_tokens(b)
    B, N, M, C, T = b["node_mask"].shape[0], tok.N, tok.M, tok.C, tok.T
    dev = b["node_mask"].device
    kind = torch.cat([torch.full((1,), TOKEN_KINDS.index("sensor")),
                      torch.full((N,), TOKEN_KINDS.index("morph_node")),
                      torch.full((M,), TOKEN_KINDS.index("assembly")), torch.full((M,), TOKEN_KINDS.index("sensor")),
                      torch.full((C,), TOKEN_KINDS.index("entity"))]).to(dev)[None].expand(B, T)
    a = b["asm_static"]
    apos = a[..., 3:6] * a[..., 8:9]
    pos, valid = torch.zeros(B, T, 3, device=dev), torch.zeros(B, T, dtype=torch.bool, device=dev)
    pos[:, tok.sl("limb")], pos[:, tok.sl("foot")] = apos, apos
    valid[:, tok.sl("limb")], valid[:, tok.sl("foot")] = b["asm_mask"], tok.foot_mask
    aid = torch.zeros(B, T, 1, device=dev, dtype=torch.long)
    aid[:, tok.sl("joint"), 0] = b["node_asm"]
    aid[:, tok.sl("limb"), 0] = torch.arange(M, device=dev)
    aid[:, tok.sl("foot"), 0] = torch.arange(M, device=dev)
    if C:
        pos[:, tok.sl("cell"), :2] = scan_xy(dev)[None]
        pos[:, tok.sl("cell"), 2] = b["terrain"]
        valid[:, tok.sl("cell")] = tok.mask[:, tok.sl("cell")]
    labels = {}
    if not deploy and "foothold_cell" in b:
        fc = b["foothold_cell"]
        y = torch.zeros(B, T, T, device=dev)
        v = torch.zeros(B, T, T, dtype=torch.bool, device=dev)
        if C:                                      # foot -> terrain-cell pairs only (no terrain scan: no label)
            v[:, tok.sl("foot"), tok.sl("cell")] = ((fc >= -1) & tok.foot_mask)[:, :, None]
            y[:, tok.sl("foot"), tok.sl("cell")] = F.one_hot(fc.clamp(min=0), C).float() * (fc >= 0)[..., None]
        labels = {"foothold_next": y, "foothold_next.valid": v}
    ctx = TokenSet("ctx", tok.mask, kind, {"pos3d": pos, "pos3d.valid": valid, "assembly_id": aid}, labels, deploy)
    qm = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
    act = TokenSet("act", qm, None, {"assembly_id": torch.arange(M, device=dev).repeat(K)[None].expand(B, -1)[..., None]},
                   {}, deploy)
    vocab = tuple(LEGGED_REL_VOCAB)
    rc = RelCtx(sets={"ctx": ctx, "act": act}, edges={s: EdgeSet(vocab, e) for s, e in legged_edges(b, tok, K).items()},
                generator=generator, deploy=deploy)
    return tok, rc


def _leg_sites(specs, heads, D, layers):
    """Per packet-block FactorSites (act>ctx, act>act); parameter-free when no factor applies."""
    return nn.ModuleList([nn.ModuleDict(dict(c=FactorSite(heads, D, "act>ctx", specs, _CARRIES["act>ctx"]),
                                             s=FactorSite(heads, D, "act>act", specs, _CARRIES["act>act"])))
                          for _ in range(layers)])


class Context(nn.Module):
    """Public context tokens: one per actuated joint (static morph + encoders) and one global token
    (IMU, osc, task view, public waypoint estimates, speed estimate); leg touch enters the assembly tokens.

    With relational factors (`specs` with an applicable ctx / act site, `legged-rel-v1` + `leg.foothold`) the token set
    is extended to [glob | joints | limbs | feet | terrain cells] (`legged_graph`) and each layer's self-attention
    carries a `ctx>ctx` FactorSite (edge biases, the foothold pair estimate). Without them (`legged-none`, the
    default) the module has neither the extra encoders nor sites: parameters, state-dict keys and outputs are those of
    the pre-relations net."""

    def __init__(self, D, heads=4, layers=2, beh_h: int = 0, specs=()):
        super().__init__()
        self.node = MLP(NODE_STATIC_DIM + 2, D)
        self.glob = MLP(GLOBAL_DIM + 6, D)
        self.beh = MLP(beh_h, D) if beh_h else None
        self.rel = relational_specs(specs)
        self.extended = bool(self.rel)
        self.deploy = False
        if self.extended:
            self.limb = MLP(ASM_DIM + 1, D)
            self.foot = MLP(4, D)
            self.cell = MLP(3, D)
        self.enc = nn.ModuleList([nn.ModuleDict(dict(
            n=nn.LayerNorm(D), a=MHA(D, heads), n2=nn.LayerNorm(D), m=MLP(D, D, 4 * D),
            **({"b": FactorSite(heads, D, "ctx>ctx", self.rel, _CARRIES["ctx>ctx"])} if self.extended else {})))
            for _ in range(layers)])

    def _tokens(self, b, beh, tok):
        x = self.node(torch.cat([b["node_static"], b["q"][..., None], b["qd"][..., None]], -1))
        if beh is not None:
            x = x + self.beh(beh)                                  # [B,N,H] demonstrated per-joint targets
        g = self.glob(torch.cat([b["ctx"], b["imu"]], -1))[:, None]
        if tok is None:
            return torch.cat([g, x], 1)
        a, touch = b["asm_static"], b["asm_touch"][..., None]
        parts = [g, x, self.limb(torch.cat([a, touch], -1)), self.foot(torch.cat([a[..., 3:6], touch], -1))]
        if tok.C:
            parts.append(self.cell(torch.cat([scan_xy(x.device)[None].expand(x.shape[0], -1, -1),
                                              b["terrain"][..., None] / 0.3], -1)))
        return torch.cat(parts, 1)

    def encode(self, b, beh=None, K: int = 1, rewire_gen=None, attn=None):
        """(tokens [B,T,D], mask [B,T], RelCtx | None). `K` is the packet knot count of the act queries the same
        RelCtx serves. `attn` (a list) collects each layer's [B,H,T,T] attention (diagnostics only)."""
        if not self.extended:
            t = self._tokens(b, beh, None)
            m = torch.cat([torch.ones_like(b["node_mask"][:, :1]), b["node_mask"]], 1)
            for L in self.enc:
                t = t + L["a"](L["n"](t), key_mask=m)
                t = t + L["m"](L["n2"](t))
            return t, m, None
        tok, rc = legged_graph(b, K, self.deploy, rewire_gen)
        t, m = self._tokens(b, beh, tok), tok.mask
        for L in self.enc:
            xn = L["n"](t)
            qa, ka = L["b"].augment(rc, xn, xn)
            r = L["a"](xn, key_mask=m, bias=L["b"].bias(rc), q_aug=qa, k_aug=ka, need_weights=attn is not None)
            if attn is not None:
                r, att = r
                attn.append(att)
            t = t + r
            t = t + L["m"](L["n2"](t))
        return t, m, rc

    def forward(self, b, beh=None):
        return self.encode(b, beh)[:2]


def _act_biases(sites, rc):
    """Per packet-block additive biases ([B,H,Q,T] cross, [B,H,Q,Q] self) of the act>ctx / act>act sites (None each
    when no factor applies). Ctx-only, so the flow computes them once per `prepare`, not once per velocity call."""
    return [(S["c"].bias(rc), S["s"].bias(rc)) for S in sites]


def _run_block(L: RelBlock, q, t, tm, qm, bx, bs, att=None):
    """`RelBlock.forward` with the act>ctx / act>act biases and optional attention capture (same modules)."""
    xn = L.n1(q)
    r = L.x(xn, kv=t, key_mask=tm, bias=bx, need_weights=att is not None)
    if att is not None:
        r, w = r
        att.append(w)
    q = q + r
    q = q + L.s(L.n2(q), key_mask=qm, bias=bs)
    return q + L.m(L.n3(q))


class AsmQueries(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.asm = MLP(ASM_DIM + 1, D)
        self.knot = nn.Embedding(K, D)

    def forward(self, b):
        a = self.asm(torch.cat([b["asm_static"], b["asm_touch"][..., None]], -1))     # [B,M,D]
        return a[:, None] + self.knot.weight[None, :, None]                            # [B,K,M,D]


class LeggedEncoder(nn.Module):
    """E: z ~ N(mu, exp(lv)) per (knot, assembly). `factors` (default `legged-none`) as `Context`; the packet blocks'
    cross / self attention then carry the `act>ctx` / `act>act` sites."""

    def __init__(self, dz=32, D=192, heads=4, layers=3, K=4, H=40, factors=None):
        super().__init__()
        self.specs = legged_specs(factors)
        self.ctx = Context(D, heads, 2, beh_h=H, specs=self.specs)
        self.q = AsmQueries(D, K)
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        if self.ctx.extended:
            self.rsites = _leg_sites(self.ctx.rel, heads, D, layers)
        self.out = nn.Linear(D, 2 * dz)
        self.dz, self.K = dz, K

    def set_deploy(self, deploy: bool = True):
        if deploy:
            assert_deployable(self.specs)
        self.ctx.deploy = deploy
        return self

    def encode(self, b, beh, rewire_gen=None, attn=None):
        """(mu, lv, RelCtx | None); `attn` collects [ctx layers..., then act>ctx / act>act per block] attention."""
        t, tm, rc = self.ctx.encode(b, beh, K=self.K, rewire_gen=rewire_gen, attn=attn)
        q = self.q(b)
        B, K, M, D = q.shape
        q = q.reshape(B, K * M, D)
        qm = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
        if rc is None:
            for L in self.blocks:
                q = L(q, t, kv_mask=tm, q_mask=qm)
        else:
            for L, (bx, bs) in zip(self.blocks, _act_biases(self.rsites, rc)):
                q = _run_block(L, q, t, tm, qm, bx, bs)
        mu, lv = self.out(q).reshape(B, K, M, 2, self.dz).unbind(3)
        am = b["asm_mask"][:, None, :, None].to(mu.dtype)
        return mu * am, lv.clamp(-8, 4), rc

    def forward(self, b, beh):
        return self.encode(b, beh)[:2]


class LeggedRealizer(nn.Module):
    """System 0. Inputs are ONLY: z, knot times, elapsed phase, morphology tokens, current encoders/IMU/touch,
    osc-v1. No task, goal, waypoint, localization or system-i state.

    The own/body-assembly routing (a node reads only its own assembly's knots, plus the body assembly's knots,
    which every node may always read) is the factor `route.own_assembly` (preset `legged-s0`, `docs/relations.md`
    section 10 R4): `SameOp` mask on field `assembly_id` between the `act` (node) and `knots` sites, with
    `params.also_key_field = "body"` for the body-assembly override. Knot validity (padded / absent assemblies)
    is the cross-attention `kv_mask`, not part of the factor value, so the combined -inf pattern is identical to
    the former inline `own & asm_mask`."""

    def __init__(self, dz=32, D=192, heads=4, layers=2, factors=None):
        super().__init__()
        self.node = MLP(NODE_STATIC_DIM + 2 + 6 + 1 + 2, D)
        self.z_in = nn.Linear(dz, D)
        self.asm = MLP(ASM_DIM, D)
        self.dt_in = MLP(D, D)
        specs = resolve(factors, default="legged-s0")
        self.route = FactorSite(heads, D, "act>knots", specs, ("assembly_id",))
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, 1)
        self.D = D

    def _route_bias(self, b, K, M, knot_valid, device):
        knot_asm = torch.arange(M, device=device).repeat(K)                                  # [K*M], k-major
        knot_asm_b = knot_asm[None, :].expand(b["node_asm"].shape[0], -1)
        body_field = knot_asm_b == b["body_asm"][:, None]
        rc = RelCtx(sets={
            "act": TokenSet("act", b["node_mask"], fields={"assembly_id": b["node_asm"]}),
            "knots": TokenSet("knots", knot_valid, fields={"assembly_id": knot_asm_b, "body": body_field}),
        })
        return self.route.bias(rc)

    def forward(self, z, b, phase, knot_times=None):
        B, K, M, _ = z.shape
        N = b["node_static"].shape[1]
        kt = torch.as_tensor(KNOT_TIMES if knot_times is None else knot_times, dtype=z.dtype, device=z.device)
        rel = kt[None] - phase[:, None]
        tok = self.z_in(z) + self.dt_in(sinusoidal(rel, self.D))[:, :, None] + self.asm(b["asm_static"])[:, None]
        tok = tok.reshape(B, K * M, -1)
        knot_valid = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
        bias = self._route_bias(b, K, M, knot_valid, z.device)
        nt = torch.gather(b["asm_touch"], 1, b["node_asm"])                                  # own foot touch
        osc = torch.stack([torch.sin(2 * math.pi * b["osc"]), torch.cos(2 * math.pi * b["osc"])], -1)
        x = self.node(torch.cat([b["node_static"], b["q"][..., None], b["qd"][..., None],
                                 b["imu"][:, None].expand(-1, N, -1), nt[..., None], osc[:, None].expand(-1, N, -1)], -1))
        for L in self.blocks:
            x = L(x, tok, kv_mask=knot_valid, q_mask=b["node_mask"], bias_x=bias)
        return self.out(x).squeeze(-1) * b["node_mask"]


def legged_probe(dz=32, K=4, D=128, heads=4, max_m=MAX_M, metadata_only=False, seed=1234, factors=None) -> ReadoutProbe:
    """The packet probe (former `LeggedProbe`): a `ReadoutProbe` on preset `probes:legged-v1` (docs/relations.md
    section 10 R4). `goal` / `disp` / `subtask` / `fall` read every assembly (address "asm"); `legged_probe_read`
    below gathers the sample's body-assembly row, which is exactly the former per-sample query
    `code(asm_code[body_asm])`, since "asm" applies the same `asm_in` map to every assembly and cross-attention
    queries are independent of each other. Old `LeggedProbe` checkpoints load through `remap_legged_probe_state`
    (a data-level key map, docs/relations.md section 4)."""
    return ReadoutProbe(dz, K, specs=factors, width=D, heads=heads, max_assemblies=max_m,
                        metadata_only=metadata_only, seed=seed, preset="probes:legged-v1")


def build_legged_rep(lc: dict, factors=None, H: int = H):
    """(E, R, P) of a latent config `lc` (`dz`, `width`) on `factors` (the resolved list holding the probe queries AND
    the relation factors; None = pre-relations layout: `legged-none` + the default probe preset)."""
    specs = legged_specs(factors)
    E = LeggedEncoder(dz=lc["dz"], D=lc["width"], H=H, factors=specs)
    R = LeggedRealizer(dz=lc["dz"], D=lc["width"])
    P = legged_probe(dz=lc["dz"], factors=None if factors is None else specs)
    return E, R, P


def legged_probe_read(P: ReadoutProbe, z, asm_mask, body_asm) -> dict:
    """P(z, asm_mask) narrowed to the sample level: every `knot×asm` query (contact) squeezed to [B,K,M], every `asm`
    query (goal [B,4], disp [B,6], subtask [B,N_SUBTASK], fall [B,1], and `com_support` [B,2] = (mu, logvar) when that
    factor is on) gathered at the sample's body-assembly row."""
    out = P(z, asm_mask)
    idx1 = body_asm.view(-1, 1, 1)
    res = {}
    for _, r in P.queries:
        o = out[r.query]
        res[r.query] = o.squeeze(-1) if r.address == "knot×asm" else o.gather(1, idx1.expand(-1, -1, o.shape[-1])).squeeze(1)
    return res


def probe_terms(out: dict, lab: dict, b) -> tuple[dict, dict, dict]:
    """(predictions, labels, masks) in the layout `probes.readout_loss` / `readout_metrics` take, from
    `legged_probe_read`'s output, the batch labels `lab` and the batch `b`. contact gets a trailing 1 (its bce squeezes
    it) and the leg mask; `goal` its validity; `com_support` its label validity (a missing label key is a masked
    label: absent from the returned dicts, so no term)."""
    pred = dict(out)
    if "contact" in pred:
        pred["contact"] = pred["contact"][..., None]
    labels = {"contact_k": lab["contact_k"].float(), "goal": lab["goal"], "disp": lab["disp"], "subtask": lab["subtask"],
              "fall": lab["fall"].float()}
    masks = {"contact": b["asm_is_leg"][:, None, :].expand(-1, out["contact"].shape[1], -1) if "contact" in out else None,
             "goal": lab["goal_valid"].bool()}
    if "com_support" in lab:
        labels["com_support"] = lab["com_support"]
        masks["com_support"] = lab["com_support_valid"].bool()
    return pred, labels, {k: v for k, v in masks.items() if v is not None}


def remap_legged_probe_state(state: dict) -> dict:
    """Data-level key map (docs/relations.md section 4, unit R4): a former `LeggedProbe` state dict onto
    `ReadoutProbe`'s layout. `code` / `kcode` (Linear maps of the fixed `asm_code` / `knot_code` buffers, reused for
    both the packet content tokens and the query positions) become `asm_in` / `kq_in` (the same op on the same
    buffer); the packet content's knot term `kcode(knot_code)` is baked once into `ReadoutProbe.knot.weight` (a
    plain per-knot embedding there rather than a linear map of the frozen code), so both probes compute the
    identical packet content and query features from the same weights. `z_in` / `qtype` / `const` / `asm_code` /
    `knot_code` / the two attention layers (`a1`, `a2` -> `att`, `att2`) / `n1` / `n2` / `mlp` / `heads.*` carry over
    unchanged (`probes:legged-v1`'s query order, contact/goal/disp/subtask/fall, matches the old `qtype` index and
    `heads` names). A no-op on a state dict already in the `ReadoutProbe` layout."""
    if "asm_in.weight" in state:
        return dict(state)
    s = dict(state)
    knot_code = s.pop("knot_code")
    kcode_w, kcode_b = s.pop("kcode.weight"), s.pop("kcode.bias")
    out = {"z_in.weight": s.pop("z_in.weight"), "z_in.bias": s.pop("z_in.bias"),
           "knot.weight": knot_code @ kcode_w.T + kcode_b,
           "asm_in.weight": s.pop("code.weight"), "asm_in.bias": s.pop("code.bias"),
           "kq_in.weight": kcode_w, "kq_in.bias": kcode_b,
           "qtype.weight": s.pop("qtype.weight"), "const": s.pop("const"),
           "asm_code": s.pop("asm_code"), "knot_code": knot_code}
    for old, new in (("a1", "att"), ("a2", "att2"), ("n1", "n1"), ("n2", "n2"), ("mlp", "mlp")):
        for k in [k for k in s if k.startswith(old + ".")]:
            out[new + k[len(old):]] = s.pop(k)
    for k in [k for k in s if k.startswith("heads.")]:
        out[k] = s.pop(k)
    if s:
        raise ValueError(f"remap_legged_probe_state: unmapped legacy keys {sorted(s)}")
    return out


@torch.no_grad()
def probe_metrics(out, lab, b):
    """Raw (sum, count) pairs for aggregation."""
    lm = b["asm_is_leg"][:, None, :].expand_as(out["contact"])
    y = lab["contact_k"].bool()
    pred = out["contact"] > 0
    res = dict(contact_acc=(int(((pred == y) & lm).sum()), int(lm.sum())),
               contact_swing_acc=(int(((pred == y) & lm & ~y).sum()), int((lm & ~y).sum())))
    gv = lab["goal_valid"]
    ge = (out["goal"][:, :2] - lab["goal"]).norm(dim=-1)
    res["goal_err"] = (float((ge * gv).sum()), int(gv.sum()))
    de = (out["disp"][:, :2] - lab["disp"][:, :2]).norm(dim=-1)
    res["disp_xy_err"] = (float(de.sum()), len(de))
    res["disp_yaw_err"] = (float((out["disp"][:, 2] - lab["disp"][:, 2]).abs().sum()), len(de))
    res["subtask_acc"] = (int((out["subtask"].argmax(-1) == lab["subtask"]).sum()), len(de))
    fp = out["fall"][:, 0] > 0
    res["fall_acc"] = (int((fp == lab["fall"].bool()).sum()), len(de))
    return res


@dataclass
class FlowCache:
    """`LeggedFlow.prepare`: the public context tokens, the act queries, the per-block act biases and the forward's
    RelCtx (None without relational factors; its estimates feed `estimates_loss`)."""
    t: Any
    tm: Any
    q0: Any
    rc: Any = None
    bias: Any = None


class LeggedFlow(nn.Module):
    """System i: rectified flow over the standardized packet, conditioned on PUBLIC context only. `factors` (default
    `legged-none`) as `LeggedEncoder`; a `leg.foothold` pair estimate is then supervised through `estimates_loss`."""

    def __init__(self, dz=32, D=256, heads=4, layers=4, K=4, factors=None):
        super().__init__()
        self.specs = legged_specs(factors)
        self.ctx = Context(D, heads, 2, specs=self.specs)
        self.q = AsmQueries(D, K)
        self.z_in = nn.Linear(dz, D)
        self.t_in = MLP(D, D)
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        if self.ctx.extended:
            self.rsites = _leg_sites(self.ctx.rel, heads, D, layers)
        self.out = nn.Linear(D, dz)
        self.register_buffer("z_mean", torch.zeros(dz))
        self.register_buffer("z_std", torch.ones(dz))
        self.K, self.dz = K, dz

    def set_deploy(self, deploy: bool = True):
        if deploy:
            assert_deployable(self.specs)
        self.ctx.deploy = deploy
        return self

    def prepare(self, b, rewire_gen=None, attn=None) -> FlowCache:
        t, tm, rc = self.ctx.encode(b, None, K=self.K, rewire_gen=rewire_gen, attn=attn)
        return FlowCache(t, tm, self.q(b), rc, None if rc is None else _act_biases(self.rsites, rc))

    def velocity(self, zt, tau, cache, b, attn=None):
        B, K, M, _ = zt.shape
        h = cache.q0 + self.z_in(zt) + self.t_in(sinusoidal(tau, cache.q0.shape[-1]))[:, None, None]
        h = h.reshape(B, K * M, -1)
        qm = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
        if cache.rc is None:
            for L in self.blocks:
                h = L(h, cache.t, kv_mask=cache.tm, q_mask=qm)
        else:
            for L, (bx, bs) in zip(self.blocks, cache.bias):
                h = _run_block(L, h, cache.t, cache.tm, qm, bx, bs, attn)
        return self.out(h).reshape(B, K, M, -1)

    def normalize(self, z):
        return (z - self.z_mean) / self.z_std

    def denormalize(self, z):
        return z * self.z_std + self.z_mean

    def loss(self, b, z_target, probe_fn=None, w_sem=0.0, tau_min=0.6):
        cache = self.prepare(b)
        x1 = self.normalize(z_target)
        am = b["asm_mask"][:, None, :, None].to(x1.dtype)
        eps = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=x1.device)
        t_ = tau[:, None, None, None]
        zt = ((1 - t_) * eps + t_ * x1) * am
        v = self.velocity(zt, tau, cache, b)
        fl = (((v - (x1 - eps)) ** 2) * am).sum() / (am.sum() * x1.shape[-1])
        logs = dict(flow=float(fl.detach()))
        loss = fl
        if probe_fn is not None and w_sem > 0:
            zc = self.denormalize(zt + (1 - t_) * v) * am
            keep = (tau >= tau_min)[:, None, None, None]
            zc = torch.where(keep, zc, zc.detach())
            pl, pl_logs = probe_fn(zc)
            loss = loss + w_sem * pl
            logs.update({f"zhat_{k}": x for k, x in pl_logs.items()})
        if cache.rc is not None and cache.rc.estimates and not cache.rc.deploy:
            el, elogs, _ = estimates_loss(cache.rc, self.specs)        # the factors' pair estimates vs their labels
            loss = loss + el
            logs.update(elogs)
        return loss, logs

    @torch.no_grad()
    def sample(self, b, nfe=8, generator=None):
        cache = self.prepare(b)
        B, M = b["asm_mask"].shape
        am = b["asm_mask"][:, None, :, None].float()
        z = torch.randn(B, self.K, M, self.dz, device=am.device, generator=generator) * am
        for k in range(nfe):
            tau = torch.full((B,), k / nfe, device=z.device)
            z = z + (1.0 / nfe) * self.velocity(z, tau, cache, b)
        return self.denormalize(z) * am
