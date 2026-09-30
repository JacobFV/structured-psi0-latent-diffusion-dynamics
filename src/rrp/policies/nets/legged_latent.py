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
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.features.legged import NODE_STATIC_DIM, ASM_DIM, GLOBAL_DIM, KNOT_TIMES, MAX_M
from rrp.policies.nets.attention import MHA, RelBlock
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.probes import ReadoutProbe
from rrp.policies.relations.base import RelCtx, TokenSet, resolve
from rrp.policies.relations.ops import FactorSite

N_SUBTASK = 4


class Context(nn.Module):
    """Public context tokens: one per actuated joint (static morph + encoders) and one global token
    (IMU, osc, task view, public waypoint estimates, speed estimate); leg touch enters the assembly tokens."""

    def __init__(self, D, heads=4, layers=2, beh_h: int = 0):
        super().__init__()
        self.node = MLP(NODE_STATIC_DIM + 2, D)
        self.glob = MLP(GLOBAL_DIM + 6, D)
        self.beh = MLP(beh_h, D) if beh_h else None
        self.enc = nn.ModuleList([nn.ModuleDict(dict(n=nn.LayerNorm(D), a=MHA(D, heads), n2=nn.LayerNorm(D),
                                                     m=MLP(D, D, 4 * D))) for _ in range(layers)])

    def forward(self, b, beh=None):
        x = self.node(torch.cat([b["node_static"], b["q"][..., None], b["qd"][..., None]], -1))
        if beh is not None:
            x = x + self.beh(beh)                                  # [B,N,H] demonstrated per-joint targets
        g = self.glob(torch.cat([b["ctx"], b["imu"]], -1))[:, None]
        t = torch.cat([g, x], 1)
        m = torch.cat([torch.ones_like(b["node_mask"][:, :1]), b["node_mask"]], 1)
        for L in self.enc:
            t = t + L["a"](L["n"](t), key_mask=m)
            t = t + L["m"](L["n2"](t))
        return t, m


class AsmQueries(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.asm = MLP(ASM_DIM + 1, D)
        self.knot = nn.Embedding(K, D)

    def forward(self, b):
        a = self.asm(torch.cat([b["asm_static"], b["asm_touch"][..., None]], -1))     # [B,M,D]
        return a[:, None] + self.knot.weight[None, :, None]                            # [B,K,M,D]


class LeggedEncoder(nn.Module):
    def __init__(self, dz=32, D=192, heads=4, layers=3, K=4, H=40):
        super().__init__()
        self.ctx = Context(D, heads, 2, beh_h=H)
        self.q = AsmQueries(D, K)
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, 2 * dz)
        self.dz = dz

    def forward(self, b, beh):
        t, tm = self.ctx(b, beh)
        q = self.q(b)
        B, K, M, D = q.shape
        q = q.reshape(B, K * M, D)
        qm = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
        for L in self.blocks:
            q = L(q, t, kv_mask=tm, q_mask=qm)
        mu, lv = self.out(q).reshape(B, K, M, 2, self.dz).unbind(3)
        am = b["asm_mask"][:, None, :, None].to(mu.dtype)
        return mu * am, lv.clamp(-8, 4)


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


def legged_probe_read(P: ReadoutProbe, z, asm_mask, body_asm) -> dict:
    """P(z, asm_mask) narrowed to the former LeggedProbe output: contact [B,K,M], goal [B,4], disp [B,6],
    subtask [B,N_SUBTASK], fall [B,1] (gathered at each sample's body-assembly index)."""
    out = P(z, asm_mask)
    idx1 = body_asm.view(-1, 1, 1)
    return dict(contact=out["contact"].squeeze(-1),
               **{k: out[k].gather(1, idx1.expand(-1, -1, out[k].shape[-1])).squeeze(1)
                  for k in ("goal", "disp", "subtask", "fall")})


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


def gnll(pred, target, mask=None, lv_min=-8.0):
    d = target.shape[-1]
    mu, lv = pred[..., :d], pred[..., d:].clamp(lv_min, 6)
    nll = 0.5 * (((target - mu) ** 2) / lv.exp() + lv + math.log(2 * math.pi)).sum(-1)
    if mask is None:
        return nll.mean()
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def probe_loss(out, lab, b, lv_min=-8.0):
    lm = (b["asm_is_leg"][:, None, :].expand_as(out["contact"])).float()
    L = dict(contact=(F.binary_cross_entropy_with_logits(out["contact"], lab["contact_k"].float(), reduction="none")
                      * lm).sum() / lm.sum().clamp(min=1),
             goal=gnll(out["goal"], lab["goal"], lab["goal_valid"], lv_min=lv_min),
             disp=gnll(out["disp"], lab["disp"], lv_min=lv_min),
             subtask=F.cross_entropy(out["subtask"], lab["subtask"]),
             fall=F.binary_cross_entropy_with_logits(out["fall"][:, 0], lab["fall"].float()))
    return sum(L.values()), {f"probe_{k}": float(v.detach()) for k, v in L.items()}


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


class LeggedFlow(nn.Module):
    """System i: rectified flow over the standardized packet, conditioned on PUBLIC context only."""

    def __init__(self, dz=32, D=256, heads=4, layers=4, K=4):
        super().__init__()
        self.ctx = Context(D, heads, 2)
        self.q = AsmQueries(D, K)
        self.z_in = nn.Linear(dz, D)
        self.t_in = MLP(D, D)
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, dz)
        self.register_buffer("z_mean", torch.zeros(dz))
        self.register_buffer("z_std", torch.ones(dz))
        self.K, self.dz = K, dz

    def prepare(self, b):
        t, tm = self.ctx(b)
        return t, tm, self.q(b)

    def velocity(self, zt, tau, cache, b):
        t, tm, q0 = cache
        B, K, M, _ = zt.shape
        h = q0 + self.z_in(zt) + self.t_in(sinusoidal(tau, q0.shape[-1]))[:, None, None]
        h = h.reshape(B, K * M, -1)
        qm = b["asm_mask"][:, None].expand(B, K, M).reshape(B, K * M)
        for L in self.blocks:
            h = L(h, t, kv_mask=tm, q_mask=qm)
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
