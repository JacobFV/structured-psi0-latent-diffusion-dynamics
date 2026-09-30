"""Pointer nets (ComputerWorld `cw_pointer`; docs/architecture.md 3, 5): public context encoder `UICtx`, representation
encoder E, learned system 0 (realizer) R, system i flow S and the BC baseline.

Imports torch at module level; `rrp.policies.pointer` stays torch-free because it reaches these classes only through
`rrp.policies.pointer.nets()` (lazy import). Widget / instruction / event sizes and the relation-factor config live in
`rrp.policies.pointer.spec`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.envs.computerworld import UI_REL_VOCAB
from rrp.policies.nets.attention import MHA, RelBlock
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.pointer_vocab import (KNOT_TIMES, LC, LI, N_BOUND, N_KEYCLS, N_ROLE, N_SYM, NW, POINTER_FACTORS_PRESET,
                                           UI_CARRIES, WF)
from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet, get_factor, resolve
from rrp.policies.relations.ops import FactorSite


def _zero_mha_out(m: MHA) -> MHA:
    """Zero-init an MHA's output projection so it contributes exactly 0 to a residual sum: the same
    zero-bias-equivalence idiom docs/relations.md 3.2 uses for factor gates/aug, reused here (D-144 R6) for an
    entire unused `RelBlock` attention stage."""
    nn.init.zeros_(m.o.weight)
    nn.init.zeros_(m.o.bias)
    return m


def cross_relblock(D, heads) -> RelBlock:
    """`RelBlock` standing in for the former pointer `Block(cross=True)` (cross-attention + MLP only, no
    self-attention among the query tokens): the unused self stage (`.s`) is zero-init-output, an exact no-op
    both fresh (bit-identical to the old 2-stage forward) and from an old checkpoint (nothing is loaded into
    `.s`, so it stays zero-init -- `rrp.policies.pointer.checkpoint`)."""
    b = RelBlock(D, heads)
    _zero_mha_out(b.s)
    return b


def self_relblock(D, heads) -> RelBlock:
    """`RelBlock` standing in for the former pointer `Block(cross=False)` (self-attention + MLP only): the
    unused cross stage (`.x`) is zero-init-output, same reasoning as `cross_relblock`."""
    b = RelBlock(D, heads)
    _zero_mha_out(b.x)
    return b


class LabelBag(nn.Module):
    """Widget label -> one vector: mean of char and (char, position) embeddings (embedding_bag: the [B, NW, LC, D]
    tensor is never materialized)."""

    def __init__(self, D):
        super().__init__()
        self.c = nn.Embedding(N_SYM, D, padding_idx=0)
        self.cp = nn.Embedding(N_SYM * LC + 1, D, padding_idx=0)
        self.register_buffer("pos", torch.arange(LC))

    def forward(self, ch):
        B, W, L = ch.shape
        c = ch.reshape(B * W, L)
        cp = torch.where(c > 0, c * LC + self.pos + 1, torch.zeros_like(c))
        e = F.embedding_bag(c, self.c.weight, mode="mean", padding_idx=0) + \
            F.embedding_bag(cp, self.cp.weight, mode="mean", padding_idx=0)
        return e.reshape(B, W, -1)


def drag_to_label(b):
    """`ui.drag_to`'s pair label from the PUBLIC batch fields alone (`relgen.ui.drag_to_fn`'s definition, evaluated on
    what a policy sees): the focused widget (`wfocusrank == 0`, CW's only public signal of the widget being interacted
    with) -> the nearest OTHER widget by `wpos3d`. -> (y [B, NW, NW] float, valid [B, NW, NW] bool); every pair of
    present widgets is a scored candidate, only the focused widget's row has a 1, and only when another widget exists.
    A batch without stored geometry (`wgeo_ok` false) yields no valid pair."""
    wmask, pos = b["wmask"], b["wpos3d"]
    ok = wmask & b.get("wgeo_ok", wmask)
    foc = ok & (b["wfocusrank"] == 0) if "wfocusrank" in b else torch.zeros_like(ok)
    d = (pos[:, :, None] - pos[:, None]).norm(dim=-1)
    eye = torch.eye(ok.shape[1], dtype=torch.bool, device=ok.device)[None]
    cand = ok[:, None, :] & ok[:, :, None] & ~eye
    nearest = torch.where(cand, d, torch.full_like(d, float("inf"))).argmin(-1)              # [B, NW] per row
    has = cand.any(-1)
    y = (F.one_hot(nearest, ok.shape[1]).bool() & (foc & has)[..., None]).float()
    return y, ok[:, :, None] & ok[:, None, :]


class UICtx(nn.Module):
    """Public context tokens: NW widget tokens (label chars + role + bound entity + geometry, pointer-relative
    centre), LI instruction characters, NH own-event tokens and one proprio token; `layers` self-attention
    blocks. `specs` (D-144 R20 follow-up, docs/relations.md 10 row R6's own lead_question): resolved
    `FactorSpec`s (`PolicyConfig(factors=...).specs()`) applied at the widget self-attention (`ctx>ctx`) via a
    `TokenSet`/`RelCtx` built from R20's public UI fields/edges + the screen-geometry fields (`_relctx`); the
    default `()` (== `resolve(None, default=POINTER_FACTORS_PRESET)`, the empty preset) makes every per-layer
    `FactorSite` parameter-free, so `.bias()` / `.augment()` return `None` / `(None, None)` and an existing
    checkpoint loads and runs byte-identically."""

    def __init__(self, D=128, heads=4, layers=3, specs=()):
        super().__init__()
        self.specs = tuple(specs)
        # C1: `ui.drag_to` (probe source) is supervised by `drag_to_label(b)`, attached to the `ctx` token set only when
        # a spec asks for that label; `record_rc` keeps the last forward's `RelCtx` (`last_rc`) so the trainer can run
        # `relations.estimates_loss` on the estimates the forward wrote (the head is never silently left untrained)
        self.wants_drag_to = any(s.control != "off" and get_factor(s.name).label == "drag_to" for s in self.specs)
        self.record_rc, self.last_rc = False, None
        self.sym = nn.Embedding(N_SYM, D)
        self.cpos = nn.Embedding(LI, D)
        self.bag = LabelBag(D)
        self.role, self.bound = nn.Embedding(N_ROLE, D), nn.Embedding(N_BOUND, D)
        self.lab, self.wf = MLP(D, D), MLP(WF + 2, D)
        self.hk, self.hf = nn.Embedding(4, D), MLP(5, D)
        self.prop = MLP(4, D)
        self.typ = nn.Embedding(4, D)
        self.blocks = nn.ModuleList([self_relblock(D, heads) for _ in range(layers)])
        self.rel = nn.ModuleList([FactorSite(heads, D, "ctx>ctx", specs, UI_CARRIES) for _ in range(layers)])
        self.D = D

    def label_tokens(self, b):
        return self.lab(self.bag(b["wch"]))

    def _relctx(self, b, T):
        """Widget-token `TokenSet`/`RelCtx` at the `ctx>ctx` self-attention site: R20's `ui-rel-v1` edges
        (`b["wuiedges"]`, all-zero -- a no-op -- when the batch has none, e.g. no `table` was passed to
        `widget_features`) and the screen-geometry fields `pos3d` / `cam_uvd` / `zlayer` / `parent_id`
        (`b["wpos3d"]` etc., `ui.same_window` / `ui.above`'s own `same` / `order` ops over the LAST two), padded
        past the NW widget slots to the full `[instr, hist, prop]` token count `T` this site's self-attention
        actually spans (those tokens carry no widget geometry / UI-graph membership; `parent_id` pads with -1,
        `same_window`'s own "never matches" sentinel, matching `ui_public_fields`' null-slot convention)."""
        wmask = b["wmask"]
        B, device, pad = wmask.shape[0], wmask.device, T - NW
        pos3d = F.pad(b["wpos3d"], (0, 0, 0, pad))
        camuvd = F.pad(b["wcamuvd"], (0, 0, 0, pad))
        zlayer = F.pad(b.get("wzlayer", torch.zeros(B, NW, device=device)), (0, pad))[..., None]
        parent = F.pad(b.get("wparent", torch.full((B, NW), -1, dtype=torch.long, device=device)),
                       (0, pad), value=-1)[..., None]
        mask = torch.cat([wmask, torch.ones(B, pad, dtype=torch.bool, device=device)], 1)
        geo_ok = b.get("wgeo_ok", wmask)        # a pack without stored geometry (zeros) says so: `Demos.batch`
        geo_ok = F.pad(geo_ok, (0, pad))
        labels = {}
        if self.wants_drag_to:
            y, yv = drag_to_label(b)
            labels = {"drag_to": F.pad(y, (0, pad, 0, pad)), "drag_to.valid": F.pad(yv, (0, pad, 0, pad))}
        ts = TokenSet("ctx", mask, fields={"pos3d": pos3d, "pos3d.valid": geo_ok, "cam_uvd": camuvd,
                                           "cam_uvd.valid": geo_ok, "zlayer": zlayer, "parent_id": parent},
                      labels=labels)
        wuiedges = b.get("wuiedges")
        edges = (torch.zeros(B, NW, NW, len(UI_REL_VOCAB), device=device) if wuiedges is None
                 else wuiedges.to(pos3d.dtype))
        edges = F.pad(edges, (0, 0, 0, pad, 0, pad))
        return RelCtx(sets={"ctx": ts}, edges={"ctx>ctx": EdgeSet(UI_REL_VOCAB, edges)})

    def forward(self, b):
        B = b["ptr"].shape[0]
        rel = b["wf"][..., :2] - b["ptr"][:, None]
        w = (self.label_tokens(b) + self.role(b["wrole"]) + self.bound(b["wbound"])
             + self.wf(torch.cat([b["wf"], rel], -1)) + self.typ.weight[0])
        ins = self.sym(b["instr"]) + self.cpos.weight[:LI] + self.typ.weight[1]
        h = b["hist"]
        hr = h[..., 2:4] - b["ptr"][:, None]
        ht = (self.hk(h[..., 0].long()) + self.sym(h[..., 1].long())
              + self.hf(torch.cat([h[..., 2:4], hr, h[..., 4:5] / 20.0], -1))
              + self.typ.weight[2])
        p = self.prop(torch.stack([b["ptr"][:, 0], b["ptr"][:, 1], b["btn"], b["tick"] / 100.0], -1))[:, None] \
            + self.typ.weight[3]
        x = torch.cat([w, ins, ht, p], 1)
        m = torch.cat([b["wmask"], b["instr"] > 0, h[..., 0] > 0, torch.ones(B, 1, dtype=torch.bool,
                                                                            device=x.device)], 1)
        rc = self._relctx(b, x.shape[1])
        self.last_rc = rc if self.record_rc else None
        for L, site in zip(self.blocks, self.rel):
            xn = L.n2(x)                                  # the exact pre-self-attention hidden (`.x` is a
                                                            # zero-init no-op, docs 3.2, so this equals n2 of L's
                                                            # own input): kernel-compatible q/k augmentation
            bias_s, aug_s = site.bias(rc), site.augment(rc, xn, xn)
            x = L(x, kv=x, q_mask=m, bias_s=bias_s, aug_s=aug_s)
        return x, m


class Knots(nn.Module):
    """K knot queries (M = 1 assembly: the pointer tool)."""

    def __init__(self, D, K=len(KNOT_TIMES)):
        super().__init__()
        self.q = nn.Parameter(torch.randn(K, D) * 0.02)
        self.t = MLP(D, D)
        self.register_buffer("kt", torch.tensor(KNOT_TIMES, dtype=torch.float32))

    def forward(self, B):
        return (self.q + self.t(sinusoidal(self.kt, self.q.shape[1])))[None].expand(B, -1, -1)


def tick_tokens(D):
    return nn.ModuleDict(dict(f=MLP(6, D), key=nn.Embedding(N_KEYCLS, D), j=nn.Embedding(8, D)))


def embed_ticks(mod, a):
    """a: dict of demo chunk tensors dxy [B,H,2], xy [B,H,2], btn [B,H], key [B,H] (class), valid [B,H]."""
    H = a["btn"].shape[1]
    x = mod["f"](torch.cat([a["dxy"], a["xy"], a["btn"][..., None], a["valid"][..., None].float()], -1))
    return x + mod["key"](a["key"]) + mod["j"].weight[:H]


class PointerEncoder(nn.Module):
    """E (training only): public context at t + demonstrated tick commands a[t : t+7] -> z [B, K, 1, dz]."""

    def __init__(self, dz=16, D=128, heads=4, layers=2, factors=None):
        super().__init__()
        self.ctx = UICtx(D, heads, 2, specs=resolve(factors, default=POINTER_FACTORS_PRESET))
        self.knots, self.ticks = Knots(D), tick_tokens(D)
        self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, 2 * dz)
        self.dz = dz

    def forward(self, b, a):
        t, m = self.ctx(b)
        kv = torch.cat([t, embed_ticks(self.ticks, a)], 1)
        km = torch.cat([m, a["valid"]], 1)
        q = self.knots(t.shape[0])
        for L in self.blocks:
            q = L(q, kv=kv, kv_mask=km)
        mu, lv = self.out(q).chunk(2, -1)
        return mu[:, :, None], lv.clamp(-8, 4)[:, :, None]


class PointerRealizer(nn.Module):
    """Learned system 0. Inputs ONLY: z, knot times, phase since valid_from, measured pointer (x, y), button.
    Outputs for this tick: pointer step (dxy / 60 px), button logit, key logits (0 = none)."""

    def __init__(self, dz=16, D=128, heads=4, layers=2):
        super().__init__()
        self.z_in, self.dt = nn.Linear(dz, D), MLP(D, D)
        self.loc = MLP(3, D)
        self.ph = MLP(D, D)
        self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
        self.xy, self.btn, self.key = nn.Linear(D, 2), nn.Linear(D, 1), nn.Linear(D, N_KEYCLS)
        self.register_buffer("kt", torch.tensor(KNOT_TIMES, dtype=torch.float32))
        self.D = D

    def forward(self, z, phase, ptr, btn):
        tok = self.z_in(z[:, :, 0]) + self.dt(sinusoidal(self.kt[None] - phase[:, None], self.D))
        q = (self.loc(torch.cat([ptr, btn[:, None]], -1)) + self.ph(sinusoidal(phase, self.D)))[:, None]
        for L in self.blocks:
            q = L(q, kv=tok)
        q = q[:, 0]
        return self.xy(q), self.btn(q)[:, 0], self.key(q)


class PointerFlow(nn.Module):
    """System i: rectified flow over the standardized packet, conditioned on the public context only."""

    def __init__(self, dz=16, D=128, heads=4, layers=3, factors=None):
        super().__init__()
        self.ctx = UICtx(D, heads, 3, specs=resolve(factors, default=POINTER_FACTORS_PRESET))
        self.knots = Knots(D)
        self.z_in, self.t_in = nn.Linear(dz, D), MLP(D, D)
        self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
        self.self_blocks = nn.ModuleList([self_relblock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, dz)
        self.register_buffer("z_mean", torch.zeros(dz))
        self.register_buffer("z_std", torch.ones(dz))
        self.dz = dz

    def velocity(self, zt, tau, cache):
        t, m = cache
        h = self.knots(zt.shape[0]) + self.z_in(zt[:, :, 0]) + self.t_in(sinusoidal(tau, t.shape[-1]))[:, None]
        for X, S in zip(self.blocks, self.self_blocks):
            h = X(h, kv=t, kv_mask=m)
            h = S(h, kv=h)
        return self.out(h)[:, :, None]

    def loss(self, b, z_target, probe_fn=None, w_sem=0.0, tau_min=0.6):
        cache = self.ctx(b)
        x1 = (z_target - self.z_mean) / self.z_std
        eps = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=x1.device)
        t_ = tau[:, None, None, None]
        zt = (1 - t_) * eps + t_ * x1
        v = self.velocity(zt, tau, cache)
        fl = ((v - (x1 - eps)) ** 2).mean()
        logs = dict(flow=float(fl.detach()))
        loss = fl
        if probe_fn is not None and w_sem > 0:
            zc = (zt + (1 - t_) * v) * self.z_std + self.z_mean
            keep = (tau >= tau_min)[:, None, None, None]
            pl, pl_logs = probe_fn(torch.where(keep, zc, zc.detach()))
            loss = loss + w_sem * pl
            logs.update({f"zhat_{k}": x for k, x in pl_logs.items()})
        return loss, logs

    @torch.no_grad()
    def sample(self, b, nfe=8, noise=None):
        """Euler sample of the packet. `noise` [B, K, 1, dz] is the initial flow noise (rollout: one row per env and
        packet, `rrp.policies.pointer.runtime.flow_noise`, so a row never depends on the rest of the batch); None
        draws fresh unseeded noise (validation / probes)."""
        cache = self.ctx(b)
        B = b["ptr"].shape[0]
        shape = (B, len(KNOT_TIMES), 1, self.dz)
        if noise is None:
            z = torch.randn(shape, device=b["ptr"].device)
        else:
            if tuple(noise.shape) != shape:
                raise ValueError(f"flow noise shape {tuple(noise.shape)} != {shape}")
            z = noise.to(b["ptr"].device, torch.float32)
        for k in range(nfe):
            tau = torch.full((B,), k / nfe, device=z.device)
            z = z + self.velocity(z, tau, cache) / nfe
        return z * self.z_std + self.z_mean


class PointerBC(nn.Module):
    """BC baseline (same public inputs): 7 tick queries -> absolute pointer xy (normalized), button, key logits."""

    def __init__(self, D=128, heads=4, layers=3, H=7, factors=None):
        super().__init__()
        self.ctx = UICtx(D, heads, 3, specs=resolve(factors, default=POINTER_FACTORS_PRESET))
        self.q = nn.Parameter(torch.randn(H, D) * 0.02)
        self.blocks = nn.ModuleList([cross_relblock(D, heads) for _ in range(layers)])
        self.self_blocks = nn.ModuleList([self_relblock(D, heads) for _ in range(layers)])
        self.xy, self.btn, self.key = nn.Linear(D, 2), nn.Linear(D, 1), nn.Linear(D, N_KEYCLS)

    def forward(self, b):
        t, m = self.ctx(b)
        h = self.q[None].expand(t.shape[0], -1, -1)
        for X, S in zip(self.blocks, self.self_blocks):
            h = X(h, kv=t, kv_mask=m)
            h = S(h, kv=h)
        return self.xy(h), self.btn(h)[..., 0], self.key(h)
