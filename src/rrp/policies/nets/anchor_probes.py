"""Anchor-relative packet probes (W12; query version "probe-anchor-v1").

Same rules as rrp.models.latent_probes.PacketProbe: the probe sees ONLY the received packet z [B,K,M,dz], the query
type and opaque FIXED random handle codes (knot index, packet slot, contact-pair index). No labels, features or
descriptors enter a query. `metadata_only=True` is the no-z control probe.

Queries (per knot k and packet slot m unless stated; labels from rrp.data.contact_segments.anchor_relative_targets,
privileged, supervision only):
  tcp_in_own(k, m)       TCP pose in m's own most recent contact-anchor frame   (Gaussian over pos dm + 6D rot)
  tcp_in_support(k, m)   TCP pose in the maintained support anchor frame        (Gaussian, 9-d)
  held_in_tcp(k, m)      pose of the object m holds, in m's TCP frame           (Gaussian, 9-d)
  held_in_support(k, m)  pose of the object m holds, in the support frame       (Gaussian, 9-d)
  normal_in_tcp(k, m)    live contact normal in the TCP frame                   (unit vector; loss 1 - cos)
  contact_phase(k, m)    free/approach/make/maintain/slide/pivot/release         (classification)
  contact_persist(k, c)  pair c in contact continuously from packet start to knot k (BCE)
Gaussian NLL is BOUNDED (D-085): log-variance clamped to [lv_min, 6] with lv_min = -4 by default (the recipe that fixed
the gradient starvation of system 0). `per_loss_grad_norms` gives the per-loss gradient norms to log (D-085 lesson).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.nets.attention import MHA
from rrp.policies.nets.flow import MLP

ANCHOR_PROBE_VERSION = "probe-anchor-v1"
POSE_QUERIES = ("tcp_in_own", "tcp_in_support", "held_in_tcp", "held_in_support")
ANCHOR_QUERIES = POSE_QUERIES + ("normal_in_tcp", "contact_phase", "contact_persist")
N_PHASES = 7
LV_MIN_BOUNDED = -4.0


class AnchorProbe(nn.Module):
    def __init__(self, dz: int, knots: int, width: int = 128, heads: int = 4, max_assemblies: int = 2,
                 max_pairs: int = 16, metadata_only: bool = False, seed: int = 4321):
        super().__init__()
        self.metadata_only = metadata_only
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("asm_code", F.normalize(torch.randn(max_assemblies, 16, generator=g), dim=-1))
        self.register_buffer("knot_code", F.normalize(torch.randn(knots, 16, generator=g), dim=-1))
        self.register_buffer("pair_code", F.normalize(torch.randn(max_pairs, 16, generator=g), dim=-1))
        D = width
        self.z_in = nn.Linear(dz, D)
        self.knot = nn.Embedding(knots, D)
        self.asm_in = nn.Linear(16, D)
        self.kq_in = nn.Linear(16, D)
        self.pair_in = nn.Linear(16, D)
        self.qtype = nn.Embedding(len(ANCHOR_QUERIES), D)
        self.const = nn.Parameter(torch.zeros(1, 1, D))
        self.att, self.att2 = MHA(D, heads), MHA(D, heads)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mlp = MLP(D, D, 2 * D)
        self.heads = nn.ModuleDict({q: nn.Linear(D, 18) for q in POSE_QUERIES})
        self.heads["normal_in_tcp"] = nn.Linear(D, 3)
        self.heads["contact_phase"] = nn.Linear(D, N_PHASES)
        self.heads["contact_persist"] = nn.Linear(D, 1)

    def _tokens(self, z, zmask):
        B, K, M, _ = z.shape
        base = self.knot.weight[None, :, None] + self.asm_in(self.asm_code[:M])[None, None]
        if self.metadata_only:
            t = (self.const[:, :, None] + base).expand(B, K, M, -1).reshape(B, K * M, -1)
        else:
            t = (self.z_in(z) + base).reshape(B, K * M, -1)
        km = zmask[:, None, :].expand(B, K, M).reshape(B, K * M)
        return t, km

    def _read(self, q, t, km):
        r = q + self.att(self.n1(q), kv=t, key_mask=km)
        r = r + self.att2(self.n2(r), kv=t, key_mask=km)
        return r + self.mlp(r)

    def forward(self, z: torch.Tensor, zmask: torch.Tensor, n_pairs: int = 0) -> dict:
        B, K, M, _ = z.shape
        t, km = self._tokens(z, zmask)
        qi = {q: i for i, q in enumerate(ANCHOR_QUERIES)}
        kc = self.kq_in(self.knot_code[:K])                                   # [K,D]
        ac = self.asm_in(self.asm_code[:M])                                   # [M,D]
        km_q = (kc[:, None] + ac[None]).reshape(K * M, -1)                    # [K*M,D]
        out = {}
        for q in POSE_QUERIES + ("normal_in_tcp", "contact_phase"):
            r = self._read((self.qtype.weight[qi[q]] + km_q)[None].expand(B, -1, -1), t, km)
            out[q] = self.heads[q](r).reshape(B, K, M, -1)
        out["normal_in_tcp"] = F.normalize(out["normal_in_tcp"], dim=-1)
        if n_pairs:
            pc = self.pair_in(self.pair_code[:n_pairs])                        # [C,D]
            kp = (kc[:, None] + pc[None]).reshape(K * n_pairs, -1)
            r = self._read((self.qtype.weight[qi["contact_persist"]] + kp)[None].expand(B, -1, -1), t, km)
            out["contact_persist"] = self.heads["contact_persist"](r).reshape(B, K, n_pairs)
        return out


# ----------------------------------------------------------------------------------------------- losses
def bounded_gaussian_nll(pred, target, mask, lv_min: float = LV_MIN_BOUNDED, lv_max: float = 6.0):
    """pred [..., 2d] = (mean, log-variance); target [..., d]; mask [...]. Mean NLL over masked items (summed over d).
    The log-variance floor bounds the loss below (D-085): NLL >= 0.5 d (lv_min + log 2 pi)."""
    d = target.shape[-1]
    mu, lv = pred[..., :d], pred[..., d:2 * d].clamp(lv_min, lv_max)
    nll = 0.5 * (((target - mu) ** 2) / lv.exp() + lv + math.log(2 * math.pi)).sum(-1)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def nll_lower_bound(d: int, lv_min: float = LV_MIN_BOUNDED) -> float:
    return 0.5 * d * (lv_min + math.log(2 * math.pi))


def anchor_probe_loss(out: dict, lab: dict, lv_min: float = LV_MIN_BOUNDED, weights: dict | None = None):
    """lab keys (batched anchor_relative_targets): <pose>[B,K,M,9] + <pose>_mask[B,K,M]; normal_in_tcp[B,K,M,3] +
    mask; phase[B,K,M]; contact_persist[B,K,C] + pair_mask[B,C]; slot_mask[B,M]. Returns (total, per-loss tensors)."""
    w = weights or {}
    sm = lab["slot_mask"][:, None, :]
    L = {}
    for q in POSE_QUERIES:
        L[q] = bounded_gaussian_nll(out[q], lab[q], lab[q + "_mask"] & sm, lv_min)
    nm = (lab["normal_in_tcp_mask"] & sm).float()
    cos = (out["normal_in_tcp"] * lab["normal_in_tcp"]).sum(-1)
    L["normal_in_tcp"] = ((1 - cos) * nm).sum() / nm.sum().clamp(min=1)
    ph = F.cross_entropy(out["contact_phase"].reshape(-1, N_PHASES), lab["phase"].reshape(-1).long(), reduction="none")
    pm = sm.expand_as(lab["phase"]).reshape(-1).float()
    L["contact_phase"] = (ph * pm).sum() / pm.sum().clamp(min=1)
    if "contact_persist" in out and "contact_persist" in lab:
        pmask = lab["pair_mask"][:, None, :].expand_as(lab["contact_persist"]).float()
        bce = F.binary_cross_entropy_with_logits(out["contact_persist"], lab["contact_persist"].float(), reduction="none")
        L["contact_persist"] = (bce * pmask).sum() / pmask.sum().clamp(min=1)
    total = sum(w.get(k, 1.0) * v for k, v in L.items())
    return total, L


def rot6d_to_mat_t(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., :3], x[..., 3:6]
    c0 = F.normalize(a, dim=-1)
    c1 = F.normalize(b - (c0 * b).sum(-1, keepdim=True) * c0, dim=-1)
    return torch.stack([c0, c1, torch.cross(c0, c1, dim=-1)], -1)


@torch.no_grad()
def anchor_probe_metrics(out: dict, lab: dict) -> dict:
    """(sum, count) pairs: position error (m), geodesic rotation error (deg) per pose query; normal angle (deg); phase
    and persistence accuracy."""
    sm = lab["slot_mask"][:, None, :]
    res = {}
    for q in POSE_QUERIES:
        m = lab[q + "_mask"] & sm
        n = int(m.sum())
        pe = (out[q][..., :3] - lab[q][..., :3]).norm(dim=-1) / 10.0
        Rp, Rt = rot6d_to_mat_t(out[q][..., 3:9]), rot6d_to_mat_t(lab[q][..., 3:9])
        tr = (Rp * Rt).sum((-1, -2))
        ang = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
        res[f"{q}_pos_err_m"] = (float((pe * m).sum()), n)
        res[f"{q}_rot_err_deg"] = (float((ang * m).sum()), n)
    nm = lab["normal_in_tcp_mask"] & sm
    a = torch.rad2deg(torch.arccos((out["normal_in_tcp"] * lab["normal_in_tcp"]).sum(-1).clamp(-1, 1)))
    res["normal_in_tcp_err_deg"] = (float((a * nm).sum()), int(nm.sum()))
    pm = sm.expand_as(lab["phase"])
    res["contact_phase_acc"] = (int(((out["contact_phase"].argmax(-1) == lab["phase"]) & pm).sum()), int(pm.sum()))
    if "contact_persist" in out and "contact_persist" in lab:
        cm = lab["pair_mask"][:, None, :].expand_as(lab["contact_persist"])
        ok = ((out["contact_persist"] > 0) == lab["contact_persist"].bool()) & cm
        res["contact_persist_acc"] = (int(ok.sum()), int(cm.sum()))
    return res


def per_loss_grad_norms(losses: dict, params) -> dict:
    """L2 norm of d(loss_i)/d(params) for every named loss (for logging next to the clipped total; D-085: a loss whose
    gradient dominates the shared clip starves the others). Uses autograd.grad with retain_graph; call every N steps."""
    params = [p for p in params if p.requires_grad]
    out = {}
    for k, L in losses.items():
        if not torch.is_tensor(L) or not L.requires_grad:
            out[k] = 0.0
            continue
        gs = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        out[k] = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in gs if g is not None) + 0.0))
    return out


def batch_targets(targets: list[dict], slot_mask, n_pairs: int | None = None) -> dict:
    """Stack per-sample anchor_relative_targets dicts (numpy) into the torch label dict anchor_probe_loss expects."""
    import numpy as np
    keys = [q for q in POSE_QUERIES] + [q + "_mask" for q in POSE_QUERIES] + ["normal_in_tcp", "normal_in_tcp_mask", "phase"]
    lab = {k: torch.from_numpy(np.stack([t[k] for t in targets])) for k in keys}
    lab["slot_mask"] = torch.as_tensor(np.asarray(slot_mask), dtype=torch.bool)
    C = n_pairs if n_pairs is not None else targets[0]["contact_persist"].shape[-1]
    if C:
        lab["contact_persist"] = torch.from_numpy(np.stack([t["contact_persist"][:, :C] for t in targets]))
        lab["pair_mask"] = torch.ones(len(targets), C, dtype=torch.bool)
    return lab
