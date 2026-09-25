"""Packet-only semantic probes: probe(received z, query_type, opaque handles) -> answer (R38, sections 4-5).

The probe sees ONLY the transmitted latent tensor, the query type and opaque handle codes (entity slot index,
assembly index in the packet). Handle codes are FIXED random vectors (addresses, not features): no object feature,
descriptor, label or context vector enters a query. `metadata_only=True` builds the control probe that receives no
latent at all (answers achievable from query/handle alone); shuffled-latent controls are applied at evaluation.

Query types (operational definitions, labels from the privileged bus, used only as supervision):
  visible(e)         e is inside the front camera frustum and not occluded (ray test)
  looking_at(e)      angle between the front camera optical axis and e (deg/30, regression)
  focused_on(e)      e is bound to a patient/target/destination role of an ACTIVE event (public runtime)
  held_by(e, m)      e touched by >= 2 bodies of manipulator m's hand assembly
  acting_on(e, m)    any contact between e and manipulator m's hand assembly
  rel_pos(e, m)      e position minus m's TCP, base frame (Gaussian: mean + log-variance)
  observed_effect(e) e REALIZED displacement over the packet horizon (Gaussian; label future_disp). Formerly named
                     `desired_delta`; renamed because it is observed motion, not intended task change. The output
                     dict keeps `desired_delta` as a deprecated alias and old checkpoints load (key remap).
  goal_effect(e)     (optional, probe cfg goal_effect=True) task-goal displacement from the task spec: for the object
                     bound as patient of a not-yet-succeeded event with a destination, destination position minus
                     object position (public estimates); zero for every other entity (label from binding_aug).
  subtask(m)         operator of m's active event (classification)
These are not an implication chain (an occluded object can be focused on or held).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MHA
from .flow import MLP

ENTITY_QUERIES = ("visible", "looking_at", "focused_on")
ENTITY_MANIP_QUERIES = ("held_by", "acting_on", "rel_pos")
ENTITY_EFFECT_QUERIES = ("observed_effect",)
MANIP_QUERIES = ("subtask",)
ALL_QUERIES = ENTITY_QUERIES + ENTITY_MANIP_QUERIES + ENTITY_EFFECT_QUERIES + MANIP_QUERIES


class PacketProbe(nn.Module):
    def __init__(self, dz: int, knots: int, width: int = 128, heads: int = 4, max_entities: int = 8,
                 max_assemblies: int = 2, n_operators: int = 12, metadata_only: bool = False, seed: int = 1234,
                 goal_effect: bool = False):
        super().__init__()
        self.metadata_only = metadata_only
        self.goal_effect = goal_effect
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("ent_code", F.normalize(torch.randn(max_entities, 16, generator=g), dim=-1))
        self.register_buffer("asm_code", F.normalize(torch.randn(max_assemblies, 16, generator=g), dim=-1))
        D = width
        self.z_in = nn.Linear(dz, D)
        self.knot = nn.Embedding(knots, D)
        self.asm_in = nn.Linear(16, D)
        self.ent_in = nn.Linear(16, D)
        self.qtype = nn.Embedding(len(ALL_QUERIES) + int(goal_effect), D)
        self.const = nn.Parameter(torch.zeros(1, 1, D))
        self.att = MHA(D, heads)
        self.att2 = MHA(D, heads)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.heads = nn.ModuleDict(dict(visible=nn.Linear(D, 1), looking_at=nn.Linear(D, 1), focused_on=nn.Linear(D, 1),
                                        held_by=nn.Linear(D, 1), acting_on=nn.Linear(D, 1), rel_pos=nn.Linear(D, 6),
                                        observed_effect=nn.Linear(D, 6), subtask=nn.Linear(D, n_operators)))
        if goal_effect:
            self.heads["goal_effect"] = nn.Linear(D, 6)
        self.mlp = MLP(D, D, 2 * D)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for k in [k for k in state_dict if k.startswith(prefix + "heads.desired_delta.")]:   # pre-rename checkpoints
            state_dict[k.replace("heads.desired_delta.", "heads.observed_effect.")] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _tokens(self, z, zmask):
        B, K, M, _ = z.shape
        if self.metadata_only:
            t = self.const.expand(B, K * M, -1) + (self.knot.weight[None, :, None] +
                                                    self.asm_in(self.asm_code[:M])[None, None]).reshape(1, K * M, -1)
        else:
            t = self.z_in(z) + self.knot.weight[None, :, None] + self.asm_in(self.asm_code[:M])[None, None]
            t = t.reshape(B, K * M, -1)
        km = zmask[:, None, :].expand(B, K, M).reshape(B, K * M)
        return t, km

    def _read(self, q, t, km):
        r = q + self.att(self.n1(q), kv=t, key_mask=km)
        r = r + self.att2(self.n2(r), kv=t, key_mask=km)
        return r + self.mlp(r)

    def forward(self, z: torch.Tensor, zmask: torch.Tensor, n_entities: int) -> dict:
        """z [B,K,M,dz] (the received packet), zmask [B,M]. Returns predictions for all entity/assembly handles."""
        B, K, M, _ = z.shape
        t, km = self._tokens(z, zmask)
        S = n_entities
        e = self.ent_in(self.ent_code[:S])                  # [S,D]
        a = self.asm_in(self.asm_code[:M])                  # [M,D]
        qi = {q: i for i, q in enumerate(ALL_QUERIES)}
        out = {}
        for q in ENTITY_QUERIES + ENTITY_EFFECT_QUERIES:
            qq = (self.qtype.weight[qi[q]] + e)[None].expand(B, S, -1)
            r = self._read(qq, t, km)
            out[q] = self.heads[q](r)                       # [B,S,*]
        for q in ENTITY_MANIP_QUERIES:
            qq = (self.qtype.weight[qi[q]] + e[:, None] + a[None]).reshape(1, S * M, -1).expand(B, -1, -1)
            r = self._read(qq, t, km)
            out[q] = self.heads[q](r).reshape(B, S, M, -1)
        qq = (self.qtype.weight[qi["subtask"]] + a)[None].expand(B, M, -1)
        out["subtask"] = self.heads["subtask"](self._read(qq, t, km))   # [B,M,ops]
        if self.goal_effect:
            qq = (self.qtype.weight[len(ALL_QUERIES)] + e)[None].expand(B, S, -1)
            out["goal_effect"] = self.heads["goal_effect"](self._read(qq, t, km))
        out["desired_delta"] = out["observed_effect"]        # deprecated alias
        return out


def _goal_terms(out, lab, smask):
    """Goal-effect loss/metric only when both the probe head and the label exist."""
    return "goal_effect" in out and "goal_effect" in lab


def gaussian_nll(pred6, target, mask):
    mu, logvar = pred6[..., :3], pred6[..., 3:].clamp(-8, 6)
    nll = 0.5 * (((target - mu) ** 2) / logvar.exp() + logvar + math.log(2 * math.pi)).sum(-1)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def probe_loss(out: dict, lab: dict, smask: torch.Tensor, m0: int = 0) -> tuple[torch.Tensor, dict]:
    """lab: held/contact/visible/focus [B,S] (manipulator 0 for held/contact/rel), rel_tcp/future_disp [B,S,3],
    gaze [B,S], subtask [B]. Positions scaled to decimeters for conditioning.
    Multi-assembly labels (held_m/contact_m [B,S,M], rel_tcp_m [B,S,M,3], subtask_m [B,M], packet slot order)
    switch the manipulator-indexed queries to ALL packet slots."""
    if "held_m" in lab:
        return probe_loss_multi(out, lab, smask)
    m = smask.float()
    den = m.sum().clamp(min=1)
    bce = lambda logit, y: (F.binary_cross_entropy_with_logits(logit.squeeze(-1), y.float(), reduction="none")
                            * m).sum() / den
    L = dict(
        visible=bce(out["visible"], lab["visible"]),
        focused_on=bce(out["focused_on"], lab["focus"]),
        held_by=bce(out["held_by"][:, :, m0], lab["held"]),
        acting_on=bce(out["acting_on"][:, :, m0], lab["contact"]),
        looking_at=((out["looking_at"].squeeze(-1) - lab["gaze"] / 30).pow(2) * m).sum() / den,
        rel_pos=gaussian_nll(out["rel_pos"][:, :, m0], lab["rel_tcp"] * 10, smask),
        observed_effect=gaussian_nll(out["observed_effect"], lab["future_disp"] * 10, smask),
        subtask=F.cross_entropy(out["subtask"][:, m0], lab["subtask"].long()),
    )
    if _goal_terms(out, lab, smask):
        L["goal_effect"] = gaussian_nll(out["goal_effect"], lab["goal_effect"] * 10, smask)
    total = sum(L.values())
    return total, {f"probe_{k}": float(v.detach()) for k, v in L.items()}


@torch.no_grad()
def probe_metrics(out: dict, lab: dict, smask: torch.Tensor, m0: int = 0) -> dict:
    """Accuracy / error metrics (raw sums for aggregation)."""
    if "held_m" in lab:
        return probe_metrics_multi(out, lab, smask)
    m = smask.bool()
    res = {}
    for q, key in (("visible", "visible"), ("focused_on", "focus"), ("held_by", "held"), ("acting_on", "contact")):
        logit = out[q][..., 0] if q in ENTITY_QUERIES else out[q][:, :, m0, 0]
        pred = logit > 0
        y = lab[key].bool()
        res[q] = (int(((pred == y) & m).sum()), int(m.sum()))
        pos = y & m
        res[q + "_pos"] = (int(((pred == y) & pos).sum()), int(pos.sum()))   # balanced view on rare positives
    err = (out["rel_pos"][:, :, m0, :3] / 10 - lab["rel_tcp"]).norm(dim=-1)
    res["rel_pos_err_m"] = (float((err * m).sum()), int(m.sum()))
    derr = (out["observed_effect"][..., :3] / 10 - lab["future_disp"]).norm(dim=-1)
    res["observed_effect_err_m"] = (float((derr * m).sum()), int(m.sum()))
    res["desired_delta_err_m"] = res["observed_effect_err_m"]          # deprecated alias (same quantity)
    res["subtask"] = (int((out["subtask"][:, m0].argmax(-1) == lab["subtask"].long()).sum()), int(len(lab["subtask"])))
    if _goal_terms(out, lab, smask):
        res.update(goal_metrics(out, lab, m))
    return res


@torch.no_grad()
def goal_metrics(out, lab, m) -> dict:
    """goal_effect error on all slots and on goal-bearing slots (bound patient), plus a zero-prediction baseline."""
    g = lab["goal_effect"]
    err = (out["goal_effect"][..., :3] / 10 - g).norm(dim=-1)
    gp = (g.norm(dim=-1) > 1e-6) & m
    return dict(goal_effect_err_m=(float((err * m).sum()), int(m.sum())),
                goal_effect_err_patient_m=(float((err * gp).sum()), int(gp.sum())),
                goal_effect_zero_baseline_patient_m=(float((g.norm(dim=-1) * gp).sum()), int(gp.sum())))


def probe_loss_multi(out: dict, lab: dict, smask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Every packet slot m answers its own held_by/acting_on/rel_pos/subtask queries (role-addressed)."""
    m = smask.float()
    den = m.sum().clamp(min=1)
    M = lab["held_m"].shape[-1]
    mm = m[:, :, None].expand(-1, -1, M)
    denm = mm.sum().clamp(min=1)
    bce = lambda logit, y, w, d: (F.binary_cross_entropy_with_logits(logit, y.float(), reduction="none") * w).sum() / d
    L = dict(
        visible=bce(out["visible"].squeeze(-1), lab["visible"], m, den),
        focused_on=bce(out["focused_on"].squeeze(-1), lab["focus"], m, den),
        held_by=bce(out["held_by"][:, :, :M, 0], lab["held_m"], mm, denm),
        acting_on=bce(out["acting_on"][:, :, :M, 0], lab["contact_m"], mm, denm),
        looking_at=((out["looking_at"].squeeze(-1) - lab["gaze"] / 30).pow(2) * m).sum() / den,
        rel_pos=gaussian_nll(out["rel_pos"][:, :, :M], lab["rel_tcp_m"] * 10, mm.bool()),
        observed_effect=gaussian_nll(out["observed_effect"], lab["future_disp"] * 10, smask),
        subtask=F.cross_entropy(out["subtask"][:, :M].reshape(-1, out["subtask"].shape[-1]),
                                lab["subtask_m"].reshape(-1).long()),
    )
    total = sum(L.values())
    return total, {f"probe_{k}": float(v.detach()) for k, v in L.items()}


@torch.no_grad()
def probe_metrics_multi(out: dict, lab: dict, smask: torch.Tensor) -> dict:
    """Per packet slot m: held_by / acting_on accuracy (all and on positives), rel_pos error, subtask accuracy;
    plus the slot-independent queries. Keys are suffixed @m (m = packet slot, i.e. role order)."""
    m = smask.bool()
    res = {}
    for q, key in (("visible", "visible"), ("focused_on", "focus")):
        pred = out[q][..., 0] > 0
        y = lab[key].bool()
        res[q] = (int(((pred == y) & m).sum()), int(m.sum()))
        pos = y & m
        res[q + "_pos"] = (int(((pred == y) & pos).sum()), int(pos.sum()))
    derr = (out["observed_effect"][..., :3] / 10 - lab["future_disp"]).norm(dim=-1)
    res["observed_effect_err_m"] = (float((derr * m).sum()), int(m.sum()))
    res["desired_delta_err_m"] = res["observed_effect_err_m"]          # deprecated alias
    M = lab["held_m"].shape[-1]
    for a in range(M):
        for q, key in (("held_by", "held_m"), ("acting_on", "contact_m")):
            pred = out[q][:, :, a, 0] > 0
            y = lab[key][:, :, a].bool()
            res[f"{q}@{a}"] = (int(((pred == y) & m).sum()), int(m.sum()))
            pos = y & m
            res[f"{q}_pos@{a}"] = (int(((pred == y) & pos).sum()), int(pos.sum()))
            neg = ~y & m
            res[f"{q}_neg@{a}"] = (int(((pred == y) & neg).sum()), int(neg.sum()))
        err = (out["rel_pos"][:, :, a, :3] / 10 - lab["rel_tcp_m"][:, :, a]).norm(dim=-1)
        res[f"rel_pos_err_m@{a}"] = (float((err * m).sum()), int(m.sum()))
        res[f"subtask@{a}"] = (int((out["subtask"][:, a].argmax(-1) == lab["subtask_m"][:, a].long()).sum()),
                               int(lab["subtask_m"].shape[0]))
    # pooled over slots (comparable with single-assembly keys)
    for k in ("held_by", "held_by_pos", "acting_on", "acting_on_pos", "rel_pos_err_m", "subtask"):
        xs = [res[f"{k}@{a}"] for a in range(M)]
        res[k] = (sum(x for x, _ in xs), sum(n for _, n in xs))
    return res
