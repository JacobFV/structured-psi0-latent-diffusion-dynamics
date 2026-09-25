"""Counterfactual task-binding edits on a collated Batch (D-032 follow-up, track `binding`).

Why: in every training episode the task patient is canonical scene slot 0 (same descriptor hash) and slot 1 is the
destination, so the `focused_on` label is (almost) a constant of the slot index (metadata-only probe: 0.93) and the
demonstrated trajectory also points at slot 0. Nothing in the data forces an encoder to read the supplied binding.

`rebind(batch, src, dst)` swaps ALL structural edges (relations in both orientations, incidence pointers) of scene
slots `src` and `dst`, i.e. the task graph now binds the roles of `src` to `dst` (and vice versa). Scene tokens,
morphology, proprio and the demonstrated trajectory are untouched. Scene slots only have edges to task-bank tokens
(role_points_to, patient/target/destination_of, pred_arg), so a symmetric permutation of the slot rows/columns of
ctx_rel is exactly a task-binding swap. `focus` (defined from the public binding of ACTIVE events) follows the
swap; physical labels (held, contact, rel_pos, future displacement, visibility, gaze) describe the body/world and
are unchanged. Used identically for both latent variants (only semantic_weight differs) and by the post-hoc test.
"""
from __future__ import annotations

import torch

from rrp.data.features import HASH_DIM
from .batch import Batch

FOCUS_RELS = (4, 5, 6)          # patient_of, target_of, destination_of


def slot_has_edges(batch: Batch) -> torch.Tensor:
    """[B,S] True where the scene slot has any structural edge (i.e. is bound in the task graph)."""
    o, S = batch.bank_offset["scene"], batch.bank_tokens["scene"].shape[1]
    r = batch.ctx_rel
    return r[:, o:o + S].flatten(2).any(-1) | r[:, :, o:o + S].transpose(1, 2).flatten(2).any(-1)


def focus_from_batch(batch: Batch) -> torch.Tensor:
    """Recompute the public focus label from the (possibly edited) relations: slots pointed to by
    patient/target/destination relations from ACTIVE event tokens (same rule as learning/packed._focus)."""
    ot, T = batch.bank_offset["task"], batch.bank_tokens["task"].shape[1]
    os_, S = batch.bank_offset["scene"], batch.bank_tokens["scene"].shape[1]
    tt, tk, tm = batch.bank_tokens["task"], batch.bank_kind["task"], batch.bank_mask["task"]
    active = (tk == 0) & tm & (tt[..., HASH_DIM + 2] > 0.5)                       # [B,T]
    r = batch.ctx_rel[:, ot:ot + T, os_:os_ + S][..., list(FOCUS_RELS)].any(-1)   # [B,T,S]
    return (r & active[..., None]).any(1)


STATUS_SUCCEEDED = 3            # features.STATUS index


def goal_effect_from_batch(batch: Batch) -> torch.Tensor:
    """Task-goal displacement label [B,S,3] (m, base frame), from the task spec and public estimates only:
    for every event that is not yet succeeded and has both a patient and a destination bound to scene slots,
    goal[patient] = pos[destination] - pos[patient]; zero for every other slot. Recomputed after `rebind`, so it
    follows the supplied binding (unlike future_disp, which is the realized motion)."""
    ot, T = batch.bank_offset["task"], batch.bank_tokens["task"].shape[1]
    os_, S = batch.bank_offset["scene"], batch.bank_tokens["scene"].shape[1]
    tt, tk, tm = batch.bank_tokens["task"], batch.bank_kind["task"], batch.bank_mask["task"]
    open_ev = (tk == 0) & tm & (tt[..., HASH_DIM + STATUS_SUCCEEDED] < 0.5)          # [B,T]
    r = batch.ctx_rel[:, ot:ot + T, os_:os_ + S]                                       # [B,T,S,R]
    pat = r[..., 4] & open_ev[..., None]
    dst = r[..., 6] & open_ev[..., None]
    pos = batch.bank_tokens["scene"][..., :3]                                          # [B,S,3] public estimate
    has_dst = dst.any(-1)                                                              # [B,T]
    dpos = (dst.float() @ pos) / dst.float().sum(-1, keepdim=True).clamp(min=1)         # [B,T,3]
    w = (pat & has_dst[..., None]).float()                                             # [B,T,S]
    tgt = torch.einsum("bts,btc->bsc", w, dpos) / w.sum(1)[..., None].clamp(min=1)     # [B,S,3]
    return torch.where(w.sum(1)[..., None] > 0, tgt - pos, torch.zeros_like(pos))


def choose_swap(batch: Batch, slot_ok: torch.Tensor, gen: torch.Generator | None = None):
    """Per sample: src = random bound slot, dst = random valid UNBOUND slot (else another bound slot).
    Returns src [B], dst [B], ok [B] (False when fewer than two valid slots or no bound slot)."""
    bound = slot_has_edges(batch) & slot_ok
    B, S = bound.shape
    u = torch.rand(B, S, generator=gen, device="cpu").to(bound.device)
    src = torch.where(bound, u, torch.full_like(u, -1)).argmax(1)
    others = slot_ok & (torch.arange(S, device=bound.device)[None] != src[:, None])
    u2 = torch.rand(B, S, generator=gen, device="cpu").to(bound.device)
    score = torch.where(others, u2 + 2.0 * (~bound).float(), torch.full_like(u2, -1))   # prefer unbound dst
    dst = score.argmax(1)
    ok = bound.any(1) & others.any(1)
    return src, dst, ok


def slot_perm(batch: Batch, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """[B,C] context permutation that swaps scene slots src<->dst (identity elsewhere)."""
    B, C = batch.ctx_mask.shape
    o = batch.bank_offset["scene"]
    perm = torch.arange(C, device=src.device).repeat(B, 1)
    b = torch.arange(B, device=src.device)
    perm[b, o + src] = o + dst
    perm[b, o + dst] = o + src
    return perm


def rebind(batch: Batch, src: torch.Tensor, dst: torch.Tensor) -> Batch:
    """New Batch whose task graph binds slot src's roles to dst and vice versa (tokens unchanged)."""
    perm = slot_perm(batch, src, dst)
    B, C = perm.shape
    rel = batch.ctx_rel
    ix_q = perm[:, :, None, None].expand(-1, -1, C, rel.shape[-1])
    rel = torch.gather(rel, 1, ix_q)                                   # rows
    ix_k = perm[:, None, :, None].expand(-1, C, -1, rel.shape[-1])
    rel = torch.gather(rel, 2, ix_k)                                   # columns
    ar = batch.act_rel
    act = torch.gather(ar, 2, perm[:, None, :, None].expand(-1, ar.shape[1], -1, ar.shape[-1]))
    p = batch.pointers
    mapped = torch.gather(perm, 1, p.clamp(min=0).flatten(1)).view_as(p)
    ptr = torch.where(p >= 0, mapped, p)
    d = dict(batch.__dict__)
    d.update(ctx_rel=rel, act_rel=act, pointers=ptr)
    return Batch(**d)


def swap_slot_labels(lab: dict, src: torch.Tensor, dst: torch.Tensor, keys=("focus",)) -> dict:
    """Binding-defined labels follow the swap; everything else (physical truth) is unchanged."""
    out = dict(lab)
    b = torch.arange(src.shape[0], device=src.device)
    for k in keys:
        x = lab[k].clone()
        x[b, src], x[b, dst] = lab[k][b, dst], lab[k][b, src]
        out[k] = x
    return out


def index_batch(batch: Batch, idx: torch.Tensor) -> Batch:
    def f(x):
        if isinstance(x, torch.Tensor):
            return x[idx]
        if isinstance(x, dict):
            return {k: f(v) for k, v in x.items()} if all(isinstance(v, torch.Tensor) for v in x.values()) else x
        return x
    return Batch(**{k: (v if k == "bank_offset" else f(v)) for k, v in batch.__dict__.items()})


def cat_batch(b1: Batch, b2: Batch) -> Batch:
    def f(x, y):
        if isinstance(x, torch.Tensor):
            return torch.cat([x, y], 0)
        if isinstance(x, dict):
            return {k: f(x[k], y[k]) for k in x} if all(isinstance(v, torch.Tensor) for v in x.values()) else x
        return x
    return Batch(**{k: (v if k == "bank_offset" else f(v, getattr(b2, k))) for k, v in b1.__dict__.items()})


def augment(batch: Batch, a, v, lab: dict, frac: float, gen: torch.Generator | None = None):
    """Append counterfactual-binding copies of a random `frac` of the (swappable) samples.
    Returns (batch', a', v', lab', n_factual, cf_info) with the factual rows first."""
    B = batch.B
    slot_ok = batch.bank_mask["scene"] & lab["slot_valid"].bool()
    src, dst, ok = choose_swap(batch, slot_ok, gen)
    k = int(round(frac * B))
    cand = torch.nonzero(ok).flatten()
    if k == 0 or len(cand) == 0:
        return batch, a, v, lab, B, None
    pick = cand[torch.randperm(len(cand), generator=gen)[:k].to(cand.device)]
    bc = rebind(index_batch(batch, pick), src[pick], dst[pick])
    lc = swap_slot_labels({kk: x[pick] for kk, x in lab.items()}, src[pick], dst[pick])
    if "goal_effect" in lab:                   # binding-defined: recompute from the rebound graph
        lc["goal_effect"] = goal_effect_from_batch(bc)
    out_b = cat_batch(batch, bc)
    out_l = {kk: torch.cat([lab[kk], lc[kk]], 0) for kk in lab}
    return out_b, torch.cat([a, a[pick]], 0), torch.cat([v, v[pick]], 0), out_l, B, \
        dict(pick=pick, src=src[pick], dst=dst[pick])
