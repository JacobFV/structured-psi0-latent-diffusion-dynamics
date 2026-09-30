"""Collate PolicyInputs into padded tensors.

Context tokens are the concatenation of the four banks (morph, scene, task, interact) in a
fixed order, each padded to the batch max. Relations become multi-hot tensors:
  ctx_rel  [B, C, C, R]  context->context (query row, key column)
  act_rel  [B, N, C, R]  action node (query) -> context (key)
  node_rel [B, N, N, R]  action node -> action node (kinematic parent)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rrp.policies.features.featurizer import (BANKS, N_REL, HASH_DIM, REL, NODE_ANCHOR_SLICE, ASM_POS_SLICE,
                                              ASM_ZCOL_SLICE, ASM_XCOL_SLICE, SCENE_POS_SLICE, SCENE_STD_SLICE,
                                              SCENE_KNOWN_COL)
from rrp.policies.relations.base import EdgeSet, TokenSet, TOKEN_KINDS
from rrp.policies.relations.catalog import SUPPORT_REL_VOCAB

MORPH_DIM = 44
BANK_DIMS = {"morph": MORPH_DIM, "scene": 27, "task": 59, "interact": 32}
NODE_DIM = 44
MORPH_BANK_ID = BANKS.index("morph")
NODE_IN_ASSEMBLY = REL["node_in_assembly"]

# bank-local token subtype id (PolicyInput.token_kind, per the comments in `features.featurizer.__call__`) -> the
# shared TOKEN_KINDS vocabulary (docs/relations.md section 2). R12: TokenSet.kind of the `ctx` bank.
_KIND_ID = {n: i for i, n in enumerate(TOKEN_KINDS)}
_BANK_LOCAL_KIND = {
    "morph": {0: "morph_node", 1: "passive_joint", 2: "assembly"},
    "scene": {0: "entity", 1: "pad"},
    "task": {0: "task_event", 1: "task_role", 2: "predicate", 3: "pad"},
    "interact": {0: "receipt", 1: "sensor", 2: "pad"},
}


def _global_kind(bank: str, local: torch.Tensor) -> torch.Tensor:
    table = torch.full((8,), _KIND_ID["pad"], dtype=torch.long)
    for k, name in _BANK_LOCAL_KIND[bank].items():
        table[k] = _KIND_ID[name]
    return table[local.long()]


@dataclass
class Batch:
    bank_tokens: dict          # bank -> [B, T, F]
    bank_mask: dict            # bank -> [B, T] bool (True = valid)
    bank_kind: dict            # bank -> [B, T] long
    bank_text: dict            # bank -> [B, T, HASH_DIM] (serialized pointer text, unstructured mode)
    bank_offset: dict          # bank -> int offset into concatenated context
    ctx_mask: torch.Tensor     # [B, C]
    node_feats: torch.Tensor   # [B, N, F]
    node_mask: torch.Tensor    # [B, N]
    ctx_rel: torch.Tensor      # [B, C, C, R] bool
    act_rel: torch.Tensor      # [B, N, C, R] bool
    node_rel: torch.Tensor     # [B, N, N, R] bool
    pointers: torch.Tensor     # [B, P, 2] (src ctx idx, dst ctx idx), -1 padded
    extra: dict

    def to(self, device):
        def mv(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            if isinstance(x, dict):
                return {k: mv(v) for k, v in x.items()}
            return x
        return Batch(**{k: mv(v) for k, v in self.__dict__.items()})

    @property
    def B(self):
        return self.node_feats.shape[0]


def collate_inputs(inputs: list, extra_tokens: dict | None = None) -> Batch:
    B = len(inputs)
    T = {b: max(max(len(pi.tokens[b]) for pi in inputs), 1) for b in BANKS}
    offs, o = {}, 0
    for b in BANKS:
        offs[b] = o
        o += T[b]
    C = o
    toks, masks, kinds, texts = {}, {}, {}, {}
    for b in BANKS:
        F = BANK_DIMS[b]
        x = np.zeros((B, T[b], F), np.float32)
        m = np.zeros((B, T[b]), bool)
        k = np.zeros((B, T[b]), np.int64)
        t = np.zeros((B, T[b], HASH_DIM), np.float32)
        for i, pi in enumerate(inputs):
            n = len(pi.tokens[b])
            x[i, :n, :pi.tokens[b].shape[1]] = pi.tokens[b]
            m[i, :n] = True
            k[i, :n] = pi.token_kind[b]
            pt = pi.pointer_text[b]
            t[i, :len(pt)] = pt
        toks[b], masks[b], kinds[b], texts[b] = (torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(k),
                                                 torch.from_numpy(t))
    N = max(pi.act_node_feats.shape[0] for pi in inputs)
    nf = np.zeros((B, N, NODE_DIM), np.float32)
    nm = np.zeros((B, N), bool)
    ctx_rel = np.zeros((B, C, C, N_REL), bool)
    act_rel = np.zeros((B, N, C, N_REL), bool)
    node_rel = np.zeros((B, N, N, N_REL), bool)
    P = max(max(len(pi.pointers) for pi in inputs), 1)
    ptr = -np.ones((B, P, 2), np.int64)
    offs_arr = np.array([offs[b] for b in BANKS])
    for i, pi in enumerate(inputs):
        n = pi.act_node_feats.shape[0]
        nf[i, :n] = pi.act_node_feats
        nm[i, :n] = True
        R = np.asarray(pi.relations).reshape(-1, 5)
        if len(R):
            kc = offs_arr[R[:, 2]] + R[:, 3]
            a = R[:, 0] == -1
            act_rel[i, R[a, 1], kc[a], R[a, 4]] = True
            c = ~a
            qc = offs_arr[R[c, 0]] + R[c, 1]
            ctx_rel[i, qc, kc[c], R[c, 4]] = True
            # node->node kinematic relation among action-node morph tokens
            kpm = (R[:, 0] == 0) & (R[:, 2] == 0) & (R[:, 4] == 16) & (R[:, 1] < n) & (R[:, 3] < n)
            node_rel[i, R[kpm, 1], R[kpm, 3], 16] = True
            node_rel[i, R[kpm, 3], R[kpm, 1], 16] = True
        Pp = np.asarray(pi.pointers).reshape(-1, 4)
        if len(Pp):
            ptr[i, :len(Pp), 0] = offs_arr[Pp[:, 0]] + Pp[:, 1]
            ptr[i, :len(Pp), 1] = offs_arr[Pp[:, 2]] + Pp[:, 3]
    ctx_mask = torch.cat([masks[b] for b in BANKS], 1)
    return Batch(toks, masks, kinds, texts, offs, ctx_mask, torch.from_numpy(nf), torch.from_numpy(nm),
                 torch.from_numpy(ctx_rel), torch.from_numpy(act_rel), torch.from_numpy(node_rel),
                 torch.from_numpy(ptr), dict(extra_tokens or {}))


# ---------------------------------------------------------------------------------------------------------------
# R12 (D-144, docs/relations.md sections 2 & 10): `ctx` / `act` TokenSet fields, derived from the collated tensors
# (no dataset rewrite, no per-sample loops: the same code runs at collate, train and deploy time). Field dims /
# provenance are declared once in `catalog.FIELDS`; this module only supplies the VALUES. `cam_uvd` needs a live
# camera: the caller puts it (and its `.valid`) in `batch.extra["ctx_fields"]`, `attach_cam_uvd` builds it.
def _cat_banks(batch: "Batch", fn, dim: int, dtype=torch.float32) -> torch.Tensor:
    """Concatenate a per-bank [B, T, dim] value (fn(bank) or None -> zeros) over the ctx bank order."""
    outs = []
    for b in BANKS:
        B, T = batch.bank_mask[b].shape
        v = fn(b)
        outs.append(torch.zeros(B, T, dim, dtype=dtype) if v is None else v)
    return torch.cat(outs, 1)


def _last_true(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(any, index of the last True) along the final dim of a bool tensor."""
    n = m.shape[-1]
    return m.any(-1), (n - 1 - m.flip(-1).long().argmax(-1))


def ctx_geometry_fields(batch: "Batch") -> dict:
    """`pos3d` (+`.var`) and `orient` of the concatenated `ctx` bank:
      - morph action / passive-joint tokens: `pos3d` = the public-FK joint anchor (`NODE_ANCHOR_SLICE`), always
        valid; no `orient` (a joint anchor is a point, not a frame).
      - morph assembly tokens: `pos3d` = the frame origin (`ASM_POS_SLICE`), always valid; `orient` = the 3x3
        frame rotation matrix, flattened row-major (`FieldDef("orient", 9, ...)`), reconstructed from its stored
        R[:, 2] / R[:, 0] columns (`ASM_ZCOL_SLICE` / `ASM_XCOL_SLICE`) via R[:, 1] = cross(R[:, 2], R[:, 0])
        (right-handed orthonormal frame).
      - scene tokens: `pos3d` = the tracked slot position (`SCENE_POS_SLICE`), valid iff `known`
        (`SCENE_KNOWN_COL`; an unknown slot is a masked belief, not a public zero); `.var` = the tracked position
        covariance, recovered from the stored `log(std + 1e-4) / 5` (`SCENE_STD_SLICE`); no `orient` (object
        frames are unknown to the tracker, docs/relations.md section 2).
      - task / interact tokens: no geometry; `pos3d` / `orient` stay invalid. Padding is never valid.
    """
    mt, mk, mm = batch.bank_tokens["morph"], batch.bank_kind["morph"], batch.bank_mask["morph"]
    st, sk, sm = batch.bank_tokens["scene"], batch.bank_kind["scene"], batch.bank_mask["scene"]
    node, asm = (mk < 2) & mm, (mk == 2) & mm
    known = (sk == 0) & sm & (st[..., SCENE_KNOWN_COL] > 0.5)
    pos = {"morph": torch.where(asm[..., None], mt[..., ASM_POS_SLICE], mt[..., NODE_ANCHOR_SLICE]) * (node | asm)[..., None],
           "scene": st[..., SCENE_POS_SLICE] * known[..., None]}
    valid = {"morph": node | asm, "scene": known}
    zcol, xcol = mt[..., ASM_ZCOL_SLICE], mt[..., ASM_XCOL_SLICE]
    R = torch.stack([xcol, torch.cross(zcol, xcol, dim=-1), zcol], -1)          # columns x, y = z(cross)x, z
    std = (torch.exp(5.0 * st[..., SCENE_STD_SLICE]) - 1e-4).clamp(min=0.0)
    zero_b = {b: torch.zeros_like(batch.bank_mask[b]) for b in BANKS}
    return {"pos3d": _cat_banks(batch, lambda b: pos.get(b), 3),
            "pos3d.valid": torch.cat([valid.get(b, zero_b[b]) for b in BANKS], 1),
            "pos3d.var": _cat_banks(batch, lambda b: (std ** 2) * known[..., None] if b == "scene" else None, 3),
            "orient": _cat_banks(batch, lambda b: R.flatten(-2) * asm[..., None] if b == "morph" else None, 9),
            "orient.valid": torch.cat([asm if b == "morph" else zero_b[b] for b in BANKS], 1)}


def ctx_id_fields(batch: "Batch") -> dict:
    """`entity_id` / `assembly_id` of the `ctx` bank (docs/relations.md section 2, R15 `id.same_*`), read off the
    EXISTING `ctx_rel` / `pointers` tensors (no dataset rewrite):
      - `entity_id`: every ctx token defaults to its OWN flattened ctx index (self-identifying); a token that is
        the source of EXACTLY ONE incidence pointer (a task-role token referencing an entity, a receipt
        referencing an event) instead takes that pointer's DESTINATION index, so two tokens referring to the same
        public entity compare equal under the `same` op. A predicate-estimate token can point at SEVERAL argument
        entities at once (`pred_arg`, one pointer per argument); such a fan-out source is not one entity's alias,
        so it keeps its own self id (its per-argument bindings stay available as `pred_arg` edges). Valid on
        every real token.
      - `assembly_id`: every ctx token defaults to -1 (invalid, no assembly); a morph assembly token's own id is
        its flattened index; any OTHER ctx token with a `node_in_assembly` edge to one (today: per-manipulator
        sensor tokens of the `interact` bank) takes that assembly's index. `node_in_assembly` is stored BOTH
        directions, so only rows whose KEY is a morph assembly token count (the reverse rows must not overwrite an
        assembly's own id). Action-NODE `assembly_id` (the `act` set) is `act_assembly_id`: a node's edge is
        act>ctx, never ctx>ctx.
    """
    B, C = batch.ctx_mask.shape
    own = torch.arange(C).expand(B, C)
    ptr = batch.pointers
    src, dst = ptr[..., 0], ptr[..., 1]
    ok = (src >= 0) & (dst >= 0)
    cnt = torch.zeros(B, C, dtype=torch.long).scatter_add(1, src.clamp(min=0), ok.long())
    dsum = torch.zeros(B, C, dtype=torch.long).scatter_add(1, src.clamp(min=0), torch.where(ok, dst, 0))
    ent = torch.where(cnt == 1, dsum, own)
    mo = batch.bank_offset["morph"]
    is_asm = torch.zeros(B, C, dtype=torch.bool)
    is_asm[:, mo:mo + batch.bank_kind["morph"].shape[1]] = (batch.bank_kind["morph"] == 2) & batch.bank_mask["morph"]
    has, last = _last_true(batch.ctx_rel[..., NODE_IN_ASSEMBLY] & is_asm[:, None, :])
    asm = torch.where(has, last, torch.where(is_asm, own, torch.full_like(own, -1)))
    return {"entity_id": ent[..., None].float(), "entity_id.valid": batch.ctx_mask.clone(),
            "assembly_id": asm[..., None].float(), "assembly_id.valid": (asm >= 0) & batch.ctx_mask}


def act_assembly_id(batch: "Batch") -> dict:
    """`assembly_id` of the `act` token set (action nodes): the node's `node_in_assembly` act>ctx edge, giving its
    owning morph assembly token's flattened `ctx` index."""
    has, last = _last_true(batch.act_rel[..., NODE_IN_ASSEMBLY])
    asm = torch.where(has, last, torch.full_like(last, -1))
    return {"assembly_id": asm[..., None].float(), "assembly_id.valid": (asm >= 0) & batch.node_mask}


def attach_cam_uvd(pos3d: torch.Tensor, pos3d_valid: torch.Tensor, cameras: list) -> dict:
    """`cam_uvd` from an ALREADY-DERIVED `pos3d` field, projected per batch item through its own declared scene
    camera (`envs.mujoco.sensors.project_points`; `cameras[i] = (model, data, cam_name)`, one triple per item --
    collation has no live MuJoCo state of its own, so this is a separate step from `ctx_geometry_fields`; the caller
    stores the result in `batch.extra["ctx_fields"]`). Positions behind the camera or with invalid `pos3d` stay invalid."""
    from rrp.envs.mujoco.sensors import project_points
    pos = pos3d.detach().cpu().numpy() if hasattr(pos3d, "detach") else np.asarray(pos3d)
    val = pos3d_valid.detach().cpu().numpy() if hasattr(pos3d_valid, "detach") else np.asarray(pos3d_valid)
    B, C, _ = pos.shape
    out = np.zeros((B, C, 3), np.float32)
    out_valid = np.zeros((B, C), bool)
    for i, (model, data, cam) in enumerate(cameras):
        m = val[i]
        if not m.any():
            continue
        uvd = project_points(model, data, cam, pos[i, m])
        out[i, m] = uvd
        out_valid[i, m] = uvd[:, 2] > 1e-9
    return {"cam_uvd": torch.from_numpy(out), "cam_uvd.valid": torch.from_numpy(out_valid)}


_DERIVED = {"pos3d": ctx_geometry_fields, "orient": ctx_geometry_fields, "entity_id": ctx_id_fields,
            "assembly_id": ctx_id_fields}
_ARM_FAMILIES = ("arm", "dual")          # the families this collate path serves; others build their own sets


def _pad_ctx(d: dict, pad: int) -> dict:
    return {k: F.pad(v, (0, 0) * (v.dim() - 2) + (0, pad)) if pad and v.dim() >= 2 else v for k, v in d.items()}


def relation_token_sets(family: str, batch: "Batch", labels: dict | None = None, deploy: bool = False,
                        fields=None, pad_ctx: int = 0) -> dict:
    """The `ctx` / `act` `TokenSet`s of a net family (docs/relations.md section 11): `kind`, every public / estimated
    field the family declares (`FAMILIES[family].sets`; `fields=()` skips them when no factor reads one), then
    `batch.extra["ctx_fields"]` (camera-derived `cam_uvd`), and, for training, `labels` =
    {"ctx": {name: tensor, name + ".valid": mask}, "act": {...}} (names must be ones the family attaches). Labels are
    privileged: passing them with `deploy=True` raises `PrivilegedInput`. `pad_ctx` appends invalid ctx positions
    (VLM image tokens) to every ctx tensor."""
    from rrp.policies.relations.base import FAMILIES, FactorError, PrivilegedInput
    if family not in FAMILIES:
        raise FactorError(f"unknown net family {family!r}; families: {sorted(FAMILIES)}")
    if family not in _ARM_FAMILIES:
        raise FactorError(f"family {family!r} builds its own token sets; this collate path serves {_ARM_FAMILIES}")
    if labels and deploy:
        raise PrivilegedInput("relation labels passed with deploy=True (labels are supervision-only)")
    ft = FAMILIES[family]
    want = tuple(ft.sets["ctx"]) if fields is None else tuple(fields)
    cf = {}
    for n in want:
        if n in _DERIVED:
            cf.update({k: v for k, v in _DERIVED[n](batch).items() if k.split(".")[0] == n})
    for k, v in batch.extra.get("ctx_fields", {}).items():
        if k.split(".")[0] not in ft.sets["ctx"]:
            raise FactorError(f"ctx field {k!r} is not one family {family!r} fills ({list(ft.sets['ctx'])})")
        cf[k] = v
    af = act_assembly_id(batch) if "assembly_id" in ft.sets["act"] and ("assembly_id" in want or fields is None) else {}
    kind = torch.cat([_global_kind(b, batch.bank_kind[b]) for b in BANKS], 1)
    mask = batch.ctx_mask
    if pad_ctx:
        mask = F.pad(mask, (0, pad_ctx))
        kind = F.pad(kind, (0, pad_ctx), value=_KIND_ID["pad"])
    sets = {"ctx": TokenSet("ctx", mask, kind, _pad_ctx(cf, pad_ctx), deploy=deploy),
            "act": TokenSet("act", batch.node_mask, None, af, deploy=deploy)}
    for name, lab in (labels or {}).items():
        if name not in sets:
            raise FactorError(f"labels for unknown token set {name!r}")
        for k in lab:
            if k.split(".valid")[0] not in ft.labels.get(name, ()):
                raise FactorError(f"label {k!r} is not one family {family!r} attaches to {name!r} "
                                  f"({list(ft.labels.get(name, ()))})")
        sets[name].labels = _pad_ctx(dict(lab), pad_ctx) if name == "ctx" else dict(lab)
    return sets


# ---------------------------------------------------------------------------------------------------------------
# R18 (D-144, docs/relations.md sections 5.4 & 10): candidate interaction edges -- a soft PUBLIC EdgeSet of manipulator
# -> graspable and object -> support / destination pairs, over the SAME `ctx` token set `relation_token_sets` above
# builds (site "ctx>ctx"). `task.next_contact` (catalog.py) is a `bilinear` factor: it reads token HIDDENS, never
# this EdgeSet directly (docs 3.2's `bilinear` op has no `edges:*` field); the candidates this function names are
# instead the pool `harness.data.relgen.task.next_contact_sample` turns into `reveal` / `surprise` targets (R9). No
# affordance labels exist in these fixtures, so every currently-populated candidate row is a UNIFORM prior over its
# targets -- deliberately NOT gated by the `known` bit (an occluded object is still a valid interaction candidate;
# resolving which one is the epistemic point of this unit, docs 5.4 / research/relations_catalog.md J).
CAND_REL_VOCAB = ("graspable", "support", "destination")
_CAND_GRASPABLE, _CAND_SUPPORT, _CAND_DESTINATION = range(len(CAND_REL_VOCAB))


def candidate_interaction_edges(inputs: list, batch: "Batch") -> EdgeSet:
    """`EdgeSet(CAND_REL_VOCAB, [B,C,C,3])`, `prov="public"` (docs `Prov`, section 1): row `i` -> uniform prior mass
    1/|candidates| over its valid targets, 0 elsewhere (a proper per-row distribution; an all-zero row means no
    candidate this channel applies to, e.g. no sensor token or fewer than two scene entities).
      - `graspable`:   manipulator touch/grip SENSOR tokens (`interact` bank, local kind 1 -- the featurizer's
                       per-manipulator `declared_sensor_channels`, `features/featurizer.py`) -> every valid SCENE
                       entity token (`scene` bank, local kind 0).
      - `support` / `destination`: every valid scene entity token -> every OTHER valid scene entity token (an
                       object cannot be its own support / destination).
    """
    B, C = batch.B, batch.ctx_mask.shape[1]
    data = np.zeros((B, C, C, len(CAND_REL_VOCAB)), np.float32)
    scene_off, inter_off = batch.bank_offset["scene"], batch.bank_offset["interact"]
    scene_kind_all = batch.bank_kind["scene"].numpy()
    scene_mask_all = batch.bank_mask["scene"].numpy()
    inter_kind_all = batch.bank_kind["interact"].numpy()
    inter_mask_all = batch.bank_mask["interact"].numpy()
    for i in range(B):
        entity_idx = scene_off + np.where(scene_mask_all[i] & (scene_kind_all[i] == 0))[0]
        sensor_idx = inter_off + np.where(inter_mask_all[i] & (inter_kind_all[i] == 1))[0]
        if len(entity_idx) and len(sensor_idx):
            data[i][np.ix_(sensor_idx, entity_idx, [_CAND_GRASPABLE])] = 1.0 / len(entity_idx)
        if len(entity_idx) > 1:
            p = 1.0 / (len(entity_idx) - 1)
            for q in entity_idx:
                others = entity_idx[entity_idx != q]
                data[i, q, others, _CAND_SUPPORT] = p
                data[i, q, others, _CAND_DESTINATION] = p
    return EdgeSet(CAND_REL_VOCAB, torch.from_numpy(data), prov="public")


# ---------------------------------------------------------------------------------------------------------------
# rel-geo (D-144 addendum, item 2; R17 follow-up, archived research/tracks/rel-r17.md "for the lead"): `ix.force_flow`'s
# "edges:support-v1" EdgeSet (`SUPPORT_REL_VOCAB = ("support",)`, `catalog.py`) -- the DIRECT support graph
# (id-keyed, the shape `harness.data.relgen.support.support_matrix` returns), over the SAME `ctx` SCENE-bank token
# positions `relation_token_sets` above serves. This module (`policies.nets`, architecture.md layer 4) never
# imports `harness.data.relgen.support` itself (layer 5; `tests/unit/test_layering.py` forbids the upward edge) --
# `graphs` is the caller-supplied `support_matrix(view)` result per batch item (a harness-layer caller, e.g. a
# training loop or this unit's own tests, computes it and passes the plain dict in), and `views` is one privileged
# `envs.base` view (docs 5.1's read-only truth interface, layer 3, a downward/allowed import; the same
# one-handle-per-item pattern `attach_cam_uvd`'s `cameras` already uses) used ONLY for its `token_entity("scene",
# slot)` accessor (unit R7), which maps a scene-bank local slot back onto the entity id `support_matrix` keys its
# graph by -- the same identifier
# space `relgen.support`'s own `TokenIndex`-based labels assume, resolved here directly off a live `Batch` instead
# of an offline `TokenIndex`. `ix.force_flow`'s `flow` operator (`ClosureOp`, relations/ops.py, unedited) computes
# the transitive closure of whichever direct edge channel it is pointed at (`params.edge="support"`) itself, at
# attention time -- this function only needs to supply that one direct channel.
#
# Exposed as a pure function only (matching R18's `candidate_interaction_edges` precedent exactly) -- NOT wired into
# `relation_token_sets` / any net's `RelCtx.edges`. That wiring, plus the PROBE-sourced half of this same EdgeSet
# (built from `ix.support`'s own learned bilinear pair estimate rather than the privileged label), is net-side
# plumbing this unit does not do: `relations/ops.py`'s `BilinearOp.features` returns q/k KERNEL features for the
# `aug` form, never the raw `[B, Q, K]` pair score, and `FieldReadouts` explicitly skips `op="bilinear"` -- exposing
# that score as an `EdgeSet` needs a new read hook there, which is an operator-level change this unit's rules
# forbid making unilaterally (`relations/ops.py` never touched; see research/decisions.md D-144 addendum, flagged
# in lead_questions rather than implemented). Only the GT/label half (this function) is buildable without one.
def support_edges(inputs: list, batch: "Batch", graphs: list | None = None, views: list | None = None) -> EdgeSet:
    """`EdgeSet(SUPPORT_REL_VOCAB, [B,C,C,1])`, `prov="privileged"` (docs `Prov`; this is ground truth, the training
    /diagnostics-only `source="gt"` side of `ix.force_flow` -- the deploy guard blocks it outside training, same as
    every other privileged edge/label in this registry). `data[i, q, k, 0] = 1.0` iff the scene entity resolved at
    ctx position `q` directly supports the one at `k` (`graphs[i]`, one `relgen.support.support_matrix(view)` dict
    per item; the transitive closure is `ix.force_flow`'s own operator's job, not this function's). A missing
    `graphs[i]` / `views[i]` (`None`, or either list shorter than `inputs`), or a scene with fewer than two entities
    `token_entity` can resolve, contributes an all-zero row -- never an error, matching
    `candidate_interaction_edges`'s degenerate-batch behaviour."""
    B, C = batch.B, batch.ctx_mask.shape[1]
    data = np.zeros((B, C, C, 1), np.float32)
    scene_off = batch.bank_offset["scene"]
    scene_kind_all = batch.bank_kind["scene"].numpy()
    scene_mask_all = batch.bank_mask["scene"].numpy()
    graphs, views = graphs or [], views or []
    for i in range(min(B, len(graphs), len(views))):
        graph, view = graphs[i], views[i]
        if graph is None or view is None:
            continue
        slots = np.where(scene_mask_all[i] & (scene_kind_all[i] == 0))[0]
        if len(slots) == 0:
            continue
        idx_of_id = {}
        for s in slots.tolist():
            eid = view.token_entity("scene", s)
            if eid is not None:
                idx_of_id[eid] = scene_off + s
        if len(idx_of_id) < 2:
            continue
        for a, row in graph.items():
            qi = idx_of_id.get(a)
            if qi is None:
                continue
            for b, supports in row.items():
                if not supports:
                    continue
                ki = idx_of_id.get(b)
                if ki is not None:
                    data[i, qi, ki, 0] = 1.0
    return EdgeSet(SUPPORT_REL_VOCAB, torch.from_numpy(data), prov="privileged")
