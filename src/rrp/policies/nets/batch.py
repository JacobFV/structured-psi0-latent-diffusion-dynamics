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
# R12 (D-144, docs/relations.md sections 2 & 10): `ctx` / `act` TokenSet fields, derived from token columns already
# in `PolicyInput` (no dataset rewrite). Field dims / provenance are declared once in `catalog.FIELDS`; this module
# only supplies the VALUES. `cam_uvd` needs a live camera and is attached separately (`attach_cam_uvd`).
def ctx_geometry_fields(inputs: list, batch: "Batch") -> dict:
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
      - task / interact tokens: no geometry; `pos3d` / `orient` stay invalid.
    """
    B, C = batch.B, batch.ctx_mask.shape[1]
    pos = np.zeros((B, C, 3), np.float32)
    pos_valid = np.zeros((B, C), bool)
    var = np.zeros((B, C, 3), np.float32)
    orient = np.zeros((B, C, 9), np.float32)
    orient_valid = np.zeros((B, C), bool)
    moff, soff = batch.bank_offset["morph"], batch.bank_offset["scene"]
    for i, pi in enumerate(inputs):
        mk, mt = pi.token_kind["morph"], pi.tokens["morph"]
        midx = moff + np.arange(len(mk))
        node = mk < 2
        pos[i, midx[node]] = mt[node][:, NODE_ANCHOR_SLICE]
        pos_valid[i, midx[node]] = True
        asm = mk == 2
        if asm.any():
            pos[i, midx[asm]] = mt[asm][:, ASM_POS_SLICE]
            pos_valid[i, midx[asm]] = True
            zcol, xcol = mt[asm][:, ASM_ZCOL_SLICE], mt[asm][:, ASM_XCOL_SLICE]
            R = np.stack([xcol, np.cross(zcol, xcol), zcol], axis=-1)   # columns x, y = z(cross)x, z
            orient[i, midx[asm]] = R.reshape(len(R), 9)
            orient_valid[i, midx[asm]] = True
        sk, st = pi.token_kind["scene"], pi.tokens["scene"]
        known = (sk == 0) & (st[:, SCENE_KNOWN_COL] > 0.5)
        if known.any():
            sidx = soff + np.arange(len(sk))
            pos[i, sidx[known]] = st[known][:, SCENE_POS_SLICE]
            pos_valid[i, sidx[known]] = True
            std = np.clip(np.exp(5.0 * st[known][:, SCENE_STD_SLICE]) - 1e-4, 0.0, None)
            var[i, sidx[known]] = std ** 2
    t = torch.from_numpy
    return {"pos3d": t(pos), "pos3d.valid": t(pos_valid), "pos3d.var": t(var),
            "orient": t(orient), "orient.valid": t(orient_valid)}


def ctx_id_fields(inputs: list, batch: "Batch") -> dict:
    """`entity_id` / `assembly_id` of the `ctx` bank (docs/relations.md section 2, R15 `id.same_*`), read off the
    EXISTING `relations` / `pointers` arrays (no dataset rewrite):
      - `entity_id`: every ctx token defaults to its OWN flattened ctx index (self-identifying); a token that is
        the source of EXACTLY ONE incidence pointer (a task-role token referencing an entity, a receipt
        referencing an event) instead takes that pointer's DESTINATION index, so two tokens referring to the same
        public entity compare equal under the `same` op. A predicate-estimate token can point at SEVERAL argument
        entities at once (`pred_arg`, one pointer per argument); such a fan-out source is not one entity's alias,
        so it keeps its own self id (its per-argument bindings stay available as `pred_arg` edges). Always valid.
      - `assembly_id`: every ctx token defaults to -1 (invalid, no assembly); a morph assembly token's own id is
        its flattened index; any OTHER ctx token with a `node_in_assembly` edge to one (today: per-manipulator
        sensor tokens of the `interact` bank) takes that assembly's index. Action-NODE `assembly_id` (the `act`
        set) is `act_assembly_id` below: a node's `node_in_assembly` edge is act>ctx (`relations[:, 0] == -1`),
        never a ctx>ctx one.
    """
    B, C = batch.B, batch.ctx_mask.shape[1]
    ent = np.zeros((B, C), np.int64)
    asm = -np.ones((B, C), np.int64)
    offs = batch.bank_offset
    for i, pi in enumerate(inputs):
        for b in BANKS:
            n = len(pi.token_kind[b])
            ent[i, offs[b]:offs[b] + n] = offs[b] + np.arange(n)
        mk = pi.token_kind["morph"]
        aidx = offs["morph"] + np.where(mk == 2)[0]
        asm[i, aidx] = aidx
        R = np.asarray(pi.relations).reshape(-1, 5)
        # `node_in_assembly` is stored BOTH directions (member -> assembly AND assembly -> member, e.g.
        # featurizer.py's sensor-token relations), so only the member -> assembly direction, i.e. rows whose KEY
        # is itself a morph assembly token, means "the query token belongs to this assembly" -- the reverse rows
        # (an assembly token as query) must NOT overwrite that assembly's own self id.
        ctx_edges = R[(R[:, 0] != -1) & (R[:, 4] == NODE_IN_ASSEMBLY) & (R[:, 2] == MORPH_BANK_ID)]
        if len(ctx_edges):
            kind_ok = mk[ctx_edges[:, 3]] == 2
            ctx_edges = ctx_edges[kind_ok]
        if len(ctx_edges):
            qflat = np.array([offs[BANKS[qb]] for qb in ctx_edges[:, 0]]) + ctx_edges[:, 1]
            kflat = offs["morph"] + ctx_edges[:, 3]
            asm[i, qflat] = kflat
        P = np.asarray(pi.pointers).reshape(-1, 4)
        if len(P):
            sflat = np.array([offs[BANKS[sb]] for sb in P[:, 0]]) + P[:, 1]
            dflat = np.array([offs[BANKS[db]] for db in P[:, 2]]) + P[:, 3]
            uniq, counts = np.unique(sflat, return_counts=True)
            single = set(uniq[counts == 1].tolist())
            keep = np.array([s in single for s in sflat])
            ent[i, sflat[keep]] = dflat[keep]
    ctx_mask_np = batch.ctx_mask.numpy()
    ent_valid = ctx_mask_np.copy()
    asm_valid = (asm >= 0) & ctx_mask_np
    t = torch.from_numpy
    return {"entity_id": t(ent[..., None].astype(np.float32)), "entity_id.valid": t(ent_valid),
            "assembly_id": t(asm[..., None].astype(np.float32)), "assembly_id.valid": t(asm_valid)}


def act_assembly_id(inputs: list, batch: "Batch") -> dict:
    """`assembly_id` of the `act` token set (action nodes): the node's `node_in_assembly` act>ctx edge
    (`relations[:, 0] == -1`), giving its owning morph assembly token's flattened `ctx` index."""
    B, N = batch.node_mask.shape
    offs = batch.bank_offset
    asm = -np.ones((B, N), np.int64)
    for i, pi in enumerate(inputs):
        R = np.asarray(pi.relations).reshape(-1, 5)
        node_edges = R[(R[:, 0] == -1) & (R[:, 4] == NODE_IN_ASSEMBLY)]
        if len(node_edges):
            kflat = np.array([offs[BANKS[kb]] for kb in node_edges[:, 2]]) + node_edges[:, 3]
            asm[i, node_edges[:, 1]] = kflat
    valid = (asm >= 0) & batch.node_mask.numpy()
    t = torch.from_numpy
    return {"assembly_id": t(asm[..., None].astype(np.float32)), "assembly_id.valid": t(valid)}


def attach_cam_uvd(pos3d: torch.Tensor, pos3d_valid: torch.Tensor, cameras: list) -> dict:
    """`cam_uvd` from an ALREADY-DERIVED `pos3d` field, projected per batch item through its own declared scene
    camera (`envs.mujoco.sensors.project_points`; `cameras[i] = (model, data, cam_name)`, one triple per item --
    collation has no live MuJoCo state of its own, so this is a separate step from `ctx_geometry_fields`).
    Positions behind the camera or with invalid `pos3d` stay invalid."""
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


def relation_token_sets(inputs: list, batch: "Batch", cameras: list | None = None) -> dict:
    """R12: the `ctx` / `act` `TokenSet`s (docs/relations.md section 2) carrying every field this unit owns.
    `cameras` (one `(model, data, cam_name)` triple per item) also attaches `cam_uvd`; omit it to skip that
    field (e.g. when only ids / geometry are needed, or no camera is declared for the scene)."""
    fields = {**ctx_geometry_fields(inputs, batch), **ctx_id_fields(inputs, batch)}
    if cameras is not None:
        fields.update(attach_cam_uvd(fields["pos3d"], fields["pos3d.valid"], cameras))
    ctx_kind = torch.cat([_global_kind(b, batch.bank_kind[b]) for b in BANKS], 1)
    ctx = TokenSet(name="ctx", mask=batch.ctx_mask, kind=ctx_kind, fields=fields)
    act = TokenSet(name="act", mask=batch.node_mask, fields=act_assembly_id(inputs, batch))
    return {"ctx": ctx, "act": act}


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
