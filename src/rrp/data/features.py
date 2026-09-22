"""Featurizer: the ONLY definition of what a policy may see.

Inputs: PolicyObservation (public), RobotSpec (public static morphology), and FK computed
from measured joints with the declared kinematic model (public). Never PrivilegedTruth.

Output `PolicyInput` has four typed context banks plus action-node descriptors:
  morph   : one token per actuated joint (action entity) + passive/mimic joints + assemblies
  scene   : one token per canonical object slot (+ optional VLM image tokens, added later)
  task    : event tokens + ordered role-slot incidence tokens + predicate-estimate tokens
  interact: receipt/frame tokens (anchors, frames) + per-manipulator touch/grip tokens
Pointers (incidence) are explicit integer indices; `relations` lists typed token pairs for
structural attention bias. The unstructured baseline replaces pointers by serialized text
features of the same facts (see `serialize_pointers`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.observation import PolicyObservation
from rrp.contracts.robot import RobotSpec
from rrp.contracts.task import EntityBinding, OutputBinding

ROLE_TYPES = ["actor", "patient", "instrument", "target", "source", "destination", "reference", "support",
              "cooperating_actor"]
STATUS = ["pending", "ready", "active", "succeeded", "failed", "cancelled", "blocked"]
ASM_KINDS = ["arm", "hand", "gripper", "leg", "base", "torso", "head", "tool", "wheel_base", "body"]
OUT_TYPES = ["frame_estimate", "contact_anchor", "alignment_receipt", "completion_receipt"]
REL = {  # typed structural relations (query token -> key token)
    "same_node": 0, "node_in_assembly": 1, "actor_of": 2, "support_of": 3, "patient_of": 4, "target_of": 5,
    "destination_of": 6, "enables": 7, "maintained": 8, "output_to": 9, "produced": 10, "consumed_by": 11,
    "pred_arg": 12, "node_actor_of": 13, "role_in_event": 14, "role_points_to": 15, "kin_parent": 16,
}
N_REL = len(REL)
HASH_DIM = 16
BANKS = ["morph", "scene", "task", "interact"]


def text_hash(s: str, dim: int = HASH_DIM) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding (fixed, not learned, not a name table)."""
    v = np.zeros(dim, np.float32)
    for w in s.lower().replace("_", " ").split():
        h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 20) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, np.float32)
    if 0 <= i < n:
        v[i] = 1
    return v


@dataclass
class ActionSpace:
    """Per action-node normalization from PUBLIC spec (no demonstration statistics).
    arm joints: a = (target - q_chunk_start) / delta_scale ; gripper: a in [-1 (open), +1 (closed)]."""
    node_group: list          # command group name per action node
    node_col: list            # column within group
    lower: np.ndarray
    upper: np.ndarray
    is_gripper: np.ndarray
    open_value: np.ndarray
    closed_value: np.ndarray
    delta_scale: float = 0.5
    joint_qadr_names: list = field(default_factory=list)

    def normalize(self, groups_seq: list[dict], q0: np.ndarray) -> np.ndarray:
        H = len(groups_seq)
        a = np.zeros((H, len(self.node_group)), np.float32)
        for h, g in enumerate(groups_seq):
            for n, (gn, c) in enumerate(zip(self.node_group, self.node_col)):
                v = g[gn][c]
                if self.is_gripper[n]:
                    a[h, n] = 2 * (v - self.open_value[n]) / (self.closed_value[n] - self.open_value[n]) - 1
                else:
                    a[h, n] = (v - q0[n]) / self.delta_scale
        return a

    def denormalize(self, a: np.ndarray, q0: np.ndarray) -> list[dict]:
        out = []
        for h in range(a.shape[0]):
            g: dict[str, list] = {}
            for n, (gn, c) in enumerate(zip(self.node_group, self.node_col)):
                if self.is_gripper[n]:
                    v = self.open_value[n] + (a[h, n] + 1) / 2 * (self.closed_value[n] - self.open_value[n])
                else:
                    v = q0[n] + self.delta_scale * a[h, n]
                v = float(np.clip(v, self.lower[n], self.upper[n]))
                g.setdefault(gn, [None] * (max(cc for gg, cc in zip(self.node_group, self.node_col) if gg == gn) + 1))
                g[gn][c] = v
            out.append(g)
        return out


def action_space(spec: RobotSpec, gripper_meta: dict | None = None) -> ActionSpace:
    contract = spec.controller_contracts[0]
    groups, cols, lo, hi, isg, ov, cv, jn = [], [], [], [], [], [], [], []
    acts = {a.address: a for a in spec.actuators}
    jaddr = {j.address: j for j in spec.joints}
    for g in contract.command_groups:
        for c, addr in enumerate(g.actuators):
            groups.append(g.name)
            cols.append(c)
            lo.append(g.lower[c])
            hi.append(g.upper[c])
            grip = g.semantic == "gripper"
            isg.append(grip)
            if grip:
                three = bool(gripper_meta and gripper_meta.get("kind") == "three_finger")
                ov.append(g.lower[c] if three else g.upper[c])
                cv.append(g.upper[c] if three else g.lower[c])
            else:
                ov.append(0.0)
                cv.append(0.0)
            j = acts[addr].joint
            jn.append(jaddr[j].name if j else None)
    return ActionSpace(groups, cols, np.array(lo), np.array(hi), np.array(isg), np.array(ov), np.array(cv),
                       joint_qadr_names=jn)


@dataclass
class PolicyInput:
    """Numpy token banks for one observation (unbatched)."""
    tokens: dict            # bank -> [T_b, F_b] float32
    token_kind: dict        # bank -> [T_b] int (subtype id within bank)
    act_node_feats: np.ndarray          # [N, F_node]  static+dynamic features of action nodes
    act_node_morph_index: np.ndarray    # [N] index of the node's token in the morph bank
    relations: np.ndarray   # [R, 5] (q_bank, q_idx, k_bank, k_idx, rel_type); q_bank -1 = action node
    pointers: np.ndarray    # [P, 4] (src_bank, src_idx, dst_bank, dst_idx) incidence pointers
    pointer_text: dict      # bank -> [T_b, HASH_DIM] serialized-text replacement for pointer features
    q0: np.ndarray          # [N] measured position of action-node joints (for delta actions)
    meta: dict = field(default_factory=dict)


class Featurizer:
    """Stateless given (spec, meta); uses a private MjData for public FK."""

    def __init__(self, model: mujoco.MjModel, spec: RobotSpec, prefix: str, meta: dict, base_pos, base_yaw: float,
                 manipulator_bindings: dict, robot_index: int = 0):
        self.model = model
        self.spec = spec
        self.prefix = prefix
        self.meta = meta
        self.data = mujoco.MjData(model)
        self.base = np.asarray(base_pos, float)
        self.base_yaw = float(base_yaw)
        self.bindings = manipulator_bindings        # task manipulator entity -> assembly id
        self.robot_index = robot_index
        self.aspace = action_space(spec, meta.get("gripper_params"))
        self._build_static()

    # ------------------------------------------------------------------ static morphology
    def _build_static(self):
        s, m = self.spec, self.model
        jaddr = {j.address: j for j in s.joints}
        links = {l.address: l for l in s.links}
        children = {}
        for j in s.joints:
            children.setdefault(j.parent_link, []).append(j)
        depth = {}
        for j in s.joints:
            depth[j.address] = j.child_link.count("/")
        acts = [a for a in s.actuators]
        contract = s.controller_contracts[0]
        order = [addr for g in contract.command_groups for addr in g.actuators]
        amap = {a.address: a for a in acts}
        asm_of_link = {}
        for a in s.assemblies:
            for mem in a.members:
                if mem not in asm_of_link or len(a.members) < len(next(x for x in s.assemblies
                                                                         if x.id == asm_of_link[mem]).members):
                    asm_of_link[mem] = a.id
        self.asm_ids = [a.id for a in s.assemblies]
        mimic_count = {}
        for j in s.joints:
            if j.mimic_of:
                mimic_count[j.mimic_of] = mimic_count.get(j.mimic_of, 0) + 1
        n_desc = {}
        for j in s.joints:     # downstream joint count
            n_desc[j.address] = sum(1 for k in s.joints if k.child_link.startswith(j.child_link + "/"))
        feats, jnames, asm_idx = [], [], []
        for addr in order:
            a = amap[addr]
            j = jaddr[a.joint]
            l = links[j.child_link]
            rng = j.range or [-math.pi, math.pi]
            asm = asm_of_link.get(j.child_link)
            kind = next(x.kind for x in s.assemblies if x.id == asm) if asm else "body"
            f = np.concatenate([
                one_hot(["hinge", "slide"].index(j.type) if j.type in ("hinge", "slide") else -1, 2),
                np.asarray(j.axis, np.float32),
                np.array([rng[0] / math.pi, rng[1] / math.pi, (rng[1] - rng[0]) / (2 * math.pi)], np.float32),
                np.array([depth[j.address] / 10, n_desc[j.address] / 10, math.log(l.mass + 1e-3),
                          float(np.linalg.norm(l.pos_in_parent)), math.log((a.kp or 1.0) + 1) / 10,
                          math.log(abs((a.force_range or [0, 1])[1]) + 1) / 10,
                          float(mimic_count.get(j.address, 0))], np.float32),
                one_hot(ASM_KINDS.index(kind) if kind in ASM_KINDS else len(ASM_KINDS) - 1, len(ASM_KINDS)),
                np.array([1.0 if a.address in {x for g in contract.command_groups if g.semantic == "gripper"
                                               for x in g.actuators} else 0.0], np.float32),
            ])
            feats.append(f)
            jnames.append(j.name)
            asm_idx.append(self.asm_ids.index(asm) if asm in self.asm_ids else -1)
        self.node_static = np.stack(feats).astype(np.float32)
        self.node_joint_names = jnames
        self.node_asm = np.array(asm_idx)
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jnames])
        self.dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jnames])
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jnames]
        self.ranges = np.array([(jaddr[amap[a].joint].range or [-math.pi, math.pi]) for a in order], np.float32)
        # passive (mimic/unactuated) joints are morphology tokens but NOT action entities
        self.passive = [j for j in s.joints if j.type in ("hinge", "slide") and j.name not in jnames]
        self.all_robot_joint_names = [j.name for j in s.joints if j.type in ("hinge", "slide")]
        self.parent_node = []
        link_to_node = {jaddr[amap[a].joint].child_link: i for i, a in enumerate(order)}
        for addr in order:
            j = jaddr[amap[addr].joint]
            p, par = j.parent_link, -1
            while p is not None:
                if p in link_to_node:
                    par = link_to_node[p]
                    break
                pj = next((x for x in s.joints if x.child_link == p), None)
                p = pj.parent_link if pj else None
            self.parent_node.append(par)
        self.static_dim = self.node_static.shape[1]

    # ------------------------------------------------------------------ helpers
    def _to_base(self, p):
        c, s_ = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        d = np.asarray(p, float) - self.base
        return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]], np.float32)

    def _fk(self, q_robot: dict[str, float]):
        self.data.qpos[:] = self.model.qpos0
        for n, v in q_robot.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.data.qpos[self.model.jnt_qposadr[jid]] = v
        mujoco.mj_kinematics(self.model, self.data)

    # ------------------------------------------------------------------ main
    def __call__(self, obs: PolicyObservation, prev_action: np.ndarray | None = None) -> PolicyInput:
        ns = obs.measured_node_state
        addr_to_val = dict(zip(ns.joint_addresses, ns.qpos.tolist()))
        vel_map = dict(zip(ns.joint_addresses, ns.qvel.tolist()))
        name_by_addr = {f"{self.robot_index}:{j.address}": j.name for j in self.spec.joints}
        q_robot = {name_by_addr[a]: v for a, v in addr_to_val.items() if a in name_by_addr}
        qd_robot = {name_by_addr[a]: v for a, v in vel_map.items() if a in name_by_addr}
        self._fk(q_robot)
        d = self.data
        N = len(self.node_joint_names)
        q0 = np.array([q_robot[n] for n in self.node_joint_names], np.float32)
        qd = np.array([qd_robot[n] for n in self.node_joint_names], np.float32)
        mid = self.ranges.mean(1)
        half = np.maximum((self.ranges[:, 1] - self.ranges[:, 0]) / 2, 1e-3)
        # Jacobian of the grasping assembly's TCP w.r.t. each actuated joint (public FK model)
        gasm = next((a for a in self.spec.assemblies if a.kind in ("gripper", "hand")), None)
        tcp_w = np.zeros(3)
        jac = np.zeros((6, self.model.nv))
        if gasm is not None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, gasm.frame.site)
            mujoco.mj_comPos(self.model, d)
            mujoco.mj_jacSite(self.model, d, jac[:3], jac[3:], sid)
            tcp_w = d.site_xpos[sid].copy()
        self._tcp_base = self._to_base(tcp_w)
        c, s_ = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        Rb = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        dyn = []
        for i, n in enumerate(self.node_joint_names):
            jid = self.jids[i]
            anchor = self._to_base(d.xanchor[jid])
            axis = d.xaxis[jid].astype(np.float32)
            pa = prev_action[i] if prev_action is not None else 0.0
            jp = Rb @ jac[:3, self.dadr[i]]
            jr = Rb @ jac[3:, self.dadr[i]]
            lever = self._tcp_base - anchor
            dyn.append(np.concatenate([[(q0[i] - mid[i]) / half[i], np.clip(qd[i] * 0.1, -3, 3), pa],
                                       anchor, axis, jp, jr, lever]).astype(np.float32))
        node_feats = np.concatenate([self.node_static, np.stack(dyn)], 1)

        # ---------------- morph bank: action nodes, passive joints, assemblies
        morph, mkind = [], []
        for i in range(N):
            morph.append(node_feats[i])
            mkind.append(0)
        F_m = node_feats.shape[1]
        for j in self.passive:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            v = np.zeros(F_m, np.float32)
            v[:2] = one_hot(["hinge", "slide"].index(j.type), 2)
            v[2:5] = j.axis
            v[-15:-12] = self._to_base(d.xanchor[jid])
            v[-12:-9] = d.xaxis[jid]
            morph.append(v)
            mkind.append(1)
        asm_tok_index = {}
        for a in self.spec.assemblies:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, a.frame.site)
            v = np.zeros(F_m, np.float32)
            pos = self._to_base(d.site_xpos[sid])
            R = d.site_xmat[sid].reshape(3, 3)
            v[:10] = one_hot(ASM_KINDS.index(a.kind), len(ASM_KINDS))
            v[10:13] = pos
            v[13:16] = R[:, 2]
            v[16:19] = R[:, 0]
            caps = ["grasp", "support", "push", "locomote", "observe", "carry", "insert", "mobile_base"]
            v[19:27] = [1.0 if c in a.capabilities else 0.0 for c in caps]
            asm_tok_index[a.id] = len(morph)
            morph.append(v)
            mkind.append(2)

        # ---------------- scene bank: canonical slots (public tracker)
        scene, skind = [], []
        slot_tok = {}
        for od in obs.object_descriptors:
            known = od.position_estimate is not None
            pos = self._to_base(od.position_estimate) if known else np.zeros(3, np.float32)
            std = np.sqrt(np.asarray(od.position_cov_diag, np.float32)) if od.position_cov_diag else np.ones(3)
            rel = (pos - self._tcp_base) if known else np.zeros(3, np.float32)
            v = np.concatenate([pos, np.log(std + 1e-4) / 5, [float(od.visible), float(known)],
                                text_hash(od.descriptor), rel]).astype(np.float32)
            slot_tok[od.slot] = len(scene)
            scene.append(v)
            skind.append(0)
        F_s = 3 + 3 + 2 + HASH_DIM + 3
        if not scene:
            scene.append(np.zeros(F_s, np.float32))
            skind.append(1)   # explicit null token

        # entity id -> token (bank, index): manipulators -> assembly tokens, objects/features -> slots
        ti = obs.task_input
        ent_tok = {}
        if ti is not None:
            ents = {e.id: e for e in ti.definition.entity_declarations}
            for ent, asm in self.bindings.items():
                if asm in asm_tok_index:
                    ent_tok[ent] = ("morph", asm_tok_index[asm])
            for e in ents.values():
                if e.type in ("object", "feature"):
                    m = [od.slot for od in obs.object_descriptors if od.descriptor in e.descriptor
                         or e.descriptor in od.descriptor]
                    if m:
                        ent_tok[e.id] = ("scene", slot_tok[m[0]])

        # ---------------- task bank: events, role incidences, predicate estimates
        task, tkind, ptext_task = [], [], []
        rels, ptrs = [], []
        ev_tok = {}
        F_t = HASH_DIM + len(STATUS) + 3 + HASH_DIM + len(ROLE_TYPES) + 1 + 4 + 3
        if ti is not None:
            views = {v.event_id: v for v in ti.events}
            for e in ti.definition.events:
                v = views.get(e.id)
                x = np.zeros(F_t, np.float32)
                x[:HASH_DIM] = text_hash(e.operator)
                st = STATUS.index(v.status) if v else 0
                x[HASH_DIM:HASH_DIM + 7] = one_hot(st, 7)
                x[HASH_DIM + 7:HASH_DIM + 10] = [v.attempt if v else 0, math.log1p(v.rejection_count) if v else 0,
                                                  len(e.requires_completed) + len(e.requires_active)]
                if v and v.last_rejection_reasons:
                    x[HASH_DIM + 10:2 * HASH_DIM + 10] = text_hash(" ".join(v.last_rejection_reasons[-2:]))
                ev_tok[e.id] = len(task)
                task.append(x)
                tkind.append(0)
                ptext_task.append(text_hash(e.operator))
            for e in ti.definition.events:
                eidx = ev_tok[e.id]
                for d_ in e.requires_completed:
                    rels.append(("task", eidx, "task", ev_tok[d_], REL["enables"]))
                for d_ in e.requires_active:
                    rels.append(("task", eidx, "task", ev_tok[d_], REL["maintained"]))
                for sl in e.roles:
                    x = np.zeros(F_t, np.float32)
                    off = 2 * HASH_DIM + 10
                    x[off:off + 9] = one_hot(ROLE_TYPES.index(sl.role), 9)
                    x[off + 9] = sl.ordinal
                    b = sl.binding
                    ridx = len(task)
                    rels.append(("task", ridx, "task", eidx, REL["role_in_event"]))
                    rels.append(("task", eidx, "task", ridx, REL["role_in_event"]))
                    if isinstance(b, EntityBinding):
                        tgt = ent_tok.get(b.entity.id)
                        kind = 0 if tgt and tgt[0] == "morph" else (1 if tgt else 3)
                        x[off + 10:off + 14] = one_hot(kind, 4)
                        txt = f"{sl.role} {b.entity.id} " + " ".join(
                            d.descriptor for d in ti.definition.entity_declarations if d.id == b.entity.id)
                        if tgt:
                            ptrs.append(("task", ridx, tgt[0], tgt[1]))
                            rels.append(("task", ridx, tgt[0], tgt[1], REL["role_points_to"]))
                            rtype = {"actor": "actor_of", "cooperating_actor": "actor_of", "support": "support_of",
                                     "patient": "patient_of", "target": "target_of", "destination": "destination_of"
                                     }.get(sl.role, "target_of")
                            rels.append((tgt[0], tgt[1], "task", eidx, REL[rtype]))
                            rels.append(("task", eidx, tgt[0], tgt[1], REL[rtype]))
                    else:
                        x[off + 10:off + 14] = one_hot(2, 4)
                        rels.append(("task", ev_tok[b.event_id], "task", eidx, REL["output_to"]))
                        rels.append(("task", eidx, "task", ev_tok[b.event_id], REL["output_to"]))
                        ptrs.append(("task", ridx, "task", ev_tok[b.event_id]))
                        txt = f"{sl.role} output {b.output_name} of {b.event_id} attempt {b.attempt}"
                        # validity of the bound receipt (public)
                        rv = [r for r in ti.receipts if r.event_id == b.event_id and r.attempt == b.attempt
                              and r.output_name == b.output_name]
                        x[off + 14:off + 17] = [float(bool(rv) and rv[-1].valid), float(bool(rv)),
                                                math.log1p(rv[-1].age_s) if rv else 0.0]
                    task.append(x)
                    tkind.append(1)
                    ptext_task.append(text_hash(txt))
            for pe in obs.predicate_estimates:
                x = np.zeros(F_t, np.float32)
                x[:HASH_DIM] = text_hash(pe.predicate)
                val = pe.value
                x[HASH_DIM:HASH_DIM + 3] = [float(val) if isinstance(val, (bool, int, float)) and val is not None
                                            else 0.0, float(pe.known), pe.confidence]
                pidx = len(task)
                for a in pe.args:
                    if a in ent_tok:
                        tb, tix = ent_tok[a]
                        ptrs.append(("task", pidx, tb, tix))
                        rels.append(("task", pidx, tb, tix, REL["pred_arg"]))
                        rels.append((tb, tix, "task", pidx, REL["pred_arg"]))
                task.append(x)
                tkind.append(2)
                ptext_task.append(text_hash(pe.predicate + " " + " ".join(pe.args)))
        if not task:
            task.append(np.zeros(F_t, np.float32))
            tkind.append(3)
            ptext_task.append(np.zeros(HASH_DIM, np.float32))

        # ---------------- interaction bank: receipts/frames + touch/grip per manipulator
        inter, ikind, ptext_int = [], [], []
        F_i = 4 + 3 + 3 + 3 + 3 + HASH_DIM
        if ti is not None:
            for r in ti.receipts:
                x = np.zeros(F_i, np.float32)
                x[:4] = one_hot(OUT_TYPES.index(r.type) if r.type in OUT_TYPES else -1, 4)
                if "pos" in r.value:
                    x[4:7] = self._to_base(r.value["pos"])
                if r.value.get("normal") is not None:
                    x[7:10] = r.value["normal"]
                x[10:13] = [float(r.valid), math.log1p(r.age_s), float(bool(r.value.get("tangent_yaw_uncertain")))]
                if r.covariance_diag:
                    x[13:16] = np.log(np.sqrt(np.asarray(r.covariance_diag[:3])) + 1e-4) / 5
                rix = len(inter)
                if r.event_id in ev_tok:
                    ptrs.append(("interact", rix, "task", ev_tok[r.event_id]))
                    rels.append(("interact", rix, "task", ev_tok[r.event_id], REL["produced"]))
                    rels.append(("task", ev_tok[r.event_id], "interact", rix, REL["produced"]))
                    for e in ti.definition.events:
                        for sl in e.roles:
                            b = sl.binding
                            if isinstance(b, OutputBinding) and b.event_id == r.event_id and \
                                    b.attempt == r.attempt and b.output_name == r.output_name:
                                rels.append(("interact", rix, "task", ev_tok[e.id], REL["consumed_by"]))
                                rels.append(("task", ev_tok[e.id], "interact", rix, REL["consumed_by"]))
                inter.append(x)
                ikind.append(0)
                ptext_int.append(text_hash(f"{r.type} from {r.event_id}"))
        for ch in obs.declared_sensor_channels:
            if not ch.name.startswith(f"{self.robot_index}:"):
                continue
            x = np.zeros(F_i, np.float32)
            vals = ch.values.astype(np.float32)
            if ch.kind == "touch":
                x[13:16] = [np.log1p(vals.max(initial=0)), float((vals > 0.2).sum()), np.log1p(vals.mean())]
                x[16:] = text_hash("touch")
            else:
                x[13:16] = [float(vals[0]), 0, 0]
                x[16:] = text_hash("grip width")
            ix = len(inter)
            # sensors mounted on the grasping assembly
            gasm = next((a.id for a in self.spec.assemblies if a.kind in ("gripper", "hand")), None)
            if gasm in asm_tok_index:
                rels.append(("interact", ix, "morph", asm_tok_index[gasm], REL["node_in_assembly"]))
                rels.append(("morph", asm_tok_index[gasm], "interact", ix, REL["node_in_assembly"]))
            inter.append(x)
            ikind.append(1)
            ptext_int.append(text_hash(ch.kind))
        if not inter:
            inter.append(np.zeros(F_i, np.float32))
            ikind.append(2)
            ptext_int.append(np.zeros(HASH_DIM, np.float32))

        # ---------------- action-node relations
        for i in range(N):
            rels.append((-1, i, "morph", i, REL["same_node"]))
            if self.node_asm[i] >= 0:
                a_id = self.asm_ids[self.node_asm[i]]
                rels.append((-1, i, "morph", asm_tok_index[a_id], REL["node_in_assembly"]))
                # node's manipulator is actor of event (2-hop composition via bindings)
                asms = {a_id}
                parent_asm = next((x.parent_assembly for x in self.spec.assemblies if x.id == a_id), None)
                asms |= {x.id for x in self.spec.assemblies if x.parent_assembly == a_id}
                if parent_asm:
                    asms.add(parent_asm)
                if ti is not None:
                    for e in ti.definition.events:
                        for sl in e.roles:
                            b = sl.binding
                            if sl.role in ("actor", "cooperating_actor", "support") and isinstance(b, EntityBinding) \
                                    and self.bindings.get(b.entity.id) in asms:
                                rels.append((-1, i, "task", ev_tok[e.id], REL["node_actor_of"]))
            if self.parent_node[i] >= 0:
                rels.append((-1, i, "morph", self.parent_node[i], REL["kin_parent"]))
                rels.append(("morph", i, "morph", self.parent_node[i], REL["kin_parent"]))

        bank_id = {b: k for k, b in enumerate(BANKS)}
        R = np.array([[(-1 if qb == -1 else bank_id[qb]), qi, bank_id[kb], ki, r] for (qb, qi, kb, ki, r) in rels],
                     np.int64).reshape(-1, 5)
        P = np.array([[bank_id[a], b, bank_id[c], d_] for (a, b, c, d_) in ptrs], np.int64).reshape(-1, 4)
        ptext = {"morph": np.zeros((len(morph), HASH_DIM), np.float32),
                 "scene": np.stack([text_hash(od.descriptor) for od in obs.object_descriptors]) if
                 obs.object_descriptors else np.zeros((1, HASH_DIM), np.float32),
                 "task": np.stack(ptext_task), "interact": np.stack(ptext_int)}
        return PolicyInput(tokens={"morph": np.stack(morph), "scene": np.stack(scene), "task": np.stack(task),
                                   "interact": np.stack(inter)},
                           token_kind={"morph": np.array(mkind), "scene": np.array(skind), "task": np.array(tkind),
                                       "interact": np.array(ikind)},
                           act_node_feats=node_feats, act_node_morph_index=np.arange(N), relations=R, pointers=P,
                           pointer_text=ptext, q0=q0,
                           meta=dict(observation_id=obs.observation_id, spec_hash=self.spec.spec_hash,
                                     graph_version=ti.graph_version if ti else None,
                                     runtime_version=ti.runtime_version if ti else None))


BANK_DIMS = {"morph": None, "scene": 3 + 3 + 2 + HASH_DIM + 3,
             "task": HASH_DIM + len(STATUS) + 3 + HASH_DIM + len(ROLE_TYPES) + 1 + 4 + 3,
             "interact": 4 + 3 + 3 + 3 + 3 + HASH_DIM}


def morph_dim(static_dim: int) -> int:
    return static_dim + 18
