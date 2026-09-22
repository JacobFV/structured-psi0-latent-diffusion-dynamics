"""Multi-robot featurizer: one PolicyInput over ALL robots/manipulators of a session.

Backwards compatible: the single-robot `Featurizer` is untouched; this module composes one
`Featurizer` per robot (all expressed in ONE declared workspace frame) and merges their
banks so the policy sees:
  morph   : [action nodes r0, action nodes r1, ..., passive+assembly tokens r0, r1, ...]
            -> act_node_morph_index == arange(N) exactly as for one robot
  scene   : shared canonical object slots (identical across robots)
  task    : shared event/role/predicate tokens; a role bound to ANY robot's manipulator
            (`left` -> r0 gripper, `right` -> r1 gripper, or both on one dual-arm body)
            is marked bound and points to that robot's assembly token
  interact: shared receipts + every robot's touch/width tokens, each linked to the assembly
            that carries the sensor (per-assembly channels `<i>:<asm>:<kind>` for dual bodies)
Actions: a flat action space whose command groups are namespaced `r<i>:<group>` (e.g.
`r0:arm`, `r1:gripper`, or `r0:left_arm` for ALOHA). `split_flat()` / DualSession
.command_from_flat() map them back to per-robot NativeCommands.
Only public inputs are used (PolicyObservation, RobotSpecs, FK of measured joints).
"""
from __future__ import annotations

import hashlib

import numpy as np

from rrp.contracts.observation import PolicyObservation
from .features import (Featurizer, ActionSpace, PolicyInput, BANKS, HASH_DIM, REL, one_hot)

WORKSPACE_FRAME = dict(origin=(0.0, 0.0, 0.0), yaw=0.0, name="table_world")
ROLE_KIND_OFF = 2 * HASH_DIM + 10 + 10       # role-token bound-kind one-hot offset in the task bank
BANK_ID = {b: k for k, b in enumerate(BANKS)}


def combined_hash(hashes: list[str]) -> str:
    return "multi:" + hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]


class MultiFeaturizer:
    def __init__(self, model, mounted_robots: list, frame: dict | None = None):
        fr = frame or WORKSPACE_FRAME
        self.frame = fr
        self.subs = [Featurizer(model, mr.robot_spec, mr.prefix, mr.meta, fr["origin"], fr["yaw"],
                                mr.manipulator_bindings, robot_index=i) for i, mr in enumerate(mounted_robots)]
        self.n_nodes = [len(f.node_joint_names) for f in self.subs]
        self.node_off = np.cumsum([0] + self.n_nodes[:-1]).tolist()
        self.N = int(sum(self.n_nodes))
        self.spec_hash = combined_hash([f.spec.spec_hash for f in self.subs])
        a = [f.aspace for f in self.subs]
        self.aspace = ActionSpace(
            node_group=[f"r{i}:{g}" for i, s in enumerate(a) for g in s.node_group],
            node_col=[c for s in a for c in s.node_col],
            lower=np.concatenate([s.lower for s in a]), upper=np.concatenate([s.upper for s in a]),
            is_gripper=np.concatenate([s.is_gripper for s in a]),
            open_value=np.concatenate([s.open_value for s in a]),
            closed_value=np.concatenate([s.closed_value for s in a]),
            delta_scale=a[0].delta_scale,
            joint_qadr_names=[n for s in a for n in s.joint_qadr_names])
        self.static_dim = self.subs[0].static_dim

    # public helpers used by the collection code
    def _to_base(self, p):
        return self.subs[0]._to_base(p)

    @staticmethod
    def flatten(per_robot: dict[int, dict]) -> dict:
        return {f"r{i}:{g}": list(v) for i, groups in sorted(per_robot.items()) for g, v in groups.items()}

    @staticmethod
    def split_flat(flat: dict) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for k, v in flat.items():
            ri, g = k.split(":", 1)
            out.setdefault(int(ri[1:]), {})[g] = list(v)
        return out

    def _node_grippers(self, f) -> list:
        """For bodies with several grasping assemblies (e.g. ALOHA): the gripper TCP site each
        action node drives (its own arm's gripper). None when the body has a single gripper."""
        grips = [a for a in f.spec.assemblies if a.kind in ("gripper", "hand")]
        if len(grips) < 2:
            return None
        by_id = {a.id: a for a in f.spec.assemblies}
        out = []
        for k in f.node_asm:
            a = by_id.get(f.asm_ids[k]) if k >= 0 else None
            g = None
            while a is not None and g is None:          # walk up the assembly tree
                g = a if a.kind in ("gripper", "hand") else next(
                    (x for x in grips if x.parent_assembly == a.id), None)
                a = by_id.get(a.parent_assembly) if a.parent_assembly else None
            out.append(g.frame.site if g else None)
        return out

    def _fix_multi_gripper_jacobians(self, f, pi):
        """feat-v2 node features end with [J_pos(3), J_rot(3), lever(3)] w.r.t. THE first
        gripper TCP; for multi-gripper bodies recompute them w.r.t. each node's own gripper."""
        sites = self._node_grippers(f)
        if sites is None:
            return
        import math
        import mujoco
        m, d = f.model, f.data            # holds FK at the measured q from the call just made
        mujoco.mj_comPos(m, d)
        c, s_ = math.cos(-f.base_yaw), math.sin(-f.base_yaw)
        Rb = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        jac = np.zeros((6, m.nv))
        cache = {}
        for i, site in enumerate(sites):
            if site is None:
                continue
            if site not in cache:
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
                mujoco.mj_jacSite(m, d, jac[:3], jac[3:], sid)
                cache[site] = (jac.copy(), f._to_base(d.site_xpos[sid]))
            J, tcp_b = cache[site]
            anchor = f._to_base(d.xanchor[f.jids[i]])
            v = np.concatenate([Rb @ J[:3, f.dadr[i]], Rb @ J[3:, f.dadr[i]], tcp_b - anchor]).astype(np.float32)
            pi.act_node_feats[i, -9:] = v
            pi.tokens["morph"][i, -9:] = v

    def __call__(self, obs: PolicyObservation, prev_action: np.ndarray | None = None) -> PolicyInput:
        parts = []
        for i, f in enumerate(self.subs):
            pa = None
            if prev_action is not None:
                pa = prev_action[self.node_off[i]:self.node_off[i] + self.n_nodes[i]]
            pi = f(obs, pa)
            self._fix_multi_gripper_jacobians(f, pi)
            parts.append(pi)
        # ---------------- morph bank layout
        other_off, cur = [], self.N
        for i, p in enumerate(parts):
            other_off.append(cur)
            cur += len(p.tokens["morph"]) - self.n_nodes[i]
        morph = [p.tokens["morph"][:self.n_nodes[i]] for i, p in enumerate(parts)] + \
                [p.tokens["morph"][self.n_nodes[i]:] for i, p in enumerate(parts)]
        mkind = [p.token_kind["morph"][:self.n_nodes[i]] for i, p in enumerate(parts)] + \
                [p.token_kind["morph"][self.n_nodes[i]:] for i, p in enumerate(parts)]
        # ---------------- interact bank layout: shared receipts + per-robot sensor tokens
        n_rec = int((parts[0].token_kind["interact"] == 0).sum())
        sens_idx = [np.where(p.token_kind["interact"] == 1)[0] for p in parts]
        sens_off, c = [], n_rec
        for s in sens_idx:
            sens_off.append(c)
            c += len(s)
        inter = [parts[0].tokens["interact"][:n_rec]] + [p.tokens["interact"][s] for p, s in zip(parts, sens_idx)]
        ikind = [parts[0].token_kind["interact"][:n_rec]] + [p.token_kind["interact"][s] for p, s in zip(parts, sens_idx)]
        itext = [parts[0].pointer_text["interact"][:n_rec]] + [p.pointer_text["interact"][s]
                                                               for p, s in zip(parts, sens_idx)]
        inter_t = np.concatenate(inter) if sum(len(x) for x in inter) else np.zeros(
            (1, parts[0].tokens["interact"].shape[1]), np.float32)
        ikind_t = np.concatenate(ikind) if sum(len(x) for x in ikind) else np.array([2])
        itext_t = np.concatenate(itext) if sum(len(x) for x in itext) else np.zeros((1, HASH_DIM), np.float32)

        def remap(i, bank, idx):
            if bank == BANK_ID["morph"]:
                n = self.n_nodes[i]
                return self.node_off[i] + idx if idx < n else other_off[i] + idx - n
            if bank == BANK_ID["interact"]:
                if idx < n_rec:
                    return idx
                pos = np.where(sens_idx[i] == idx)[0]
                return None if not len(pos) else sens_off[i] + int(pos[0])
            return idx

        # sensor tokens of dual-assembly bodies: link to the assembly named in the channel
        chan_asm = {}
        for i, f in enumerate(self.subs):
            chans = [ch for ch in obs.declared_sensor_channels if ch.name.startswith(f"{i}:")]
            asm_tok = {}
            k = len(f.node_joint_names) + len(f.passive)
            for a in f.spec.assemblies:
                asm_tok[a.id] = k
                k += 1
            for j, ch in enumerate(chans):
                bits = ch.name.split(":")
                if len(bits) == 3 and bits[1] in asm_tok and j < len(sens_idx[i]):
                    chan_asm[(i, int(sens_idx[i][j]))] = asm_tok[bits[1]]
        rels, ptrs = set(), set()
        bound_roles = set()
        for i, p in enumerate(parts):
            for qb, qi, kb, ki, r in p.relations.tolist():
                if qb == BANK_ID["interact"] and kb == BANK_ID["morph"] and (i, qi) in chan_asm:
                    ki = chan_asm[(i, qi)]
                if kb == BANK_ID["interact"] and qb == BANK_ID["morph"] and (i, ki) in chan_asm:
                    qi = chan_asm[(i, ki)]
                q2 = (self.node_off[i] + qi) if qb == -1 else remap(i, qb, qi)
                k2 = remap(i, kb, ki)
                if q2 is None or k2 is None:
                    continue
                rels.add((qb, int(q2), kb, int(k2), r))
            for sb, si, db, di in p.pointers.tolist():
                s2, d2 = remap(i, sb, si), remap(i, db, di)
                if s2 is None or d2 is None:
                    continue
                ptrs.add((sb, int(s2), db, int(d2)))
                if sb == BANK_ID["task"] and db == BANK_ID["morph"]:
                    bound_roles.add(si)
        task = parts[0].tokens["task"].copy()
        for r in bound_roles:            # role bound to some robot's manipulator: kind = morph pointer
            task[r, ROLE_KIND_OFF:ROLE_KIND_OFF + 4] = one_hot(0, 4)
        R = np.array(sorted(rels), np.int64).reshape(-1, 5)
        P = np.array(sorted(ptrs), np.int64).reshape(-1, 4)
        morph_t = np.concatenate(morph)
        return PolicyInput(
            tokens={"morph": morph_t, "scene": parts[0].tokens["scene"], "task": task, "interact": inter_t},
            token_kind={"morph": np.concatenate(mkind), "scene": parts[0].token_kind["scene"],
                        "task": parts[0].token_kind["task"], "interact": ikind_t},
            act_node_feats=np.concatenate([p.act_node_feats for p in parts]),
            act_node_morph_index=np.arange(self.N), relations=R, pointers=P,
            pointer_text={"morph": np.zeros((len(morph_t), HASH_DIM), np.float32),
                          "scene": parts[0].pointer_text["scene"], "task": parts[0].pointer_text["task"],
                          "interact": itext_t},
            q0=np.concatenate([p.q0 for p in parts]),
            meta=dict(parts[0].meta, spec_hash=self.spec_hash, n_robots=len(parts), nodes_per_robot=self.n_nodes,
                      frame=self.frame["name"]))


def multi_featurizer_for(session) -> MultiFeaturizer:
    return MultiFeaturizer(session.model, session.scenario.robots)
