"""W13 humanoid task command layers (label scripted_teacher, PRIVILEGED truth) and the `rl_expert` policy.

StepsHeadingTeacher (h_steps): walk +x at 0.6 vx_max, steering yaw and lateral offset to 0 -- the same law as the GPU expert
env (rrp.envs.warp.task_env.WarpStepsEnv._command). GapTeacher is the h_gap counterpart.

`rl_expert` (POLICIES key, `make_rl_expert(arg="<body>:<version>")`, architecture 14.3): the joint-level expert is a registered
LearnedTracker actor (rrp.envs.mujoco.legged_tracker.TRACKERS), loaded by the session (`make_legged_env(..., tracker=spec)`),
under the scripted command layer of the task. Its source label states what the ACTOR consumes: `learned:rl_expert:<sha>` when
its extra inputs are public (none, or the D-146 terrain scan), `privileged_teacher:rl_expert:<sha>` when they are task truth.
The command layer reads the base pose truth either way, so the policy still requires the env's privileged truth.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from rrp.core.provenance import file_digest
from rrp.core.action import NativeCommand
from rrp.core.provenance import source_label
from rrp.envs.mujoco.humanoid_scenes import GAP_SPEED_FRAC, GAP_TURN_RATE, STEPS_SPEED_FRAC, waist_joint
from rrp.envs.mujoco.legged import TRACKER_HZ
from rrp.policies.base import Act, PolicyInfo, Requirements


class StepsHeadingTeacher:
    source = "scripted_teacher"          # the scripted command layer (contract literal)
    privileged = True

    def __init__(self, session):
        self.s = session
        L = session.robots[0].meta["legged"]
        self.vx = STEPS_SPEED_FRAC * L["command_ranges"]["vx"][1]
        self.L = float(session.scenario.meta["L"])
        self.done = False

    def command_values(self) -> np.ndarray:
        rt = self.s.runtime
        if rt.succeeded() or any(i.status == "failed" for i in rt.instances.values()):
            self.done = True
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()            # PRIVILEGED
        tgt = 0.5 * math.atan(-y / self.L)
        head = math.atan2(math.sin(tgt - yaw), math.cos(tgt - yaw))
        return np.array([self.vx, 0.0, float(np.clip(1.5 * head, -GAP_TURN_RATE, GAP_TURN_RATE))])

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": self.command_values().tolist()},
                             source="scripted_teacher")


class GapTeacher:
    """h_gap_sidestep command layer (scripted_teacher, PRIVILEGED truth): the WarpGapEnv._command law. Phase 1: keep yaw 0,
    walk +x at 0.5 vx_max and sidestep onto the gap centre; phase 2 (0.5 L past the wall): stop and turn to psi_f."""
    source = "scripted_teacher"
    privileged = True

    def __init__(self, session):
        self.s = session
        self.r = session.robots[0].meta["legged"]["command_ranges"]
        self.m = session.scenario.meta
        self.phase2 = 0.0

    def command_values(self) -> np.ndarray:
        x, y, yaw = self.s.base_pose_truth()
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        if x > self.m["wall_x"] + 0.5 * self.m["L"]:
            self.phase2 = 1.0
        if self.phase2:
            e = wrap(self.m["psi_f"] - yaw)
            return np.array([0.0, 0.0, 0.0 if abs(e) < 0.1 else float(np.clip(1.5 * e, -GAP_TURN_RATE, GAP_TURN_RATE))])
        vy = float(np.clip(1.5 * math.cos(yaw) * (self.m["y_c"] - y), self.r["vy"][0], self.r["vy"][1]))
        return np.array([GAP_SPEED_FRAC * self.r["vx"][1], vy, float(np.clip(1.5 * wrap(-yaw), -GAP_TURN_RATE, GAP_TURN_RATE))])

    def act(self) -> NativeCommand:
        return NativeCommand(controller_version=self.s.controller_version(), groups={"base_velocity": self.command_values().tolist()},
                             source="scripted_teacher")


# ------------------------------------------------------------------ U2: inverse kinematics on the public body model
def _ancestors(model, body: int) -> set:
    out = set()
    while body > 0:
        out.add(int(body))
        body = int(model.body_parentid[body])
    return out


class UpperIK:
    """Damped-least-squares IK of the two palm bars on a scratch MjData of the SAME compiled model (the public body model).
    `solve(qpos, side, target)` moves the arm's own joints (the held joints whose body is an ancestor of that hand and not of the
    other one: waist and head stay put) so that the palm-bar centre reaches `target` (world) with the bar in the sagittal plane
    of the trunk (no lateral component: the bar is the forearm axis of these humanoids, so a palm squeeze from the two sides
    needs both bars to point forward / down, not out). Returns (upper joint values of that arm, position error in m).
    Everything is kinematics from the state it is given; the caller decides which state (the teacher passes the true one)."""

    def __init__(self, model: mujoco.MjModel, binding, palm_gids: dict):
        self.m, self.b, self.gid = model, binding, dict(palm_gids)
        self.d = mujoco.MjData(model)
        hand = {s: int(model.geom_bodyid[g]) for s, g in self.gid.items()}
        chain = {s: _ancestors(model, h) for s, h in hand.items()}
        jid = [int(model.actuator_trnid[a, 0]) for a in binding.held_act]
        self.arm = {}
        for s in self.gid:
            other = chain[next(k for k in self.gid if k != s)]
            idx = [i for i, j in enumerate(jid) if int(model.jnt_bodyid[j]) in chain[s] and int(model.jnt_bodyid[j]) not in other]
            self.arm[s] = np.array(idx, int)
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self._derive_bars(jid)

    BAR_ASPECT = 1.5          # a palm geom whose longest local half-extent is this many times the next one IS the bar (t1 cylinder, h1 capsule)

    def _derive_bars(self, jid):
        """Per side, from the model at the default stance: `bar_local` = the palm bar's axis in the palm geom frame (the geom's long
        axis; for a compact hand mesh (g1) the forearm direction = from the elbow anchor (the most distal arm joint at least half
        the shoulder-to-palm distance from the palm) to the palm centre), `palm_r` = the bar's half-thickness (the smaller
        cross-axis half-extent), `shoulder_j` = the most proximal arm joint. All derived, recorded by `record()`."""
        m, b, d = self.m, self.b, self.d
        b.set_default(d)
        mujoco.mj_kinematics(m, d)
        self.bar_local, self.palm_r, self.shoulder_j, self.bar_kind = {}, {}, {}, {}
        for s, g in self.gid.items():
            js = [jid[i] for i in self.arm[s]]
            depth = [_depth(m, int(m.jnt_bodyid[j])) for j in js]
            self.shoulder_j[s] = js[int(np.argmin(depth))]
            h = np.asarray(m.geom_aabb[g, 3:], float)
            order = np.argsort(h)[::-1]
            if h[order[0]] >= self.BAR_ASPECT * h[order[1]]:
                ax = np.eye(3)[order[0]]
                self.bar_kind[s] = "geom_long_axis"
                self.palm_r[s] = float(h[order[1]])
            else:
                pc = d.geom_xpos[g]
                reach = float(np.linalg.norm(pc - d.xanchor[self.shoulder_j[s]]))
                far = [(dp, j) for dp, j in zip(depth, js) if np.linalg.norm(pc - d.xanchor[j]) >= 0.5 * reach]
                elbow = max(far)[1]
                w = pc - d.xanchor[elbow]
                ax = d.geom_xmat[g].reshape(3, 3).T @ (w / np.linalg.norm(w))
                self.bar_kind[s] = "forearm"
                self.palm_r[s] = float(np.min(h[np.argsort(np.abs(ax))[:2]]))
            self.bar_local[s] = ax
        # the chest: the deepest common ancestor of the two shoulders (t1: the root Trunk; g1 / h1: the torso above the waist yaw, NOT
        # the pelvis root) -- the frame the arms move with when the waist twists
        common = set.intersection(*[_ancestors(m, int(m.body_parentid[int(m.jnt_bodyid[j])])) for j in self.shoulder_j.values()])
        self.chest_bid = max(common, key=lambda bb: _depth(m, bb)) if common else int(b.root_bid)

    def bar_axis(self, d, side: str) -> np.ndarray:
        return d.geom_xmat[self.gid[side]].reshape(3, 3) @ self.bar_local[side]

    def record(self) -> dict:
        return {"chest": self.m.body(self.chest_bid).name, **{s: dict(bar=self.bar_kind[s], bar_local=[round(float(x), 3) for x in self.bar_local[s]], palm_r=round(self.palm_r[s], 4),
                        shoulder=self.m.joint(self.shoulder_j[s]).name) for s in self.gid}}

    def palm(self, side: str) -> np.ndarray:
        return self.d.geom_xpos[self.gid[side]].copy()

    def force_offset(self, qpos, side: str, force, kp) -> np.ndarray:
        """Joint offset `J^T F / kp` that makes the arm's position servos press the palm with the world force `force` (N): a joint
        space squeeze turns along an arc round the shoulder and pushes partly down; this one is the straight push asked for."""
        m, b, d, ix = self.m, self.b, self.d, self.arm[side]
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacGeom(m, d, self.jp, self.jr, self.gid[side])
        return self.jp[:, b.held_dadr[ix]].T @ np.asarray(force, float) / np.asarray(kp, float)

    def solve(self, qpos, side: str, target, iters: int = 80, axis_w: float = 0.5, damping: float = 0.02,
              max_step: float = 0.25, level_w: float = 0.0, axis_z: float = 0.0, q_ref=None, posture_w: float = 0.0):
        m, b, d, ix = self.m, self.b, self.d, self.arm[side]
        d.qpos[:] = qpos
        qa, da = b.held_qadr[ix], b.held_dadr[ix]
        lo, hi = b.held_lo[ix], b.held_hi[ix]
        target = np.asarray(target, float)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            g = self.gid[side]
            mujoco.mj_jacGeom(m, d, self.jp, self.jr, g)
            err = target - d.geom_xpos[g]
            axis = self.bar_axis(d, side)
            lat = d.xmat[self.chest_bid].reshape(3, 3)[:, 1]              # the chest's lateral axis
            ja = lat @ np.cross(self.jr[:, da].T, axis).T
            rows, res = [self.jp[:, da], axis_w * ja[None]], [err, [-axis_w * float(lat @ axis)]]
            if level_w > 0.0:                                             # bar elevation held at `axis_z` (0: level)
                up = np.array([0.0, 0.0, 1.0])
                rows.append(level_w * (up @ np.cross(self.jr[:, da].T, axis).T)[None])
                res.append([level_w * (axis_z - float(up @ axis))])
            if posture_w > 0.0 and q_ref is not None:                    # posture rows: stay near the joint reference
                rows.append(posture_w * np.eye(len(ix)))
                res.append(posture_w * (np.asarray(q_ref, float) - d.qpos[qa]))
            J, e = np.vstack(rows), np.concatenate(res)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
            n = float(np.max(np.abs(dq)))
            if n > max_step:
                dq *= max_step / n
            d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)
            if np.linalg.norm(err) < 1e-4 and np.linalg.norm(dq) < 1e-4:
                break
        mujoco.mj_kinematics(m, d)
        return d.qpos[qa].copy(), float(np.linalg.norm(target - d.geom_xpos[self.gid[side]]))


def _geom_world_box(model, data, g: int) -> tuple[np.ndarray, np.ndarray]:
    """World-axis bounding box (lo, hi) of geom `g` from its local AABB (exact for boxes, conservative for round shapes)."""
    c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    r = data.geom_xmat[g].reshape(3, 3)
    ctr = data.geom_xpos[g] + r @ c
    ext = np.abs(r) @ h
    return ctr - ext, ctr + ext


class BodyStance:
    """Morphology-derived support and ankle-balance parameters of a humanoid body at its default stance (the public body model:
    every number below is computed from the compiled model, nothing is per body; `record()` lists them).

    * soles: per foot body, the centre of the sole = the middle of the world-x / y extent of the foot body's contact geoms at the
      lowest contact level, stored in the foot body's frame (the foot touch site is not the sole centre on every body: on g1 it is
      one heel-corner contact sphere of four).
    * ankle pitch joints: per leg, the most distal policy joint with a lateral axis (no joint-name rule; h1's single-DoF ankle and the
      2-DoF ankles of t1 / g1 alike).
    * ankle CoM-feedback gains: the ankle servos alone (kp K summed over both ankles) are below the gravity stiffness m g h of the
      inverted pendulum on most humanoids, so the static-support teachers add u = KP e + KI int(e) + KD de/dt (e = CoM ahead of the
      support centre, m) to both ankle targets. Closed-loop stiffness K (1 + KP h) = ALPHA m g h, damping ratio ZETA of the
      point-mass pendulum (inertia m h^2, servo damping included), integral rate BETA x the closed-loop natural frequency. ALPHA,
      ZETA, BETA are dimensionless; they are the values of the gains validated on t1 (U2: KP 8 rad/m, KI 2, KD 0.5, 20/20 per task),
      so t1 keeps its gains (`tests/unit/test_humanoid_teacher_morph.py`) and the other bodies get the same closed loop."""
    ALPHA, ZETA, BETA = 3.176, 0.2673, 0.0398
    G = 9.81

    def __init__(self, model: mujoco.MjModel, binding):
        m, b = model, binding
        d = mujoco.MjData(m)
        b.set_default(d)
        mujoco.mj_forward(m, d)
        self.mass = float(m.body_subtreemass[b.root_bid])
        self.com0 = d.subtree_com[b.root_bid].copy()
        fb = [int(m.site_bodyid[s]) for s in b.foot_sids]
        self.foot_body = fb
        self.foot_rot0 = [d.xmat[f].reshape(3, 3).copy() for f in fb]      # default foot orientations (heading 0)
        self.sole_local, self.sole_half_x = [], []
        for f in fb:
            gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == f and (m.geom_contype[g] or m.geom_conaffinity[g])]
            if not gs:
                raise ValueError(f"foot body {m.body(f).name!r} has no contact geom")
            boxes = [_geom_world_box(m, d, g) for g in gs]
            zmin = min(lo[2] for lo, _ in boxes)
            low = [(lo, hi) for lo, hi in boxes if lo[2] <= zmin + 0.01]          # the sole: contact geoms within 1 cm of the lowest
            lo = np.min([x[0] for x in low], axis=0)
            hi = np.max([x[1] for x in low], axis=0)
            ctr = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
            self.sole_local.append(d.xmat[f].reshape(3, 3).T @ (ctr - d.xpos[f]))
            self.sole_half_x.append(0.5 * float(hi[0] - lo[0]))
        self.support_x0 = float(np.mean([self._sole(d, i)[0] for i in range(len(fb))]))
        self.width0 = float(abs(self._sole(d, 0)[1] - self._sole(d, 1)[1]))           # default lateral sole spacing
        # ankle pitch joints: per leg the most distal lateral-axis joint among the policy joints of that leg
        jid = [int(m.actuator_trnid[a, 0]) for a in b.pol_act]
        self.ankle_ix = []
        for f in fb:
            chain = _ancestors(m, f)
            cand = [i for i, j in enumerate(jid) if int(m.jnt_bodyid[j]) in chain
                    and abs(float((d.xaxis[j])[1])) > 0.9]
            if not cand:
                raise ValueError(f"no lateral-axis (pitch) joint in the leg of {m.body(f).name!r}")
            self.ankle_ix.append(max(cand, key=lambda i: _depth(m, int(m.jnt_bodyid[jid[i]]))))
        self.ankle_ix = np.array(self.ankle_ix, int)
        acts = b.pol_act[self.ankle_ix]
        aj = [jid[i] for i in self.ankle_ix]
        self.K = float(np.sum(m.actuator_gainprm[acts, 0]))
        self.kd = float(np.sum(-m.actuator_biasprm[acts, 2]) + np.sum(m.dof_damping[m.jnt_dofadr[aj]]))
        anchors = np.array([d.xanchor[j] for j in aj])
        self.h = float(self.com0[2] - anchors[:, 2].mean())
        # dCoM_x / dq of an ankle with the foot planted: the body above turns by -dq about the joint axis through the anchor
        dx = [-float(np.cross(d.xaxis[j], self.com0 - d.xanchor[j])[0]) for j in aj]
        self.sign = -1.0 if np.mean(dx) < 0 else 1.0          # u is added as -sign * (...): a CoM ahead is pulled back
        mgh, I = self.mass * self.G * self.h, self.mass * self.h ** 2
        self.KP = max(0.0, (self.ALPHA * mgh / self.K - 1.0) / self.h)
        k_net = self.K * (1.0 + self.KP * self.h) - mgh
        self.omega = math.sqrt(max(k_net, 1e-9) / I)
        self.KD = max(0.0, 2.0 * self.ZETA * math.sqrt(max(k_net, 1e-9) * I) - self.kd) / (self.K * self.h)
        self.KI = self.BETA * self.KP * self.omega

    def feet_yaw(self, d) -> float:
        """Mean heading of the feet relative to their default-stance orientation (rad)."""
        ys = []
        for f, r0 in zip(self.foot_body, self.foot_rot0):
            rel = d.xmat[f].reshape(3, 3) @ r0.T
            ys.append(math.atan2(rel[1, 0], rel[0, 0]))
        return float(math.atan2(np.mean(np.sin(ys)), np.mean(np.cos(ys))))

    def _sole(self, d, i: int) -> np.ndarray:
        f = self.foot_body[i]
        return d.xpos[f] + d.xmat[f].reshape(3, 3) @ self.sole_local[i]

    def support_centre(self, d) -> np.ndarray:
        """World position of the middle of the two soles (live state)."""
        return np.mean([self._sole(d, i) for i in range(len(self.foot_body))], axis=0)

    def record(self) -> dict:
        return dict(mass=round(self.mass, 3), com_h=round(self.h, 4), ankle_K=self.K, ankle_kd=round(self.kd, 3),
                    support_x0=round(self.support_x0, 4), sole_half_x=[round(x, 4) for x in self.sole_half_x],
                    KP=round(self.KP, 3), KI=round(self.KI, 3), KD=round(self.KD, 3), sign=self.sign)


def _depth(model, body: int) -> int:
    n = 0
    while body > 0:
        n, body = n + 1, int(model.body_parentid[body])
    return n


def _rot_err(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    """Small-angle world rotation vector taking frame r_cur to r_tgt."""
    return 0.5 * sum(np.cross(r_cur[:, i], r_tgt[:, i]) for i in range(3))


class SquatPlanner:
    """Static squat by inverse kinematics on the public body model: both feet stay where the default stance puts them (position
    and orientation), the trunk root is lowered by `dz` and pitched forward by `pitch` about the hip axis, and the pelvis is slid
    fore-aft until the whole-body centre of mass is over the middle of the soles. `pose(dz, pitch)` returns the leg joint
    targets (policy order), the root pose and the residuals; `ok` says the IK converged inside the joint limits with the CoM
    balanced. This is a PLAN, not a controller: the squat teachers command it with an ankle CoM feedback (`SquatPickTeacher.legs_command`)."""

    def __init__(self, model: mujoco.MjModel, binding):
        self.m, self.b = model, binding
        self.d = mujoco.MjData(model)
        binding.set_default(self.d)
        mujoco.mj_kinematics(model, self.d)
        mujoco.mj_comPos(model, self.d)
        self.q_default = self.d.qpos.copy()
        self.root0 = self.d.qpos[binding.qa:binding.qa + 3].copy()
        fb = [int(model.site_bodyid[s]) for s in binding.foot_sids]
        self.foot_body = fb
        self.foot_pos = [self.d.xpos[i].copy() for i in fb]
        self.foot_rot = [self.d.xmat[i].reshape(3, 3).copy() for i in fb]
        jid = [int(model.actuator_trnid[a, 0]) for a in binding.pol_act]
        chain = [_ancestors(model, i) for i in fb]
        self.leg_ix = [np.array([k for k, j in enumerate(jid) if int(model.jnt_bodyid[j]) in c], int) for c in chain]
        assert sorted(sum((list(x) for x in self.leg_ix), [])) == list(range(len(jid))), "legs do not partition the policy joints"
        self.hip = np.mean([self.d.xanchor[jid[ix[0]]] for ix in self.leg_ix], axis=0)
        self.stance = BodyStance(model, binding)
        self.com_ref_x = self.stance.support_x0                 # the middle of the soles (not the foot touch sites: g1's is a heel corner)
        # IK seed: the default leg pose moved IN_RANGE of each joint range inside its limits. A default pose on a limit (g1: straight
        # knees, q = 0 next to the -0.087 stop) is a kinematic singularity of the foot-placement IK; t1 / h1 defaults are inside already.
        span = binding.hi - binding.lo
        self.q_seed = np.clip(binding.q0, binding.lo + self.IN_RANGE * span, binding.hi - self.IN_RANGE * span)
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    IN_RANGE = 0.05

    def _root(self, dz: float, pitch: float, hx: float):
        R = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        pos = self.hip + np.array([hx, 0.0, -dz]) + R @ (self.root0 - self.hip)
        return pos, np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])

    def _legs(self, q, iters: int = 60):
        m, b, d = self.m, self.b, self.d
        d.qpos[:] = q
        for leg, ix in enumerate(self.leg_ix):
            qa, da = b.pol_qadr[ix], b.pol_dadr[ix]
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                fb = self.foot_body[leg]
                mujoco.mj_jacBody(m, d, self.jp, self.jr, fb)
                e = np.concatenate([self.foot_pos[leg] - d.xpos[fb], _rot_err(d.xmat[fb].reshape(3, 3), self.foot_rot[leg])])
                J = np.vstack([self.jp[:, da], self.jr[:, da]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                d.qpos[qa] = np.clip(d.qpos[qa] + np.clip(dq, -0.3, 0.3), b.lo[ix], b.hi[ix])
                if np.linalg.norm(e) < 1e-5:
                    break
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        err = max(float(np.linalg.norm(np.concatenate([self.foot_pos[i] - d.xpos[f],
                                                       _rot_err(d.xmat[f].reshape(3, 3), self.foot_rot[i])])))
                  for i, f in enumerate(self.foot_body))
        return d.qpos[b.pol_qadr].copy(), err

    def pose(self, dz: float, pitch: float, com_tol: float = 0.005, foot_tol: float = 2e-3) -> dict:
        b, d = self.b, self.d
        q = self.q_default.copy()
        q[b.pol_qadr] = self.q_seed
        hx, hist = 0.0, []
        for _ in range(12):
            pos, quat = self._root(dz, pitch, hx)
            q[b.qa:b.qa + 3], q[b.qa + 3:b.qa + 7] = pos, quat
            q[b.pol_qadr], err = self._legs(q)
            q = d.qpos.copy()
            com = float(d.subtree_com[b.root_bid][0]) - self.com_ref_x
            hist.append((hx, com))
            if abs(com) < com_tol:
                break
            hx -= com if len(hist) < 2 or hist[-1][1] == hist[-2][1] else \
                com * (hist[-1][0] - hist[-2][0]) / (hist[-1][1] - hist[-2][1])
        at_limit = bool(np.any(q[b.pol_qadr] <= b.lo + 1e-3) or np.any(q[b.pol_qadr] >= b.hi - 1e-3))
        return dict(ok=bool(err < foot_tol and abs(com) < com_tol and not at_limit), legs=q[b.pol_qadr].copy(), root_pos=pos,
                    root_quat=quat, foot_err=err, com_err=com, at_limit=at_limit, dz=dz, pitch=pitch)

    def servo_targets(self, pose: dict) -> np.ndarray:
        """Leg position-servo targets that HOLD `pose` statically: legs + tau / kp, tau = the static joint torques of the planned
        configuration (gravity of the whole body carried by the two soles: vertical forces at the sole centres, split so that their
        moment balances the CoM laterally; tau = qfrc_bias - J_sole^T F). Without it the servos sag under the body weight (g1 / h1
        sink 5-10 cm below a deep squat plan and sit back off their heels). Derived from the model; clipped to the control range."""
        m, b, d = self.m, self.b, self.d
        d.qpos[:] = self.q_default
        d.qpos[b.qa:b.qa + 3], d.qpos[b.qa + 3:b.qa + 7] = pose["root_pos"], pose["root_quat"]
        d.qpos[b.pol_qadr] = pose["legs"]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        st = self.stance
        soles = [st._sole(d, i) for i in range(len(st.foot_body))]
        W = st.mass * BodyStance.G
        yc = float(d.subtree_com[b.root_bid][1])
        (yl, yr) = soles[0][1], soles[1][1]
        wl = float(np.clip((yc - yr) / (yl - yr), 0.0, 1.0)) if abs(yl - yr) > 1e-6 else 0.5
        tau = d.qfrc_bias[b.pol_dadr].copy()
        for i, (p, w) in enumerate(zip(soles, (wl, 1.0 - wl))):
            mujoco.mj_jac(m, d, self.jp, self.jr, p, st.foot_body[i])
            tau -= self.jp[:, b.pol_dadr].T @ np.array([0.0, 0.0, w * W])
        kp = m.actuator_gainprm[b.pol_act, 0]
        return np.clip(np.asarray(pose["legs"], float) + tau / kp, b.lo, b.hi)

    def at_feet(self, data, base_xy, base_yaw: float) -> "SquatPlanner":
        """A copy of this planner whose foot targets are the CURRENT feet of `data` (same compiled model), expressed in the planning
        frame (base at the origin, heading +x): after a walk the feet are wherever the gait left them, and the default-stance plan
        forced onto them twists the pelvis (t1 h_loco_pick: 0.8 rad of yaw at the hand-over, then a fall). The CoM reference
        becomes the middle of the current soles."""
        import copy
        p = copy.copy(self)
        p.d = mujoco.MjData(self.m)
        self.b.set_default(p.d)
        c, s_ = math.cos(base_yaw), math.sin(base_yaw)
        Rz = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])           # world -> planning frame
        o = np.array([base_xy[0], base_xy[1], 0.0])
        p.foot_pos, p.foot_rot = [], []
        for i, f in enumerate(self.foot_body):              # planned FLAT on the floor: the current xy and yaw, the default height / tilt
            pos = Rz @ (data.xpos[f] - o)
            pos[2] = self.foot_pos[i][2]
            rel = Rz @ data.xmat[f].reshape(3, 3) @ self.foot_rot[i].T
            fy = math.atan2(rel[1, 0], rel[0, 0])
            Ry = np.array([[math.cos(fy), -math.sin(fy), 0.0], [math.sin(fy), math.cos(fy), 0.0], [0.0, 0.0, 1.0]])
            p.foot_pos.append(pos)
            p.foot_rot.append(Ry @ self.foot_rot[i])
        sole = np.mean([self.stance._sole(data, i) for i in range(len(self.foot_body))], axis=0)
        p.com_ref_x = float((Rz @ (sole - o))[0])
        p.jp, p.jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        return p

    def sway(self, pose: dict, delta: float = 0.01) -> np.ndarray:
        """d(leg joints) / d(lateral pelvis shift) at `pose` with both feet planted (central difference of the foot-placement IK):
        the joint direction that slides the pelvis sideways, used by the lateral CoM feedback (`WholebodyTeacher.com_balanced`)."""
        b, legs = self.b, []
        for sg in (1.0, -1.0):
            q = self.q_default.copy()
            q[b.qa:b.qa + 3] = np.asarray(pose["root_pos"]) + np.array([0.0, sg * delta, 0.0])
            q[b.qa + 3:b.qa + 7] = pose["root_quat"]
            q[b.pol_qadr] = pose["legs"]
            legs.append(self._legs(q)[0])
        return (legs[0] - legs[1]) / (2.0 * delta)

    def stance_pose(self) -> dict:
        """The standing plan: the shallowest root drop (0 .. STANCE_DZ_L x L) whose pose is `ok`. Bodies whose default stance has
        straight knees (g1) cannot slide the CoM over the sole centre without a small knee bend; t1 / h1 get dz = 0."""
        L = float(self.b.L["nominal_height"])
        for dz in np.linspace(0.0, self.STANCE_DZ_L * L, 9):
            p = self.pose(float(dz), 0.0)
            if p["ok"]:
                return p
        raise RuntimeError("no standing pose with the CoM over the soles within the joint limits")

    STANCE_DZ_L = 0.05


# ------------------------------------------------------------------ U2: whole-body scripted teachers (upper body IK + registered tracker legs)
def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


_PLANS: dict = {}


def _stance(session) -> BodyStance:
    """Not cached across sessions: it holds body ids of THIS compiled scene (scenes add bodies before the robot, e.g. h_place's mark)."""
    return BodyStance(session.model, session.binding)


class WholebodyTeacher:
    """Base of the U2 teachers (source `scripted_teacher`, PRIVILEGED: it reads the true base pose, the true joint state and the
    task map). Every 50 Hz tick it returns one NativeCommand with the `legs` group = the session's registered body tracker
    (`session.body_tracker`, the rl_expert actor) acting on the teacher's base command (the static-support squat teachers replace it
    by their planned pose, see `legs_command`), and the `upper` group = the teacher's own arm / waist targets. `command_values()` is the base velocity command
    handed to the tracker; `upper_target(t)` the held-joint targets."""
    source = "scripted_teacher"
    privileged = True
    name = "wholebody"
    legs = "rl_expert"                        # source of the `legs` group: the tracker actor; "planned_com" = the static plan + ankle feedback

    def __init__(self, session):
        self.s, self.b, self.m = session, session.binding, session.model
        self.bt = session.body_tracker
        self.bt.reset(0.0)
        self.dt = 1.0 / TRACKER_HZ
        self.k = 0
        self.upper = self.b.q0_held.copy()
        self.r = session.robots[0].meta["legged"]["command_ranges"]
        self.sc = session.scenario.meta
        self.stance = _stance(session)                      # morphology-derived soles, ankle pitch joints, balance gains
        self.ankle_ix = self.stance.ankle_ix                # per leg the most distal lateral-axis joint (h1: the single ankle hinge)
        if len(self.ankle_ix) != 2:
            raise ValueError(f"{self.name}: expected two ankle pitch joints among the policy joints, got {len(self.ankle_ix)}")
        self._com_prev, self._com_int = None, 0.0

    @property
    def t(self) -> float:
        return self.k * self.dt

    def command_values(self) -> np.ndarray:
        return np.zeros(3)

    def upper_target(self) -> np.ndarray:
        return self.upper

    def legs_command(self) -> np.ndarray:
        """Joint targets of the `legs` group (policy order): the registered tracker on the base command."""
        return np.asarray(self.bt.act(self.s.data, self.command_values()), float)

    KPY, KIY, KDY = 1.0, 1.0, 0.2       # lateral CoM feedback: pelvis shift (m) per m of CoM error, per m s (integral, clipped at
    #                                     LAT_I m s) and per m/s (damping: without it g1's sideways sway grew into a fall)
    LAT_I = 0.05

    def com_balanced(self, legs: np.ndarray, sway: np.ndarray | None = None) -> np.ndarray:
        """`legs` (a planned static pose, both feet planted) plus an ankle-pitch feedback on the whole-body CoM x over the middle of the
        soles (P + I + D on the true CoM; the ankle servos alone (kp 50) sag and the open-loop plan topples). Static-support tasks use
        it instead of the tracker: the tracker's own standing height is 0.83 of the default, the plan's is 0.98."""
        d, b, st = self.s.data, self.b, self.stance
        yaw = self._feet_yaw()
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        e = float(fwd @ (d.subtree_com[b.root_bid][:2] - st.support_centre(d)[:2]))      # CoM ahead of the sole centre (m)
        de = 0.0 if self._com_prev is None else (e - self._com_prev) / self.dt
        self._com_prev, self._com_int = e, float(np.clip(self._com_int + e * self.dt, -0.3, 0.3))
        legs = np.array(legs, float)
        legs[self.ankle_ix] += -st.sign * (st.KP * e + st.KI * self._com_int + st.KD * de)
        if sway is not None:                 # lateral: slide the pelvis against the CoM's sideways error (trunk twist, one-arm reach)
            lat = np.array([-fwd[1], fwd[0]])
            ey = float(lat @ (d.subtree_com[b.root_bid][:2] - st.support_centre(d)[:2]))
            self._lat_int = float(np.clip(getattr(self, "_lat_int", 0.0) + ey * self.dt, -self.LAT_I, self.LAT_I))
            dey = 0.0 if getattr(self, "_lat_prev", None) is None else (ey - self._lat_prev) / self.dt
            self._lat_prev = ey
            legs += np.asarray(sway, float) * -(self.KPY * ey + self.KIY * self._lat_int + self.KDY * dey)
        return legs

    def _feet_yaw(self) -> float:
        return self.stance.feet_yaw(self.s.data)

    UPPER_RATE = 4.0          # rad/s: the upper-body targets move at most this fast (an IK branch switch is a ramp, not a 1.2 rad step
    #                           in one tick that throws the body: g1 7-DoF arms at the start of the opening / squeeze)

    def act(self) -> NativeCommand:
        legs = self.legs_command()
        up = np.asarray(self.upper_target(), float)
        prev = getattr(self, "_up_prev", None)
        if prev is not None:
            step = self.UPPER_RATE * self.dt
            up = prev + np.clip(up - prev, -step, step)
        self._up_prev = up.copy()
        self.k += 1
        return NativeCommand(controller_version=self.s.controller_version(),
                             groups={"legs": legs.tolist(), "upper": up.tolist()}, source="scripted_teacher")


class WalkTeacher(WholebodyTeacher):
    """h_walk (L0): walk to the goal marker at 0.6 vx_max steering the heading, stop within 0.12 m. Upper body at the default pose."""
    name = "h_walk"

    def command_values(self) -> np.ndarray:
        x, y, yaw = self.s.base_pose_truth()
        gx, gy = self.sc["goal"]
        dist = math.hypot(gx - x, gy - y)
        if dist < 0.12:
            return np.zeros(3)
        head = _wrap(math.atan2(gy - y, gx - x) - yaw)
        vx = 0.6 * self.r["vx"][1] * min(1.0, dist / 0.5) * max(0.0, math.cos(head)) ** 2
        return np.array([vx, 0.0, float(np.clip(1.5 * head, -0.5, 0.5))])


class TurnTeacher(WholebodyTeacher):
    """h_turn (L3): turn in place towards the marker at up to 0.5 rad/s; a velocity command in the body frame (P on the true base
    position, |v| <= 0.15 m/s, off once aligned so that the base comes to rest) pulls the base back to where it started (the tracker's turning steps otherwise walk it 0.17 m per rad)."""
    name = "h_turn"
    _home = None
    K_HOLD, V_HOLD = 1.5, 0.15

    def command_values(self) -> np.ndarray:
        x, y, yaw = self.s.base_pose_truth()
        if self._home is None:
            self._home = (x, y)
        mx, my = self.sc["marker"]
        err = _wrap(math.atan2(my - y, mx - x) - yaw)
        ex, ey = self._home[0] - x, self._home[1] - y
        v = np.zeros(2) if abs(err) < 0.15 else np.clip(self.K_HOLD * np.array([math.cos(yaw) * ex + math.sin(yaw) * ey, -math.sin(yaw) * ex + math.cos(yaw) * ey]),
                    -self.V_HOLD, self.V_HOLD)
        return np.array([v[0], v[1], 0.0 if abs(err) < 0.08 else float(np.clip(1.5 * err, -0.5, 0.5))])


class ReachTeacher(WholebodyTeacher):
    """h_reach (M1): the arm on the target's side reaches the point (damped least squares IK from the true state, palm-bar centre to
    the target, with a small integral offset against the servo's gravity droop); the legs hold the default stance (CoM-balanced)."""
    name = "h_reach"
    legs = "planned_com"
    T_RAMP = 2.0

    def __init__(self, session):
        super().__init__(session)
        self.ik = UpperIK(self.m, self.b, session.palm_ids())
        pl = SquatPlanner(self.m, self.b)
        st = pl.stance_pose()
        self.q_stand, self.sway_stand = pl.servo_targets(st), pl.sway(st)
        self.side = "left" if self.sc["side"] > 0 else "right"
        self.off = np.zeros(3)
        self.sol = self.b.q0_held[self.ik.arm[self.side]].copy()

    def legs_command(self) -> np.ndarray:
        return self.com_balanced(self.q_stand, sway=self.sway_stand)   # feet planted: the default stance, CoM-balanced against the reaching arm

    def upper_target(self) -> np.ndarray:
        d = self.s.data
        tgt = np.asarray(self.sc["target"], float) + self.off
        q = d.qpos.copy()
        ix = self.ik.arm[self.side]
        q[self.b.held_qadr[ix]] = self.sol
        self.sol, err = self.ik.solve(q, self.side, tgt, iters=10, axis_w=0.0)
        s = _smooth(self.t / self.T_RAMP)
        up = self.b.q0_held.copy()
        up[ix] = np.clip(up[ix] + s * (self.sol - up[ix]), self.b.held_lo[ix], self.b.held_hi[ix])
        if s >= 1.0:                                       # droop compensation once at the target
            e = np.asarray(self.sc["target"], float) - self.ik_palm_true()
            self.off = np.clip(self.off + 0.03 * e, -0.06, 0.06)
        return up

    def ik_palm_true(self) -> np.ndarray:
        return self.s.data.geom_xpos[self.ik.gid[self.side]].copy()


class SquatPickTeacher(WholebodyTeacher):
    """h_squat_pick (M2): stand still, squat (the `SquatPlanner` plan, added to the tracker's legs) while the arms open to the two
    sides of the box, squeeze the palms onto it (a 15 mm compression of the arm servos, a few N per palm), stand up with the arm
    joints frozen, hold. The plan (squat depth) is the smallest one whose open and squeeze poses the arm IK reaches inside 8 mm."""
    name = "h_squat_pick"
    legs = "planned_com"
    T0, T_SQ, T_OPEN, T_CLOSE, T_HOLD, T_UP = 0.6, 3.0, 1.5, 1.5, 0.7, 3.0
    M_OPEN, F_SQ = 0.035, 15.0                # m: palm-centre offset outside the box faces when open; N: squeeze force per palm
    I_MASK = np.array([1.0, 0.0, 1.0])
    LEAD = 0.01                                # m: the palms never rise further than this over the box centre
    LEVEL = dict(level_w=1.0, axis_z=0.0)      # the palm bars are held level from the approach on (a tilted bar pinches the box at its edges)
    PALM_DZ = 0.05                            # m: palm bar above the box centre (level bars reach that high from the squat; the pinch above the CoM hangs the box)
    POSTURE_W = 0.1                           # opening path: weight (per rad, vs 1 per m of palm position) of the arm-posture rows that pull
    #                                           the IK towards the joint-space blend hanging -> planned open pose (a redundant 7-DoF arm (g1)
    #                                           otherwise wanders overhead on the way); negligible on the 4-DoF arms (t1, h1)
    Z_CLEAR = 0.06                            # m: height of the palm-bar centre over the box centre on the way in
    W_CARRY = 1.0                             # squat fraction while carrying (h_place lowers it)

    def __init__(self, session):
        super().__init__(session)
        self.ik = UpperIK(self.m, self.b, session.palm_ids())
        self.planner = SquatPlanner(self.m, self.b)
        self.plan = self._plan()
        self.off = {s: np.zeros(3) for s in self.ik.gid}
        self.sol = {s: self.b.q0_held[self.ik.arm[s]].copy() for s in self.ik.gid}
        self.q_open = self.q_sq = None
        self.path_start = None
        self.dq_sq = {}
        self.axis_z0, self.sh0, self.rel0, self.yaw0 = {sd: 0.0 for sd in self.ik.gid}, {}, {}, 0.0
        self.sh_jid = dict(self.ik.shoulder_j)            # the most proximal arm joint (no joint-name rule)
        self.tl = self._timeline()

    # -- timeline (times in s): each entry is the END of the phase
    def _timeline(self) -> dict:
        t = {}
        t["squat"] = self.T0 + self.T_SQ
        t["open"] = t["squat"] + self.T_OPEN
        t["close"] = t["open"] + self.T_CLOSE
        t["hold"] = t["close"] + self.T_HOLD
        t["up"] = t["hold"] + self.T_UP
        return t

    def _palm_targets(self, margin: float) -> dict:
        bx, by = self.sc["box"]["xy"]
        hy, z = self.sc["box"]["half"][1], self.sc["box"]["z0"] + self.PALM_DZ
        r = self.ik.palm_r
        return {"left": np.array([bx, by + hy + r["left"] + margin, z]),
                "right": np.array([bx, by - hy - r["right"] - margin, z])}

    SQUAT_PITCH = 1.0                         # nominal trunk pitch per unit squat depth in leg lengths (rad per L; t1's U2 rule 1.5 rad/m x L 0.666 m)
    DZ_MAX_L, DZ_STEP = 0.5, 0.0125           # squat search: root drop 0 .. 0.5 L in 12.5 mm steps
    PITCH_DEV = np.arange(0.0, 1.0001, 0.05)  # rad: deviations from the nominal trunk pitch, tried in this order
    PITCH_MAX = 1.2                           # rad
    BAR_Z = (0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6)   # bar elevation (up . bar axis) tried in this order: level first
    IK_TOL = 8e-3                             # m: arm IK residual accepted for the open and squeeze poses

    def _plan(self) -> dict:
        """Search (bar elevation, trunk pitch, depth) for a static squat whose legs plan is `ok` (feet planted, CoM over the sole
        centre, inside the joint limits) and from which the arm IK reaches the open and squeeze palm poses within IK_TOL, in the
        order: level bars first (BAR_Z), then the trunk pitch closest to the nominal SQUAT_PITCH dz / L, then the shallowest depth.
        Depth, pitch and bar elevation are thus derived from each body's leg / arm kinematics and joint limits: t1 keeps its U2
        plan (level bars, nominal pitch); the bar stays in the sagittal plane, i.e. flat on the box face, at any elevation."""
        key = (self.s.scenario.robots[0].robot_spec.spec_hash, repr(self.sc["box"]), repr(self.sc["crate"]), self.M_OPEN)
        if key in _PLANS:
            self.LEVEL = dict(level_w=1.0, axis_z=_PLANS[key]["axis_z"])
            return _PLANS[key]
        pl, ik, L = self.planner, self.ik, float(self.sc["L"])
        legs = {}

        def leg_pose(dz, pitch):
            k = (round(dz, 6), round(pitch, 6))
            if k not in legs:
                legs[k] = pl.pose(dz, pitch)
            return legs[k]

        free = {}

        def reach_ok(dz, pitch, q):
            k = (round(dz, 6), round(pitch, 6))
            if k not in free:
                free[k] = max(ik.solve(q, sd, tg, axis_w=0.0)[1] for m in (self.M_OPEN, 0.0) for sd, tg in self._palm_targets(m).items())
            return free[k] < self.IK_TOL

        chosen = None
        dzs = [float(x) for x in np.arange(0.0, self.DZ_MAX_L * L + 1e-9, self.DZ_STEP)]
        for az in self.BAR_Z:
            for dev in self.PITCH_DEV:
                for dz in dzs:
                    nominal = self.SQUAT_PITCH * dz / L
                    for pitch in sorted({nominal + dev, nominal - dev}, reverse=True):
                        if not 0.0 <= pitch <= self.PITCH_MAX:
                            continue
                        pose = leg_pose(dz, float(pitch))
                        if not pose["ok"]:
                            continue
                        q = pl.q_default.copy()
                        q[self.b.qa:self.b.qa + 3], q[self.b.qa + 3:self.b.qa + 7] = pose["root_pos"], pose["root_quat"]
                        q[self.b.pol_qadr] = pose["legs"]
                        if not reach_ok(dz, float(pitch), q):              # bound: not even a free-orientation palm reaches the box
                            continue
                        sols = {(m, sd): ik.solve(q, sd, tg, axis_w=0.5, level_w=1.0, axis_z=az) for m in (self.M_OPEN, 0.0)
                                for sd, tg in self._palm_targets(m).items()}
                        errs = [e for _, e in sols.values()]
                        if max(errs) < self.IK_TOL:
                            chosen = (dz, float(pitch), max(errs), az)
                            arm_open = {sd: sols[(self.M_OPEN, sd)][0] for sd in ik.gid}
                            break
                    if chosen:
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen is None:
            raise RuntimeError(f"{self.name}: no static squat within {self.DZ_MAX_L} L puts both palms on the box faces (arm IK error > "
                               f"{self.IK_TOL * 1e3:.0f} mm or the squat leaves the joint limits / the support polygon at every trunk pitch "
                               "and bar elevation)")
        dz, pitch, err, az = chosen
        self.LEVEL = dict(level_w=1.0, axis_z=az)
        ws = np.linspace(0.0, 1.0, 9)
        poses = [pl.pose(dz * w, pitch * w) for w in ws]
        _PLANS[key] = dict(dz=dz, pitch=pitch, ik_err=err, axis_z=az, arm_open=arm_open, ws=ws, legs=np.array([pl.servo_targets(p) for p in poses]),
                           sway=np.array([pl.sway(p) for p in poses]),
                           root_pos=np.array([p["root_pos"] for p in poses]), root_quat=np.array([p["root_quat"] for p in poses]),
                           ok=[bool(p["ok"]) for p in poses])
        return _PLANS[key]

    def derived(self) -> dict:
        """The body-derived parameters this teacher runs with (recorded with the results)."""
        return dict(stance=self.stance.record(), arms=self.ik.record(), squat=dict(dz=round(self.plan["dz"], 4),
                    dz_L=round(self.plan["dz"] / float(self.sc["L"]), 4), pitch=round(self.plan["pitch"], 4), bar_z=self.plan["axis_z"],
                    ik_err=round(self.plan["ik_err"], 5)))

    def _interp(self, name: str, w: float) -> np.ndarray:
        ws, arr = self.plan["ws"], self.plan[name]
        return np.array([np.interp(w, ws, arr[:, i]) for i in range(arr.shape[1])])

    def squat_w(self) -> float:
        t, tl = self.t, self.tl
        if t < self.T0:
            return 0.0
        if t < tl["squat"]:
            return _smooth((t - self.T0) / self.T_SQ)
        if t < tl["hold"]:
            return 1.0
        return 1.0 - _smooth((t - tl["hold"]) / self.T_UP)

    def legs_command(self) -> np.ndarray:
        """The planned static pose at squat fraction w (both feet stay planted, so no stepping is needed), CoM-balanced."""
        w = self.squat_w()
        return self.com_balanced(self._interp("legs", w), sway=self._interp("sway", w))

    def twist(self) -> float:
        return 0.0

    def _solve(self, targets: dict, posture: dict | None = None) -> dict:
        """Arm IK towards `targets` from the previous solution; `posture` = {side: joint reference} adds posture rows (POSTURE_W)."""
        d = self.s.data
        out = {}
        for sd, tg in targets.items():
            ix = self.ik.arm[sd]
            q = d.qpos.copy()
            q[self.b.held_qadr[ix]] = self.sol[sd]
            kw = dict(q_ref=posture[sd], posture_w=self.POSTURE_W) if posture else {}
            self.sol[sd], _ = self.ik.solve(q, sd, tg + self.off[sd], iters=10, axis_w=0.5, **self.LEVEL, **kw)
            out[sd] = self.sol[sd]
        return out

    def _integrate(self, targets: dict, mask=np.ones(3)):
        for sd, tg in targets.items():
            e = (tg - self.s.data.geom_xpos[self.ik.gid[sd]]) * mask
            self.off[sd] = np.clip(self.off[sd] + 0.03 * e, -0.05, 0.05)

    def _open_posture(self, f: float) -> dict:
        """Joint-space reference of the opening path: hanging arm (q0) blended to the plan's open-pose IK solution."""
        u = _smooth(f)
        return {sd: (1.0 - u) * self.b.q0_held[self.ik.arm[sd]] + u * self.plan["arm_open"][sd] for sd in self.ik.gid}

    def _open_targets(self, f: float) -> dict:
        """Palm targets while the arms open (fraction f of the phase): from where the palms hang, up over the crate top beside the
        box, forward, then down to the open pose: the hands come over the crate edge, they do not sweep into its front face."""
        end = self._palm_targets(self.M_OPEN)
        if self.path_start is None:
            self.path_start = {sd: self.s.data.geom_xpos[self.ik.gid[sd]].copy() for sd in end}
        u, out = _smooth(f), {}
        for sd, tg in end.items():
            st = self.path_start[sd]
            hi = tg[2] + self.Z_CLEAR
            way = [st, np.array([st[0], tg[1], hi]), np.array([tg[0], tg[1], hi]), tg]
            seg = [float(np.linalg.norm(way[i + 1] - way[i])) for i in range(3)]
            x = u * sum(seg)
            for i in range(3):
                if x <= seg[i] or i == 2:
                    out[sd] = way[i] + (way[i + 1] - way[i]) * (min(x, seg[i]) / max(seg[i], 1e-9))
                    break
                x -= seg[i]
        return out

    def _set_arms(self, sol: dict, into: np.ndarray) -> np.ndarray:
        for sd, q in sol.items():
            ix = self.ik.arm[sd]
            into[ix] = np.clip(q, self.b.held_lo[ix], self.b.held_hi[ix])
        return into

    def upper_target(self) -> np.ndarray:
        t, tl = self.t, self.tl
        up = self.b.q0_held.copy()
        if t < tl["squat"]:
            pass                                             # the arms hang while the body goes down
        elif t < tl["open"]:
            f = (t - tl["squat"]) / self.T_OPEN
            tg = self._open_targets(f)
            self._set_arms(self._solve(tg, posture=self._open_posture(f)), up)
            if f > 0.9:
                self._integrate(tg)
            self.q_open = up.copy()
        elif t < tl["close"]:
            f = (t - tl["open"]) / self.T_CLOSE
            touch = _smooth(min(1.0, f / 0.7))                         # palms come to the box faces, then press
            self._set_arms(self._solve(self._palm_targets(self.M_OPEN * (1.0 - touch))), up)
            if f > 0.7:
                self._set_arms(self._squeeze(up, _smooth((f - 0.7) / 0.3)), up)
            self.q_sq = up.copy()
            self._note_grasp()
        else:
            up = self._grip(up)
        return self._after_grasp(up)

    def _squeeze(self, up: np.ndarray, g: float) -> dict:
        d, out = self.s.data, {}
        q = d.qpos.copy()
        for sd, ix in self.ik.arm.items():
            q[self.b.held_qadr[ix]] = up[ix]
        for sd, ix in self.ik.arm.items():
            if sd not in self.dq_sq:
                inward = np.array([0.0, -1.0 if sd == "left" else 1.0, 0.0])
                self.dq_sq[sd] = self.ik.force_offset(q, sd, self.F_SQ * inward, self.m.actuator_gainprm[self.b.held_act[ix], 0])
            out[sd] = up[ix] + g * self.dq_sq[sd]
        return out

    def _chest_yaw(self) -> float:
        r = self.s.data.xmat[self.ik.chest_bid].reshape(3, 3)
        return math.atan2(r[1, 0], r[0, 0])

    def _box_face(self, sd: str) -> np.ndarray:
        """Where the palm-bar centre sits on the box's `sd` face (privileged truth: the box pose)."""
        bid = self.m.body("box").id
        r = self.s.data.xmat[bid].reshape(3, 3)
        sg = 1.0 if sd == "left" else -1.0
        return self.s.data.xpos[bid] + sg * (self.sc["box"]["half"][1] + self.ik.palm_r[sd]) * r[:, 1] + np.array([0.0, 0.0, self.PALM_DZ])

    def _note_grasp(self):
        d = self.s.data
        self.yaw0 = self._chest_yaw()
        for sd, g in self.ik.gid.items():
            self.sh0[sd] = d.xanchor[self.sh_jid[sd]].copy()
            self.rel0[sd] = d.geom_xpos[g] - self.sh0[sd]
            self.axis_z0[sd] = float(d.geom_xmat[g].reshape(3, 3)[2, 2])

    def _grip(self, up: np.ndarray) -> np.ndarray:
        """Carry: each palm keeps its grasp offset from its shoulder (turned with the chest's yaw), the bar elevation held, the
        squeeze a straight force (`_squeeze`'s J^T F recomputed here). The arm joints are not left frozen: the chest pitches up as
        the body stands, which would roll the bars off the box faces."""
        d = self.s.data
        dyaw = self._chest_yaw() - self.yaw0
        rz = np.array([[math.cos(dyaw), -math.sin(dyaw), 0.0], [math.sin(dyaw), math.cos(dyaw), 0.0], [0.0, 0.0, 1.0]])
        sh = np.r_[self._palm_shift(), 0.0]
        tg = {sd: d.xanchor[self.sh_jid[sd]] + rz @ self.rel0[sd] + sh for sd in self.ik.gid}
        for sd in tg:                                   # centre the pinch on the box (it would tip about the palms otherwise)
            bx = d.xpos[self.m.body("box").id]
            tg[sd][2] = min(tg[sd][2], bx[2] + self.PALM_DZ + self.LEAD)
        up = up.copy()
        self._integrate(tg, mask=self.I_MASK)
        self._set_arms(self._solve(tg), up)
        q = d.qpos.copy()
        for sd, ix in self.ik.arm.items():
            q[self.b.held_qadr[ix]] = up[ix]
        r = d.xmat[self.ik.chest_bid].reshape(3, 3)
        for sd, sg in (("left", 1.0), ("right", -1.0)):
            ix = self.ik.arm[sd]
            inward = -sg * r[:, 1]
            up[ix] = np.clip(up[ix] + self.ik.force_offset(q, sd, self.F_SQ * inward, self.m.actuator_gainprm[self.b.held_act[ix], 0]),
                             self.b.held_lo[ix], self.b.held_hi[ix])
        if self.t < self.tl.get("lower", math.inf):
            self.q_sq = up.copy()
        return up

    def _after_grasp(self, up: np.ndarray) -> np.ndarray:
        return up

    def _palm_shift(self) -> np.ndarray:
        """World xy shift of both palm targets in the grip law (zero: keep the grasp offset from the shoulders)."""
        return np.zeros(2)


class PlaceTeacher(SquatPickTeacher):
    """h_place (M3): the M2 pick, then rise 5 cm, twist the trunk about the waist (a servo on the box's true angle round the waist
    axis to the mark), lower, release, and stay: the arms cannot cross the midline, so the box is carried round the waist axis."""
    name = "h_place"
    T_CARRY, T_TWIST, T_LOWER, T_REL = 1.0, 3.5, 1.0, 1.5
    W_CARRY = 0.75
    PSI_RATE = 0.35                            # rad/s, twist servo rate limit
    SHIFT_KI, SHIFT_MAX = 1.0, 0.10            # 1/s, m: the arms also move the box onto the mark (integral on the box-to-mark xy error):
    #                                            the twist turns the box about the waist axis, which a pitched trunk tilts (h1 put it 5 cm short)

    def __init__(self, session):
        super().__init__(session)
        wj = waist_joint(self.m, self.b)
        if wj is None:
            raise ValueError("h_place needs a waist yaw joint")
        self.wi, self.wsign = wj
        self.wjid = int(self.m.actuator_trnid[self.b.held_act[self.wi], 0])
        self.psi = 0.0
        self.q_hold = None

    def _timeline(self) -> dict:
        t = super()._timeline()
        t["up"] = t["hold"]                     # the stand-up of M2 is replaced by the carry
        t["carry"] = t["hold"] + self.T_CARRY
        t["twist"] = t["carry"] + self.T_TWIST
        t["lower"] = t["twist"] + self.T_LOWER
        t["release"] = t["lower"] + self.T_REL
        return t

    def squat_w(self) -> float:
        t, tl = self.t, self.tl
        if t < tl["hold"]:
            return super().squat_w()
        if t < tl["carry"]:
            return 1.0 - (1.0 - self.W_CARRY) * _smooth((t - tl["hold"]) / self.T_CARRY)
        if t < tl["twist"]:
            return self.W_CARRY
        if t < tl["lower"]:
            return self.W_CARRY + (1.0 - self.W_CARRY) * _smooth((t - tl["twist"]) / self.T_LOWER)
        return 1.0

    def twist(self) -> float:
        return self.psi

    def _palm_shift(self) -> np.ndarray:
        t, tl = self.t, self.tl
        if not hasattr(self, "_shift"):
            self._shift = np.zeros(2)
        if tl["carry"] <= t < tl["lower"]:
            err = np.asarray(self.sc["place"], float) - self.s._body_pos("box")[:2]          # PRIVILEGED box pose
            self._shift = self._shift + self.SHIFT_KI * err * self.dt
            n = float(np.linalg.norm(self._shift))
            if n > self.SHIFT_MAX:
                self._shift *= self.SHIFT_MAX / n
        return self._shift

    def _box_angle_error(self) -> float:
        ax = self.s.data.xanchor[self.wjid][:2]
        bx = self.s._body_pos("box")[:2] - ax
        mk = np.asarray(self.sc["place"]) - ax
        return _wrap(math.atan2(mk[1], mk[0]) - math.atan2(bx[1], bx[0]))

    def _after_grasp(self, up: np.ndarray) -> np.ndarray:
        t, tl = self.t, self.tl
        if t < tl["hold"]:
            return up
        if tl["hold"] <= t < tl["lower"] and t >= tl["carry"]:
            self.psi += float(np.clip(1.5 * self._box_angle_error(), -self.PSI_RATE, self.PSI_RATE)) * self.dt
        self.psi = float(np.clip(self.psi, -1.4, 1.4))
        up = up.copy()
        up[self.wi] = np.clip(self.b.q0_held[self.wi] + self.wsign * self.psi, self.b.held_lo[self.wi], self.b.held_hi[self.wi])
        if t >= tl["lower"]:
            f = _smooth((t - tl["lower"]) / self.T_REL)
            arms = np.concatenate([self.ik.arm[s] for s in self.ik.arm])
            up[arms] = self.q_sq[arms] + f * (self.q_open[arms] - self.q_sq[arms])
        return up


# ------------------------------------------------------------------ U3: C1 h_carry, C2 h_loco_pick and the held-out h_steps_carry, h_gap_cart
class CarryTeacher(SquatPickTeacher):
    """h_carry (C1): the M2 squat pick, then hand the legs over to the registered tracker (an abrupt hand-over from the planned standing
    pose; a blended one dropped the box) and carry the box to the goal: a heading law on the tracker command (turn towards the goal,
    always to the LEFT: a right turn under the payload tips the tracker over; walk at 0.6 vx_max scaled by cos(heading error), slow down
    over the last 0.5 m, halt when the BOX is within 0.15 m of the goal; the box rides ~0.36 m ahead of the base) while the arm grip law
    (`_grip`) keeps the squeeze. The tracker also runs in the shadow (output discarded)
    during the pick so that its observation history (last action, gait phase) is continuous at the hand-over. PRIVILEGED: true base and box
    poses. The flat `t1:contact_v2` tracker was trained without a payload or upper-body motion: what it does with the box is measured
    (research/tracks/humanoid.md U3), not assumed."""
    name = "h_carry"
    legs = "planned_com+rl_expert"
    STOP_M, SLOW_M, REACH = 0.15, 0.5, 0.36
    V_FRAC, GATE, WZ_MAX, WZ_KP = 0.6, 1.0, 0.5, 1.5
    LEFT_ABOVE = 0.3

    def carry_started(self) -> bool:
        return self.t >= self.tl["up"]

    def target(self) -> np.ndarray:
        return np.asarray(self.sc["goal"], float)

    def payload_xy(self) -> np.ndarray:
        return np.asarray(self.s._body_pos("box")[:2], float)

    def command_values(self) -> np.ndarray:
        if not self.carry_started():
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()
        g = self.target()
        if float(np.linalg.norm(g - self.payload_xy())) < self.STOP_M:
            return np.zeros(3)
        dist = float(np.linalg.norm(g - np.array([x, y])))                   # base to goal: the payload rides REACH ahead of the base
        head = _wrap(math.atan2(g[1] - y, g[0] - x) - yaw)
        if head < -self.LEFT_ABOVE:              # the tracker turns left more stably than right under the payload: go the long way round
            head += 2.0 * math.pi
        vx = self.V_FRAC * self.r["vx"][1] * min(1.0, max(0.0, dist - self.REACH) / self.SLOW_M) * max(0.0, math.cos(head)) ** self.GATE
        return np.array([vx, 0.0, float(np.clip(self.WZ_KP * head, -self.WZ_MAX, self.WZ_MAX))])

    def legs_command(self) -> np.ndarray:
        planned = super().legs_command()
        tracker = np.asarray(self.bt.act(self.s.data, self.command_values()), float)     # shadow until the hand-over
        if not self.carry_started():
            return planned
        return tracker


class StepsCarryTeacher(CarryTeacher):
    """h_steps_carry (HELD OUT): the C1 teacher on the staircase scene (goal beyond the mirrored staircase behind the start). The legs
    are the tracker the session was built with: the flat `t1:contact_v2` has no terrain scan and will trip on the steps; a steps
    actor (`rl_expert:<body>:<steps version>`) is what this task is for once one is registered (training is paused)."""
    name = "h_steps_carry"


class LocoPickTeacher(SquatPickTeacher):
    """h_loco_pick (C2): walk to the M2 stance in front of the crate with the tracker (heading law, slow down over the last 0.3 m, a
    minimum 0.25 m/s so that the gait gate stays open), settle for 1 s, then run the M2 pick (planned squat legs + CoM feedback, arm
    IK on the TRUE box pose, so a few cm of stance error are absorbed by the arms). The squat plan is the M2 plan of the shifted scene
    (cached under the same key). The timeline of the pick starts when the walk ends (`t` is the pick clock)."""
    name = "h_loco_pick"
    legs = "rl_expert+planned_com"
    T_SETTLE, T_WALK_MAX, T_BLEND = 1.0, 20.0, 0.6
    X_TOL, X_SETTLE, Y_TOL, V_MIN = 0.05, 0.08, 0.10, 0.25          # walk stops at X_TOL; the pick starts within X_SETTLE

    def __init__(self, session):
        self.k_pick, self._still = None, 0
        super().__init__(session)

    @property
    def t(self) -> float:
        return 0.0 if self.k_pick is None else (self.k - self.k_pick) * self.dt

    def _plan(self) -> dict:
        real, a = self.sc, self.sc["approach_m"]
        back = lambda d: {**d, "xy": [d["xy"][0] - a, d["xy"][1]]}
        self.sc = {**real, "box": back(real["box"]), "crate": back(real["crate"])}         # the plan frame: the stance at the origin
        try:
            return super()._plan()
        finally:
            self.sc = real

    def command_values(self) -> np.ndarray:
        if self.k_pick is not None:
            return np.zeros(3)
        x, y, yaw = self.s.base_pose_truth()
        ex, ey = self.sc["stance_x"] - x, -y
        dist = math.hypot(ex, ey)
        if dist > 0.3:
            head = _wrap(math.atan2(ey, ex) - yaw)
            vx = self.V_FRAC * self.r["vx"][1] * min(1.0, dist / 0.5) * max(0.0, math.cos(head)) ** 2
            return np.array([max(vx, self.V_MIN), 0.0, float(np.clip(1.5 * head, -0.5, 0.5))])
        ex, ey = self._feet_err()                         # the final approach places the FEET (the squat plan is relative to the soles)
        c, s_ = math.cos(yaw), math.sin(yaw)
        vx = float(np.clip(1.5 * (c * ex + s_ * ey), 0.0, 0.3))
        vy = float(np.clip(1.5 * (-s_ * ex + c * ey), -0.1, 0.1))
        ex_b = c * ex + s_ * ey                          # stance error along the heading; an overshoot backs up (g1 stops ~0.15 m late)
        if ex_b > self.X_TOL:
            vx = max(vx, self.V_MIN)
        elif ex_b < -self.X_TOL:
            vx = max(min(1.5 * ex_b, -self.V_MIN), self.r["vx"][0])
        else:
            vx = 0.0
        fy = self._feet_yaw()                            # square the FEET to the crate (g1's gait leaves them 0.4 rad off the pelvis heading)
        if abs(ex_b) <= self.X_TOL and abs(-s_ * ex + c * ey) <= self.X_TOL and abs(fy) <= self.YAW_TOL:
            return np.zeros(3)                           # dead band: an exact zero lets the tracker stop stepping (small commands keep
        #                                                  g1's gait stepping in place with the feet swinging +-0.5 rad)
        return np.array([vx, vy, float(np.clip(-1.5 * fy, -0.3, 0.3))])

    V_FRAC = 0.6

    def _feet_err(self) -> tuple[float, float]:
        """World xy error of the sole centre from where the plan has it (the M2 stance shifted by the approach: stance_x plus the
        default sole-centre offset). After a walk the feet are not where the default stance puts them under the pelvis (g1: 4 cm
        further back), and a base-position stance left the box out of the planned arm reach."""
        sc = self.stance.support_centre(self.s.data)
        return float(self.sc["stance_x"] + self.stance.support_x0 - sc[0]), float(-sc[1])

    def _at_stance(self) -> bool:
        ex, ey = self._feet_err()
        return (abs(ex) <= self.X_SETTLE and abs(ey) <= self.Y_TOL and abs(self._feet_yaw()) <= self.YAW_TOL
                and self.s.truth_predicate("base_speed", ["body"]) < 0.08)


    YAW_TOL = 0.15

    def legs_command(self) -> np.ndarray:
        if self.k_pick is None:
            self._still = self._still + 1 if self._at_stance() else 0
            if (self._still * self.dt >= self.T_SETTLE or self.k * self.dt >= self.T_WALK_MAX) and self._double_support():
                self.k_pick = self.k
                self._replan_at_feet()
        if self.k_pick is None:
            return np.asarray(self.bt.act(self.s.data, self.command_values()), float)
        planned = super().legs_command()
        if self.t >= self.T_BLEND:
            return planned
        a = _smooth(self.t / self.T_BLEND)                    # tracker -> planned standing pose over the pick's first T_BLEND s
        return (1.0 - a) * np.asarray(self.bt.act(self.s.data, np.zeros(3)), float) + a * planned

    def _double_support(self) -> bool:
        """Both feet on the floor (contact, sole within 1 cm of its standing height): the hand-over to the planned pose never starts
        mid-step (the ub_v1 trackers keep stepping in place under small commands; a blend from a swing foot toppled t1)."""
        d, b = self.s.data, self.b
        return bool(b.contacts(d)[0].all() and np.all(b.foot_clearance(d) < 0.01))

    def _replan_at_feet(self):
        """The leg trajectory of the pick re-planned for the feet where the walk left them (same depth / pitch as the cached plan;
        the arm IK works on the true box pose anyway)."""
        x, y, _ = self.s.base_pose_truth()
        pl = self.planner.at_feet(self.s.data, (x, y), self._feet_yaw())   # the pelvis is squared to the feet, which the approach
        #                                                                   squared to the crate (a gait leaves pelvis and feet 0.4 rad apart)
        dz, pitch = self.plan["dz"], self.plan["pitch"]
        for dev in self.PITCH_DEV:                       # a staggered stance can put the planned depth past a joint limit (g1: ankle
            cands = [c for c in (pitch + dev, pitch - dev) if 0.0 <= c <= self.PITCH_MAX]          # stop): nearest feasible pitch
            ok = [c for c in cands if pl.pose(dz, c)["ok"]]
            if ok:
                pitch = ok[0]
                break
        poses = [pl.pose(dz * w, pitch * w) for w in self.plan["ws"]]
        self.plan = {**self.plan, "legs": np.array([pl.servo_targets(p) for p in poses]), "sway": np.array([pl.sway(p) for p in poses]),
                     "root_pos": np.array([p["root_pos"] for p in poses]), "root_quat": np.array([p["root_quat"] for p in poses]),
                     "ok": [bool(p["ok"]) for p in poses], "pitch": pitch, "replanned_at_feet": True}

    @property
    def approach_s(self) -> float:
        return (self.k_pick if self.k_pick is not None else self.k) * self.dt


class CartTeacher(CarryTeacher):
    """h_gap_cart (HELD OUT): grasp the cart's handle block with the M2 pick (the handle is only a few cm below the standing palm
    height: the plan squats 2-3 cm), stand, then push: the tracker walks at 0.4 vx_max steering the base at the gap centre until the
    cart has passed the wall, then at the goal, halting when the cart is within 0.12 m of it. The squeeze friction of the grasp
    carries the push (no palm is on a rear face). Legs: the tracker after the grasp (as the carry teacher)."""
    name = "h_gap_cart"
    V_FRAC, STOP_M, REACH = 0.4, 0.12, 0.45

    def payload_xy(self) -> np.ndarray:
        return np.asarray(self.s._body_pos("cart")[:2], float)

    def target(self) -> np.ndarray:
        if self.payload_xy()[0] < self.sc["wall_x"] + 0.15:
            return np.array([self.sc["wall_x"] + 0.3, self.sc["y_c"]])
        return np.asarray(self.sc["goal"], float)


MANIP_TEACHER_VERSION = "upper_ik_v1"
ALL_MANIP_TEACHERS = {"h_walk": WalkTeacher, "h_turn": TurnTeacher, "h_reach": ReachTeacher, "h_squat_pick": SquatPickTeacher,
                      "h_place": PlaceTeacher, "h_carry": CarryTeacher, "h_loco_pick": LocoPickTeacher,
                      "h_steps_carry": StepsCarryTeacher, "h_gap_cart": CartTeacher}      # one registry: POLICIES teacher:<task>


COMMAND_LAYERS = {"h_steps": StepsHeadingTeacher, "h_gap": GapTeacher}


class RLExpertPolicy:
    """A registered learned expert under the task's scripted command layer. The env must have been built with the same
    registry actor (`tracker=<spec>`); reset() checks the loaded actor's sha256 against the registry entry."""

    def __init__(self, entry, sha: str):
        self.entry, self.sha = entry, sha
        from rrp.envs.mujoco.legged_tracker import PUBLIC_EXTRA
        public = entry.extra_obs in PUBLIC_EXTRA          # none | terrain_scan | terrain_scan+range_ring (all public sensors, D-146)
        self.label = source_label("learned" if public else "privileged_teacher", f"rl_expert:{sha[:12]}")
        caps = frozenset(PUBLIC_EXTRA.get(entry.extra_obs, ()))
        self.info = PolicyInfo("rl_expert", "learned" if public else "privileged_teacher", sha,
                               Requirements(frozenset({"base_velocity"}), tasks=frozenset(COMMAND_LAYERS), privileged=True,
                                            env_capabilities=caps))
        self.teachers, self.layers = [], []

    def reset(self, spec, task, seeds, *, envs=None):
        from rrp.envs.mujoco.legged_tracker import TrackerMismatch
        name = getattr(task, "name", task)
        if name not in COMMAND_LAYERS:
            raise KeyError(f"rl_expert has no command layer for task {name!r}; has {sorted(COMMAND_LAYERS)}")
        for e in envs:
            tr = e.tracker
            if getattr(tr, "sha256", None) != self.sha:
                raise TrackerMismatch(f"rl_expert {self.entry.spec}: the env runs {getattr(tr, 'version', tr)!r} "
                                      f"(sha {str(getattr(tr, 'sha256', None))[:12]}), not sha {self.sha[:12]}")
            if self.entry.extra_obs == "privileged" and tr.extra_fn is None:
                raise TrackerMismatch(f"rl_expert {self.entry.spec} takes privileged task inputs the env does not supply")
        self.teachers = [COMMAND_LAYERS[name](e) for e in envs]
        self.layers = [f"scripted_teacher:{name}"] * len(self.teachers)

    def act(self, obs):
        return {i: Act(self.teachers[i].act(), info=dict(source_label=self.label, command_layer=self.layers[i]))
                for i in obs}


def make_rl_expert(*, arg: str | None = None, tracker: str | None = None) -> RLExpertPolicy:
    """arg / tracker: "<body>:<version>" of a registry actor (make_policy("rl_expert:<body>:<version>"))."""
    from rrp.envs.mujoco.legged_tracker import get_entry
    spec = tracker or arg
    if not spec:
        raise ValueError("rl_expert needs a tracker spec '<body>:<version>'")
    e = get_entry(spec)
    sha = file_digest(e.actor, length=None)
    if e.sha256 is not None and e.sha256 != sha:
        from rrp.envs.mujoco.legged_tracker import TrackerMismatch
        raise TrackerMismatch(f"tracker {spec} sha256 {sha[:12]} does not match its registry pin {e.sha256[:12]}")
    return RLExpertPolicy(e, sha)


class ManipTeacherPolicy:
    """Policy adapter of the U2 teachers (POLICIES keys `teacher:h_walk` ... `teacher:h_place`): source `scripted_teacher`,
    privileged. The legs are the rl_expert actor the env was built with (`session.body_tracker`, sha in the row label) for
    h_walk / h_turn and a planned static pose with CoM feedback (`planned_com` in the label) for h_reach / h_squat_pick / h_place;
    the upper body and the base command are scripted. Every Act carries the composition in its info."""

    def __init__(self, task: str):
        if task not in ALL_MANIP_TEACHERS:
            raise KeyError(f"no U2/U3 teacher for task {task!r}; has {sorted(ALL_MANIP_TEACHERS)}")
        self.task = task
        self.info = PolicyInfo(f"teacher:{task}", "scripted_teacher", MANIP_TEACHER_VERSION,
                               Requirements(frozenset({"joint_position"}), observations=frozenset(), tasks=frozenset({task}),
                                            privileged=True))
        self.teachers, self.labels = [], []

    def reset(self, spec, task, seeds, *, envs=None):
        self.teachers = [ALL_MANIP_TEACHERS[self.task](e) for e in envs]
        def legs(t, e):
            if "rl_expert" not in t.legs:
                return t.legs
            tag = f"{getattr(e.body_tracker, 'version', '?')}:{str(getattr(e.body_tracker, 'sha256', ''))[:12]}"
            return tag if t.legs == "rl_expert" else f"{t.legs}:{tag}"       # e.g. planned_com+rl_expert:contact_v2:<sha12>
        self.labels = [source_label("scripted_teacher", f"{self.task}+legs:" + legs(t, e)) for t, e in zip(self.teachers, envs)]

    def act(self, obs):
        return {i: Act(self.teachers[i].act(), info=dict(source_label=self.labels[i], command_layer=f"scripted_teacher:{self.task}"))
                for i in obs}


def make_manip_teacher_policy(*, arg: str, version: str | None = None, options: dict | None = None) -> ManipTeacherPolicy:
    return ManipTeacherPolicy(arg)
