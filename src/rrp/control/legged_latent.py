"""Legged instantiation of the controller-facing latent packet (R38) — shared morphology, context and labels.

Packet assemblies (the M axis of z[K, M, dz]) for a legged body: one `leg` assembly per foot (its actuated joints
are the joints on the chain root -> foot), one `body` assembly (the floating trunk: no actuators of its own; it
carries whole-body locomotion intent), and for humanoids one `arm` assembly per side for the held upper-body
actuators (declared: in v1 the arm assemblies are HELD at the default pose by the native layer, so their packet
slots carry no realized motion). Assembly handles are opaque (spec hash + structural address).

Clocks (declared): system i replans every 0.4 s (20 ticks); knots 4 at (0.1, 0.3, 0.5, 0.7) s; system 0 emits native
joint position targets at the tracker/native rate 50 Hz (0.02 s; faster than the 20 Hz arm clock because
locomotion PD targets are 50 Hz native); servo/physics underneath at the body's timestep.

System-0 recurrent state `osc-v1`: a free-running gait oscillator phase = ticks_since_reset * 0.02 / gait_period
(gait_period is a declared body property). It is local controller state, not an observation of the world.

PUBLIC inputs only for system i / system 0 (see `public_context`, `local_state`); privileged truth (true base pose,
true foot contacts, true waypoint positions, fall) is recorded separately and used ONLY as supervision labels for
packet probes and as evaluation truth.
"""
from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np

from .legged_core import quat_rotate_inv

TICK_DT = 0.02
TICKS_PER_PACKET = 20                    # 0.4 s replan
KNOT_TIMES = (0.1, 0.3, 0.5, 0.7)
NODE_STATIC_DIM = 21
ASM_DIM = 10
GLOBAL_DIM = 22
ASM_KINDS = ("leg", "body", "arm")
BODY_KINDS = ("quadruped", "hexapod", "humanoid", "other")
EVENTS = ("walk_to_a", "walk_to_b", "halt")
MAX_N = 32                              # padded actuated joints (policy + held; g1 has 29)
MAX_M = 11                              # padded assemblies (8 legs + body + 2 arms)


def _body_kind(L, nf):
    k = L.get("kind", "other")
    if k in ("humanoid", "biped"):
        return "humanoid"
    if nf >= 6:
        return "hexapod"
    if nf == 4:
        return "quadruped"
    return "other"


class LeggedMorph:
    """Static morphology tokens for one compiled legged body (computed at the default pose, root frame)."""

    def __init__(self, model: mujoco.MjModel, binding, spec_hash: str):
        b, L = binding, binding.L
        d = mujoco.MjData(model)
        b.set_default(d, z=1.0)
        mujoco.mj_kinematics(model, d)
        root = b.root_bid
        rpos, rmat = d.xpos[root].copy(), d.xmat[root].reshape(3, 3).copy()
        to_root = lambda p: rmat.T @ (p - rpos)
        # actuated joints: policy actuators (output) + held actuators (arms/waist/head; held at default)
        acts = list(b.pol_act) + list(b.held_act)
        self.n_policy = len(b.pol_act)
        jids = [int(model.actuator_trnid[a, 0]) for a in acts]
        jbody = [int(model.jnt_bodyid[j]) for j in jids]
        # leg assemblies: foot ancestry
        def ancestors(bid):
            out = []
            while bid > 0:
                out.append(bid)
                bid = int(model.body_parentid[bid])
            return out
        foot_chain = [set(ancestors(f)) for f in b.foot_bids]
        nf = len(b.foot_bids)
        self.nf = nf
        foot_pos = [to_root(d.xpos[f]) for f in b.foot_bids]
        asm_of = []
        arm_sides = {}
        for k, (a, jb) in enumerate(zip(acts, jids)):
            owner = [i for i, ch in enumerate(foot_chain) if jbody[k] in ch]
            if owner and k < self.n_policy:
                asm_of.append(owner[0])
            else:
                # held upper-body joint: arm by lateral side of its anchor (waist/head -> body assembly)
                name = model.actuator(a).name.lower()
                if any(s in name for s in ("shoulder", "elbow", "wrist", "arm", "hand")):
                    side = "l" if ("left" in name or "_l" in name or name.startswith("l")) else "r"
                    arm_sides.setdefault(side, [])
                    asm_of.append(("arm", side))
                else:
                    asm_of.append("body")
        sides = sorted(arm_sides)
        self.M = nf + 1 + len(sides)
        body_idx = nf
        asm_index = []
        for x in asm_of:
            if isinstance(x, int):
                asm_index.append(x)
            elif x == "body":
                asm_index.append(body_idx)
            else:
                asm_index.append(nf + 1 + sides.index(x[1]))
        self.node_asm = np.array(asm_index, np.int64)
        self.N = len(acts)
        bk = _body_kind(L, nf)
        self.body_kind = bk
        nomh = b.nominal_height()
        feats = np.zeros((self.N, NODE_STATIC_DIM), np.float32)
        lo = np.concatenate([b.jlo, [model.jnt_range[j, 0] for j in jids[self.n_policy:]]])
        hi = np.concatenate([b.jhi, [model.jnt_range[j, 1] for j in jids[self.n_policy:]]])
        q0 = np.concatenate([b.q0, b.q0_held])
        self.q0_all = q0.astype(np.float32)
        depth = {}
        for k, j in enumerate(jids):
            anchor = to_root(d.xanchor[j])
            axis = rmat.T @ d.xaxis[j]
            a_i = self.node_asm[k]
            fp = foot_pos[a_i] if a_i < nf else np.zeros(3)
            depth.setdefault(a_i, 0)
            dep = depth[a_i]
            depth[a_i] += 1
            eff = float(model.actuator_forcerange[acts[k], 1]) if model.actuator_forcelimited[acts[k]] else 50.0
            feats[k] = np.concatenate([anchor / nomh, axis, fp / nomh, [dep / 6.0, float(k < self.n_policy)],
                                       [q0[k]], [lo[k], hi[k]], [b.action_scale], [min(eff, 500.0) / 100.0],
                                       np.eye(4)[BODY_KINDS.index(bk)], [nomh]]).astype(np.float32)
        self.node_static = feats
        asm = np.zeros((self.M, ASM_DIM), np.float32)
        for i in range(self.M):
            kind = "leg" if i < nf else ("body" if i == nf else "arm")
            members = np.nonzero(self.node_asm == i)[0]
            if kind == "leg":
                pos = foot_pos[i] / nomh
                side = 1.0 if pos[1] > 0 else -1.0
            elif kind == "body":
                pos, side = np.zeros(3), 0.0
            else:
                pos = (np.mean([to_root(d.xanchor[jids[m]]) for m in members], 0) / nomh) if len(members) else np.zeros(3)
                side = 1.0 if pos[1] > 0 else -1.0
            asm[i] = np.concatenate([np.eye(3)[ASM_KINDS.index(kind)], pos, [len(members) / 6.0, side, nomh,
                                                                              float(L["gait_period"])]])
        self.asm_static = asm
        self.asm_kind = [("leg" if i < nf else ("body" if i == nf else "arm")) for i in range(self.M)]
        self.spec_hash = spec_hash
        h = spec_hash16(spec_hash)
        self.handles = [f"asm:{h}:{k}{i}" for i, k in enumerate(self.asm_kind)]
        self.gait_period = float(L["gait_period"])
        self.action_scale = float(b.action_scale)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def active_event(runtime) -> int:
    """Public runtime: index into EVENTS of the first not-yet-completed event; len(EVENTS) when done/failed."""
    st = {e: i.status for e, i in runtime.instances.items()}
    for k, e in enumerate(EVENTS):
        if st.get(e) not in ("completed", "succeeded", "failed", "cancelled", "skipped"):
            return k
    return len(EVENTS)


def public_context(session, osc: float) -> np.ndarray:
    """PUBLIC system-i global features: IMU gyro/gravity, osc, task view (active event), waypoint estimates in the
    body frame from the declared localization sensor + detector tracks, public speed estimate."""
    b = session.binding
    imu = session._imu()
    g = quat_rotate_inv(imu["quat"], np.array([0, 0, -1.0]))
    loc = session.loc
    wps = []
    for ent in ("waypoint_a", "waypoint_b"):
        m, _ = session._track(ent)
        if m is None or loc is None:
            wps += [0.0, 0.0, 0.0, 0.0]
            continue
        dx, dy = m[0] - loc[0], m[1] - loc[1]
        c, s = math.cos(loc[2]), math.sin(loc[2])
        bx, by = c * dx + s * dy, -s * dx + c * dy
        dist = math.hypot(dx, dy)
        wps += [bx / 2.0, by / 2.0, min(dist, 5.0) / 2.0, 1.0]
    ev = active_event(session.runtime)
    evo = np.eye(len(EVENTS) + 1)[ev]
    spd = session.speed_est if np.isfinite(session.speed_est) else 0.0
    return np.concatenate([imu["gyro"] * 0.25, g, [math.sin(2 * math.pi * osc), math.cos(2 * math.pi * osc)], wps,
                           evo, [spd, float(np.isfinite(session.speed_est))]]).astype(np.float32)


def local_state(session, morph: LeggedMorph, touch_thresh: float = 1.0):
    """System-0 local inputs: joint encoders (all actuated joints), IMU gyro/gravity, per-foot touch."""
    b = session.binding
    d = session.data
    qadr = np.concatenate([b.pol_qadr, b.held_qadr]).astype(int)
    dadr = np.concatenate([b.pol_dadr, session.model.jnt_dofadr[session.model.actuator_trnid[b.held_act, 0]]
                           if len(b.held_act) else np.zeros(0, int)]).astype(int)
    q = d.qpos[qadr] - morph.q0_all
    qd = d.qvel[dadr] * 0.05
    quat, gyro = b.imu(d)
    g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
    touch = (session._touch() > touch_thresh).astype(np.float32)
    return q.astype(np.float32), qd.astype(np.float32), np.concatenate([gyro * 0.25, g]).astype(np.float32), touch


def spec_hash16(spec_hash: str) -> str:
    return hashlib.sha256(spec_hash.encode()).hexdigest()[:16]
