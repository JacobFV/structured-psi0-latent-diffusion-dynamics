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

from rrp.envs.mujoco.legged_core import quat_rotate_inv

TICK_DT = 0.02
TICKS_PER_PACKET = 20                    # 0.4 s replan
KNOT_TIMES = (0.1, 0.3, 0.5, 0.7)
NODE_STATIC_DIM = 21
ASM_DIM = 10
ASM_KINDS = ("leg", "body", "arm")
BODY_KINDS = ("quadruped", "hexapod", "humanoid", "other")
EVENTS = ("walk_to_a", "walk_to_b", "halt")       # the waypoint_contact event names (slot meaning of that task; see TaskView)
EVENT_SLOTS = 3           # the context holds one-hot over EVENT_SLOTS events + "done" (index EVENT_SLOTS) for ANY task graph
TARGET_SLOTS = 3          # ... and TARGET_SLOTS task-entity estimates (4 numbers each): the widest registered graph (h_carry,
#                           h_place, h_gap_cart: object, support / cart, goal) names three. THE one constant to widen.
LEAD_TARGET_SLOTS = 2     # slots 0 .. 1 sit inside the original 22-wide layout (cols 8:16); slots 2 .. TARGET_SLOTS-1 are appended
#                           after the speed estimate, so every index of the original layout (ctx[8:16], ctx[16:20], ctx[20:22]) holds
GLOBAL_DIM = 22 + 4 * (TARGET_SLOTS - LEAD_TARGET_SLOTS)
MAX_N = 32                              # padded actuated joints (policy + held; g1 has 29)
MAX_M = 11                              # padded assemblies (8 legs + body + 2 arms)
H = 40                    # demonstrated ticks seen by E (0.8 s)
MAX_J = 27                # realizer phases 0..0.54 s
KNOT_TICKS = [int(round(k / TICK_DT)) for k in KNOT_TIMES]


def _body_kind(L, nf):
    k = L.get("kind", "other")
    if k in ("humanoid", "biped"):
        return "humanoid"
    if nf >= 6:
        return "hexapod"
    if nf == 4:
        return "quadruped"
    return "other"


def _kinematic_structure(model, jids, node_asm, M: int, nf: int, root_bid: int):
    """(node_parent [N], asm_trunk [M]) of the real kinematic tree (HD1): `node_parent[k]` = index (among the actuated joints)
    of the nearest actuated joint above joint k (the previous actuated joint on the same body, else the last one on the
    nearest ancestor body that has one), -1 when nothing actuated lies between it and the world; `asm_trunk[m]` = MuJoCo body
    id of the trunk link assembly m hangs from: the parent body of the topmost body of its joints (legs of a biped: the
    pelvis, arms of a humanoid: the torso link, all legs of a quadruped / hexapod: the trunk); the body assembly (waist,
    head, index nf) is the root body and an empty assembly is -1. Two limbs share a trunk link iff their `asm_trunk` are equal."""
    jbody = [int(model.jnt_bodyid[j]) for j in jids]
    on_body: dict[int, list[int]] = {}
    for k in sorted(range(len(jids)), key=lambda k: (jbody[k], jids[k])):
        on_body.setdefault(jbody[k], []).append(k)
    parent = np.full(len(jids), -1, np.int64)
    for k in range(len(jids)):
        peers = on_body[jbody[k]]
        i = peers.index(k)
        if i > 0:
            parent[k] = peers[i - 1]
            continue
        bid = int(model.body_parentid[jbody[k]])
        while bid > 0 and bid not in on_body:
            bid = int(model.body_parentid[bid])
        if bid > 0:
            parent[k] = on_body[bid][-1]

    def depth(bid):
        n = 0
        while bid > 0:
            bid, n = int(model.body_parentid[bid]), n + 1
        return n

    trunk = np.full(M, -1, np.int64)
    for m in range(M):
        mem = [k for k in range(len(jids)) if node_asm[k] == m]
        if mem:
            top = min((jbody[k] for k in mem), key=depth)
            trunk[m] = int(model.body_parentid[top])
    trunk[nf] = root_bid
    return parent, trunk


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
        self.node_parent, self.asm_trunk = _kinematic_structure(model, jids, self.node_asm, self.M, nf, b.root_bid)
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


ENTITY_ROLES = ("target", "reference", "patient", "support")      # the roles whose entity the context locates (not the actor)


class TaskView:
    """What a task graph declares to the context (D-146 H4, HD1): its event ids in graph order (slot k = event k; `done` is
    always slot EVENT_SLOTS) and the entities its events walk to / act on (first TARGET_SLOTS distinct ones, in event
    order and, within an event, role order in the graph: target, reference, patient and support roles; the actor is the
    body itself); a missing slot is zeros with valid flag 0. Read from the graph the task registry names
    (`TaskSpec.graph`). waypoint_contact has only target / reference roles: its (waypoint_a, waypoint_b) are unchanged."""

    def __init__(self, graph: dict):
        self.task_id = graph["task_id"]
        self.events = tuple(e["id"] for e in graph["events"])
        seen: list[str] = []
        for e in graph["events"]:
            for r in e["roles"]:
                ent = r["binding"].get("entity", {}).get("id") if r["role"] in ENTITY_ROLES else None
                if ent and ent not in seen:
                    seen.append(ent)
        self.targets = tuple(seen)
        if len(self.events) > EVENT_SLOTS or len(self.targets) > TARGET_SLOTS:
            raise ValueError(f"task graph {self.task_id!r} has {len(self.events)} events / {len(self.targets)} entities; "
                             f"the legged context holds {EVENT_SLOTS} / {TARGET_SLOTS} (GLOBAL_DIM = {GLOBAL_DIM})")

    def as_dict(self) -> dict:
        return dict(task_id=self.task_id, events=list(self.events), targets=list(self.targets),
                    event_slots=EVENT_SLOTS, target_slots=TARGET_SLOTS)


_VIEWS: dict[str, TaskView] = {}


def task_view(graph: dict) -> TaskView:
    key = f"{graph['task_id']}@{graph['graph_version']}"
    if key not in _VIEWS:
        _VIEWS[key] = TaskView(graph)
    return _VIEWS[key]


def task_view_of(task: str) -> TaskView:
    """The view of a registered task: the graph `TaskSpec.graph` names."""
    from rrp.envs.mujoco.scenario import load_task
    from rrp.tasks.spec import get_task
    g = get_task(task).graph
    if g is None:
        raise ValueError(f"task {task!r} declares no graph; the legged context needs one")
    return task_view(load_task(g))


def active_event(runtime) -> int:
    """Public runtime: index (graph order) of the first not-yet-completed event; EVENT_SLOTS when done/failed."""
    for k, (e, i) in enumerate(runtime.instances.items()):
        if i.status not in ("completed", "succeeded", "failed", "cancelled", "skipped"):
            return k
    return EVENT_SLOTS


def target_slot_cols(k: int) -> slice:
    """Columns of `public_context` holding entity slot k (bx/2, by/2, dist/2, valid): slots 0..LEAD_TARGET_SLOTS-1 sit at 8:16
    (the pre-widening layout), the rest follow the legacy 22 columns."""
    a = 8 + 4 * k if k < LEAD_TARGET_SLOTS else 22 + 4 * (k - LEAD_TARGET_SLOTS)
    return slice(a, a + 4)


def public_context(session, osc: float) -> np.ndarray:
    """PUBLIC system-i global features: IMU gyro/gravity, osc, task view (active event), target-entity estimates in the
    body frame from the declared localization sensor + detector tracks, public speed estimate. The event and target
    slots come from the session's task graph (`TaskView`); for waypoint_contact they are (walk_to_a, walk_to_b, halt)
    and (waypoint_a, waypoint_b), exactly as before. Layout: [gyro 3, gravity 3, osc 2, entity slots 0..1 (4 each), event
    one-hot (EVENT_SLOTS + 1), speed est 2, entity slots 2.. (4 each)] = GLOBAL_DIM."""
    view = task_view(session.scenario.task)
    imu = session._imu()
    g = quat_rotate_inv(imu["quat"], np.array([0, 0, -1.0]))
    loc = session.loc
    wps = []
    for k in range(TARGET_SLOTS):
        m, _ = session._track(view.targets[k]) if k < len(view.targets) else (None, None)
        if m is None or loc is None:
            wps += [0.0, 0.0, 0.0, 0.0]
            continue
        dx, dy = m[0] - loc[0], m[1] - loc[1]
        c, s = math.cos(loc[2]), math.sin(loc[2])
        bx, by = c * dx + s * dy, -s * dx + c * dy
        dist = math.hypot(dx, dy)
        wps += [bx / 2.0, by / 2.0, min(dist, 5.0) / 2.0, 1.0]
    evo = np.eye(EVENT_SLOTS + 1)[active_event(session.runtime)]
    spd = session.speed_est if np.isfinite(session.speed_est) else 0.0
    lead = 4 * LEAD_TARGET_SLOTS
    return np.concatenate([imu["gyro"] * 0.25, g, [math.sin(2 * math.pi * osc), math.cos(2 * math.pi * osc)], wps[:lead],
                           evo, [spd, float(np.isfinite(session.speed_est))], wps[lead:]]).astype(np.float32)


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
