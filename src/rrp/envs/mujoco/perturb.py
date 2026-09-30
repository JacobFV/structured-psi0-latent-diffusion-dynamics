"""Physics perturbations for robustness sweeps (W6). Evaluation-time only: nothing here enters an observation.

A `PhysicsPerturbation` describes one departure from the training physics. It is applied in two places:

* model level (`apply_model`, before the Session is constructed, so the robot spec / morphology features the policy
  sees stay NOMINAL — the policy is not told about the shift):
    friction_scale        x all geom friction coefficients (sliding, torsional, rolling). Under contact_v2 the floor
                          has contact priority, so robot-floor mu = 0.9 x scale.
    mass_scale            x mass and inertia of every ROBOT body (whole-body mass error)
    com_offset_m          (dx, dy, dz) added to the CoM (body_ipos) of the "com body": legged = root/trunk,
                          arm = the last arm link (payload offset at the wrist)
    kp_scale / kd_scale   x the position-servo gains of every robot position actuator (affine bias, kp = -biasprm[1])
                          (kd_scale None = kp_scale)
    object_mass_scale     (arm) x mass/inertia of the task object `cube`
    object_friction_scale (arm) x friction of the cube AND the gripper finger geoms (grasp contact mu = scale x nominal,
                          since MuJoCo uses the max of the pair)
* step level (hooks installed on the session):
    latency_ms            legged: the existing actuator mode `v1lat` (rrp.bodies.actuator.ActuatorModel: ideal joints +
                          SOURCED torque/speed limits + fixed actuation latency via the cross-tick pending-target queue);
                          arm: `ctrl_delay_v0`, a FIFO delay of the robot's ctrl by round(latency / dt) physics substeps
                          (the arm has no actuator model). None = ideal actuators (the training condition).
    push_impulse_Ns       horizontal impulse J applied as a constant force J / push_duration_s on the "push body"
                          (legged root, arm last link) during [push_time_s, push_time_s + duration); direction =
                          push_dir_rad (world frame) or, if None, drawn from the episode seed only (paired across levels).
    terrain_amp_m         (legged) heightfield bumps of 0..amp m (rrp.bodies.legged.legged_world terrain flag; the
                          terrain pattern is drawn from the episode seed only).

`PhysicsPerturbation()` (all defaults) is the nominal condition and leaves every code path byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import mujoco
import numpy as np

PERTURB_VERSION = "rrp.envs.perturb/v1"


@dataclass(frozen=True)
class PhysicsPerturbation:
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    com_offset_m: tuple = (0.0, 0.0, 0.0)
    kp_scale: float = 1.0
    kd_scale: float | None = None
    latency_ms: float | None = None
    push_impulse_Ns: float = 0.0
    push_time_s: float = 4.0
    push_duration_s: float = 0.1
    push_dir_rad: float | None = None
    terrain_amp_m: float = 0.0
    object_mass_scale: float = 1.0
    object_friction_scale: float = 1.0
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def model_changes(self) -> bool:
        return (self.friction_scale != 1.0 or self.mass_scale != 1.0 or any(c != 0 for c in self.com_offset_m)
                or self.kp_scale != 1.0 or (self.kd_scale is not None and self.kd_scale != 1.0)
                or self.object_mass_scale != 1.0 or self.object_friction_scale != 1.0)

    @property
    def step_hooks(self) -> bool:
        return self.latency_ms is not None or self.push_impulse_Ns > 0

    @property
    def is_nominal(self) -> bool:
        return not (self.model_changes or self.step_hooks or self.terrain_amp_m > 0)

    def terrain(self, seed: int) -> dict | None:
        return dict(amp_m=float(self.terrain_amp_m), seed=int(seed)) if self.terrain_amp_m > 0 else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d["com_offset_m"] = list(self.com_offset_m)
        d["version"] = PERTURB_VERSION
        return d

    @staticmethod
    def from_dict(d: dict) -> "PhysicsPerturbation":
        d = {k: v for k, v in d.items() if k != "version"}
        if "com_offset_m" in d:
            d["com_offset_m"] = tuple(d["com_offset_m"])
        return PhysicsPerturbation(**d)


def push_direction(seed: int, pert: PhysicsPerturbation) -> float:
    if pert.push_dir_rad is not None:
        return float(pert.push_dir_rad)
    return float(np.random.default_rng([int(seed), 4242]).uniform(-math.pi, math.pi))


def _scale_body(m: mujoco.MjModel, bid: int, s: float):
    m.body_mass[bid] *= s
    m.body_inertia[bid] *= s


def position_actuators(m: mujoco.MjModel, act_ids) -> list[int]:
    return [int(a) for a in act_ids if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            and m.actuator_biasprm[a, 1] < 0]


def apply_model(m: mujoco.MjModel, pert: PhysicsPerturbation, *, robot_bodies, com_body: int, act_ids,
                object_bodies=(), object_contact_geoms=()) -> dict:
    """Mutate a compiled model in place. Returns a record of what was changed (for the eval row).
    robot_bodies: body ids of the robot; com_body: body receiving the CoM offset; act_ids: robot actuator ids;
    object_bodies: task-object body ids (arm cube); object_contact_geoms: geoms whose friction scales with the object
    friction factor besides the object's own (arm finger geoms)."""
    rec = {}
    if not pert.model_changes:
        return rec
    if pert.friction_scale != 1.0:
        m.geom_friction[:] *= pert.friction_scale
        rec["friction_scale"] = pert.friction_scale
    if pert.mass_scale != 1.0:
        for b in robot_bodies:
            _scale_body(m, int(b), pert.mass_scale)
        rec["mass_scale"] = pert.mass_scale
    if any(c != 0 for c in pert.com_offset_m):
        m.body_ipos[com_body] += np.asarray(pert.com_offset_m, float)
        rec["com_offset_m"] = list(pert.com_offset_m)
        rec["com_body"] = m.body(com_body).name
    kd_s = pert.kp_scale if pert.kd_scale is None else pert.kd_scale
    if pert.kp_scale != 1.0 or kd_s != 1.0:
        pa = position_actuators(m, act_ids)
        for a in pa:
            m.actuator_gainprm[a, 0] *= pert.kp_scale
            m.actuator_biasprm[a, 1] *= pert.kp_scale
            m.actuator_biasprm[a, 2] *= kd_s
        rec.update(kp_scale=pert.kp_scale, kd_scale=kd_s, n_servos=len(pa))
    if pert.object_mass_scale != 1.0:
        for b in object_bodies:
            _scale_body(m, int(b), pert.object_mass_scale)
        rec["object_mass_scale"] = pert.object_mass_scale
    if pert.object_friction_scale != 1.0:
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in set(int(b) for b in object_bodies)]
        gs += [int(g) for g in object_contact_geoms]
        for g in sorted(set(gs)):
            m.geom_friction[g] *= pert.object_friction_scale
        rec["object_friction_scale"] = pert.object_friction_scale
        rec["object_friction_geoms"] = len(set(gs))
    if pert.mass_scale != 1.0 or pert.object_mass_scale != 1.0 or any(c != 0 for c in pert.com_offset_m):
        mujoco.mj_setConst(m, mujoco.MjData(m))       # subtree masses etc. derived from body masses
    return rec


class PushHook:
    """Constant horizontal force J/duration on one body during [t0, t0 + duration) (world frame)."""

    def __init__(self, bid: int, pert: PhysicsPerturbation, seed: int):
        self.bid = int(bid)
        self.t0, self.dur = float(pert.push_time_s), float(pert.push_duration_s)
        self.dir = push_direction(seed, pert)
        F = float(pert.push_impulse_Ns) / self.dur
        self.f = np.array([F * math.cos(self.dir), F * math.sin(self.dir), 0.0])
        self.on_steps = set()          # distinct substep times with the force on (robust to repeated calls)
        self.dt = None

    def __call__(self, d: mujoco.MjData, dt: float):
        on = self.t0 <= d.time < self.t0 + self.dur - 1e-9
        d.xfrc_applied[self.bid, :3] = self.f if on else 0.0
        if on:
            self.on_steps.add(round(float(d.time) / dt))
            self.dt = dt

    def record(self) -> dict:
        s = len(self.on_steps) * (self.dt or 0.0)
        return dict(push_dir_rad=self.dir, push_force_N=float(np.linalg.norm(self.f)), push_applied_s=s,
                    push_impulse_applied_Ns=float(np.linalg.norm(self.f)) * s)


class CtrlDelay:
    """`ctrl_delay_v0` (arm): the robot's ctrl reaches the actuators n substeps late (FIFO)."""
    version = "ctrl_delay_v0"

    def __init__(self, act_ids, n: int, ctrl0: np.ndarray):
        self.ids = np.asarray(act_ids, int)
        self.n = int(n)
        self.buf = [np.array(ctrl0, float) for _ in range(self.n)]

    def __call__(self, d: mujoco.MjData):
        if self.n <= 0:
            return
        self.buf.append(d.ctrl[self.ids].copy())
        d.ctrl[self.ids] = self.buf.pop(0)


# ------------------------------------------------------------------ legged sessions
def install_legged(session, pert: PhysicsPerturbation, seed: int, *, on_substep=None, on_tick=None, on_reset=None):
    """Replace LeggedSession._tracker_tick (instance attribute) by a version with the step-level hooks. With no
    hooks the replacement performs exactly the original operations (same order), plus the read-only callbacks
    on_substep(d, dt) / on_tick(d) / on_reset() used by the motion-quality recorder. Returns a state dict."""
    s = session
    b = s.binding
    st = dict(actuator=None, push=None, first=True)
    # D-126 #14: a session built with a non-ideal actuator mode keeps ITS actuator model (reset by the session itself) unless
    # the perturbation sets its own latency (which then wins, as before). Ideal sessions: unchanged.
    own = pert.latency_ms is None and getattr(s, "actuator_model", None) is not None
    if pert.latency_ms is not None:
        from rrp.bodies.actuator import ActuatorModel
        st["actuator"] = ActuatorModel(s.model, b, 1, None, name=s.robots[0].meta["name"], randomize=False,
                                       latency_ms=float(pert.latency_ms), mode="v1lat")
    elif own:
        st["actuator"] = s.actuator_model
        st["first"] = False
    if pert.push_impulse_Ns > 0:
        st["push"] = PushHook(b.root_bid, pert, seed)
    act, push = st["actuator"], st["push"]
    dt = float(s.model.opt.timestep)
    from rrp.envs.mujoco.legged import TRACKER_HZ

    def _tracker_tick(cmd):
        d = s.data
        tgt = s.tracker.act(d, cmd)
        if act is not None:
            if st["first"]:
                act.reset(0, tgt)
                st["first"] = False
            act.command(0, tgt)
        else:
            d.ctrl[b.pol_act] = tgt
        if len(b.held_act):
            d.ctrl[b.held_act] = b.q0_held
        n = max(1, int(round(1.0 / (TRACKER_HZ * s.model.opt.timestep))))
        for _ in range(n):
            if act is not None:
                d.ctrl[b.pol_act] = act.substep_ctrl(0, d)
            if push is not None:
                push(d, dt)
            mujoco.mj_step(s.model, d)
            if on_substep is not None:
                on_substep(d, dt)
        if on_tick is not None:
            on_tick(d)

    orig_reset = s.reset

    def reset(*a, **k):
        st["first"] = not own
        if on_reset is not None:
            on_reset(False)
        obs = orig_reset(*a, **k)
        if on_reset is not None:
            on_reset(True)
        return obs

    s._tracker_tick = _tracker_tick
    s.reset = reset
    return st


# ------------------------------------------------------------------ arm sessions
def arm_parts(m: mujoco.MjModel, robot) -> dict:
    """Robot body ids, the last arm link (com/push body), finger geoms and actuator ids of a native arm session robot."""
    names = {l.name for l in robot.spec.links}
    bodies = [b for b in range(m.nbody) if m.body(b).name in names]
    last_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, robot.arm_joints[-1])
    last_link = int(m.jnt_bodyid[last_j])
    asm = next(a for a in robot.spec.assemblies if a.kind in ("gripper", "hand"))
    hand = {l.name for l in robot.spec.links if l.address in asm.members}
    fingers = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in hand and "finger" in m.body(m.geom_bodyid[g]).name]
    acts = [int(a) for ids in robot.controller.act_ids.values() for a in np.atleast_1d(ids)]
    return dict(bodies=bodies, last_link=last_link, finger_geoms=fingers, act_ids=acts)


def install_arm(session, pert: PhysicsPerturbation, seed: int) -> dict:
    """Wrap each robot controller's apply_substep (called right before every mj_step of Session.step) with the
    ctrl delay and push hooks. Nothing is installed for the nominal condition."""
    s = session
    st = dict(delay=None, push=None)
    if not pert.step_hooks:
        return st
    r = s.robots[0]
    parts = arm_parts(s.model, r)
    dt = float(s.model.opt.timestep)
    if pert.latency_ms is not None:
        n = int(round(float(pert.latency_ms) / 1000.0 / dt))
        st["delay"] = CtrlDelay(parts["act_ids"], n, s.data.ctrl[parts["act_ids"]].copy())
    if pert.push_impulse_Ns > 0:
        st["push"] = PushHook(parts["last_link"], pert, seed)
    delay, push = st["delay"], st["push"]
    orig = r.controller.apply_substep

    def apply_substep(data, alpha):
        orig(data, alpha)
        if delay is not None:
            delay(data)
        if push is not None:
            push(data, dt)

    r.controller.apply_substep = apply_substep
    return st
