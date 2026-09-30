"""Grasp contact rig (python -m rrp.cli suite grasp-rig v2 pg2 [friction_scale]) (W7, D-108): the real gripper modules (pg2 / tf3) on a vertical carriage, a cube between the fingers.
1. close (full command -> the actuator's force limit, the worst case a policy can command), 2. lift 10 cm with a
min-jerk profile (peak acc ~2.3 m/s^2), 3. hold, then ramp the cube mass x1.08 every 0.25 s until it slips (>5 mm
relative to the palm). Reports penetration (max -dist pad<->cube) during the held lift, pad normal forces, and the slip
onset vs the Coulomb prediction m* = mu * sum(N) / g. Usage: grasp_contact_rig.py [v1|v2] [pg2|tf3] [friction_scale]"""
from __future__ import annotations

import json
import math
import sys
import types

import mujoco
import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.provenance import Source
from rrp.envs.base import ActionSpace, BodyInfo, EnvSpec, StepResult
from rrp.policies.base import Act, PolicyInfo, Requirements
from rrp.tasks.spec import Judgement, TaskSpec
from rrp.bodies.fixtures import gripper_module, three_finger_module
from rrp.bodies import grasp_contact as GC


def build(version: str, kind: str, half: float = 0.022, fscale: float = 1.0, yaw: float = 0.0):
    s = mujoco.MjSpec()
    s.option.timestep = 0.002
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10
    w = s.worldbody
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.01], pos=[0, 0, -0.01],
               friction=[1.0, 0.01, 0.001])
    car = w.add_body(name="carriage", pos=[0, 0, 0])
    car.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.1, 0.3],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=50.0)
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=2.0, contype=0, conaffinity=0)
    mod = gripper_module() if kind == "pg2" else three_finger_module()
    gs = mod.spec
    # tool frame: module z (palm -> fingertips) points DOWN; tcp site at z0 + 0.75/0.85 finger length
    L = 0.055
    tcp_off = 0.04 + L * (0.75 if kind == "pg2" else 0.85)
    fr = car.add_frame(pos=[0, 0, half + 0.004 + tcp_off], quat=[0, 1, 0, 0])
    s.attach(gs, prefix="r0_", frame=fr)
    s.add_actuator(name="lift_act", target="lift", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                   gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[20000] + [0] * 9,
                   biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -20000, -400] + [0] * 7)
    b = w.add_body(name="cube", pos=[0, 0, half + 0.0005], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    b.add_freejoint(name="cube_free")
    b.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3, density=500.0,
               friction=[1.2, 0.02, 0.002], condim=4)
    info = GC.apply(s, version)
    m = s.compile()
    if fscale != 1.0:
        m.geom_friction[:] *= fscale
    return m, info


_CONTROLLER = "grasp_rig/v1"


class _RigEnv:
    """Env (rrp.envs.base) on the bare rig model (the pattern of `tracker_validation._BenchEnv`): no scene, no task runtime,
    no session. One control tick is ONE physics step (500 Hz); the command sets the two rig actuators (`grip`, `lift`).
    Everything the rig measures is read from the model / data by `_RigMeter`."""

    ENV_ID = "mujoco/grasp_rig"

    def __init__(self, model, data, kind: str, grip: int, lift: int):
        self.model, self.d, self.grip, self.lift = model, data, grip, lift
        self.k = 0                                         # ticks done
        rate = 1.0 / float(model.opt.timestep)
        self.spec = EnvSpec(
            env_id=self.ENV_ID, backend="mujoco", task="grasp_rig",
            bodies=[BodyInfo(robot=0, family="gripper", key=f"rig_{kind}", robot_spec_hash="bare_rig_model")],
            control_hz=rate, capabilities=["proprio"],
            action_spaces=[ActionSpace(group="grip", kind="joint_position", width=1, rate_hz=rate),
                           ActionSpace(group="lift", kind="joint_position", width=1, rate_hz=rate)],
            provenance=dict(physics=dict(mujoco_timestep=float(model.opt.timestep), substeps=1)))

    def observe(self):
        return types.SimpleNamespace(sensor_time=float(self.d.time))

    def reset(self, seed=None):
        raise NotImplementedError("a rig env is built reset (one episode per env)")

    def close(self):
        pass

    def step(self, command):
        self.d.ctrl[self.grip] = command.groups["grip"][0]
        self.d.ctrl[self.lift] = command.groups["lift"][0]
        mujoco.mj_step(self.model, self.d)
        self.k += 1
        energy = float(np.sum(np.abs(self.d.actuator_force * self.d.actuator_velocity))) * float(self.model.opt.timestep)
        return StepResult(observation=self.observe(), qpos=None, time=float(self.d.time), energy_j=energy)


class _RigScript:
    """The rig's schedule as a (scripted, non-learned) policy: settle with the gripper open, close over 0.4 s (then hold
    the closing command), lift 10 cm on a min-jerk profile over `t_lift` (then hold), hold, then the mass ramp (the
    actuators keep their last targets). Each phase keeps the previous phase's last command of the other actuator."""

    info = PolicyInfo("grasp_rig_script", "scripted_teacher", _CONTROLLER,
                      Requirements(frozenset({"joint_position"}), observations=frozenset()))

    def __init__(self, dt, opened, closed, n_settle, n_close, n_lift, t_lift):
        self.dt, self.opened, self.closed, self.t_lift = dt, opened, closed, t_lift
        self.n_settle, self.n_close, self.n_lift = n_settle, n_close, n_lift

    def reset(self, spec, task, seeds, *, envs=None):
        self.state = {i: dict(h=0, grip=self.opened, lift=0.0) for i in range(len(seeds))}

    def act(self, obs):
        out = {}
        for i in obs:
            st = self.state[i]
            h = st["h"]
            if self.n_settle <= h < self.n_settle + self.n_close:
                a = min(1.0, (h - self.n_settle) * self.dt / 0.4)
                st["grip"] = self.opened + (self.closed - self.opened) * a
            elif self.n_settle + self.n_close <= h < self.n_settle + self.n_close + self.n_lift:
                u = min(1.0, (h - self.n_settle - self.n_close) * self.dt / self.t_lift)
                st["lift"] = 0.10 * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
            st["h"] += 1
            out[i] = Act(NativeCommand(controller_version=_CONTROLLER, source=Source.SCRIPTED_TEACHER,
                                       groups=dict(grip=[float(st["grip"])], lift=[float(st["lift"])])))
        return out


def _rig_task(max_steps: int) -> TaskSpec:
    """Ends when the cube slips (the meter's verdict) or when the ramp's tick budget is spent."""
    def judge(env, t, max_seconds):
        return Judgement(t >= max_seconds, "timeout", "timeout" if t >= max_seconds else None)
    return TaskSpec("grasp_rig", {_RigEnv.ENV_ID: {}}, math.inf, judge, max_steps=max_steps,
                    note="grasp contact rig: close, lift, hold, mass ramp to slip", failure_reasons=("slip", "timeout"))


class _RigMeter:
    """Every measurement of the rig, as a rollout hook: penetration and pad normal forces (per tick after the close and
    during the lift), the cube's slip relative to the palm, and the mass ramp (x1.08 at the start of every 0.25 s window,
    ended by the window in which the cube slips > 5 mm or drops)."""

    def __init__(self, model, pads, n_settle, n_close, n_lift, n_win):
        self.m, self.pads = model, pads
        self.n_close_end, self.n_lift_end, self.n_win = n_settle + n_close, n_settle + n_close + n_lift, n_win
        self.cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r0_palm")
        self.cg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

    def contacts(self, d):
        pen, N, touching = 0.0, 0.0, set()
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.cg in (c.geom1, c.geom2) and (c.geom1 in self.pads or c.geom2 in self.pads):
                pen = max(pen, -float(c.dist))
                mujoco.mj_contactForce(self.m, d, i, f)
                N += abs(f[0])
                touching.add(c.geom1 if c.geom2 == self.cg else c.geom2)
        return pen, N, len(touching)

    def rel(self, d):
        return d.xpos[self.cube] - d.xpos[self.palm]

    def on_reset(self, i, env, obs):
        self.env = env
        self.pens, self.scale, self.Nw, self.slip_mass, self.N_at_slip = [], 1.0, [], None, None

    def on_act(self, i, obs, act):
        """The mass ramp is an intervention on the rig's load: at the first tick of every window the cube (mass and
        inertia) is scaled by x1.08 more."""
        k = self.env.k
        if k >= self.n_lift_end and (k - self.n_lift_end) % self.n_win == 0:
            self.scale *= 1.08
            self.m.body_mass[self.cube] = self.m0 * self.scale
            self.m.body_inertia[self.cube] = self.I0 * self.scale

    def on_step(self, i, env, act, step):
        m, d, n = self.m, env.d, env.k
        if n == self.n_close_end:
            self.pen_close, _, self.nt = self.contacts(d)
            self.rel0 = self.rel(d).copy()
        elif self.n_close_end < n <= self.n_lift_end:
            self.pens.append(self.contacts(d)[0])
            if n == self.n_lift_end:
                self.lift_slip = float(np.linalg.norm(self.rel(d) - self.rel0))
                self.held = d.xpos[self.cube][2] > 0.08
                self.pen_hold, self.N_hold, _ = self.contacts(d)
                self.m0, self.I0 = float(m.body_mass[self.cube]), m.body_inertia[self.cube].copy()
                self.base, self.N_prev = self.rel(d).copy(), self.N_hold
        elif n > self.n_lift_end:
            self.Nw.append(self.contacts(d)[1])
            if (n - self.n_lift_end) % self.n_win == 0:
                if np.linalg.norm(self.rel(d) - self.base) > 0.005 or d.xpos[self.cube][2] < 0.02:
                    self.slip_mass, self.N_at_slip = self.m0 * self.scale, float(np.median(self.Nw))
                    return Judgement(True, "failure", "slip")
                self.N_prev, self.Nw = float(np.median(self.Nw)), []

    def on_end(self, i, env, ep):
        if ep.outcome == "crash":
            return {}
        return dict(rig=dict(pen_close=self.pen_close, n_touching=self.nt, pens=self.pens, lift_slip=self.lift_slip,
                             held=bool(self.held), pen_hold=self.pen_hold, N_hold=self.N_hold, m0=self.m0,
                             slip_mass=self.slip_mass, N_prev=self.N_prev))


def run(version="v2", kind="pg2", fscale=1.0, t_close=0.8, verbose=False, yaw=0.0, hooks=()):
    """One rig run (`hooks`: extra read-only rollout hooks after the meter, e.g. the replay recorder)."""
    from rrp.harness.rollout import rollout
    m, info = build(version, kind, fscale=fscale, yaw=yaw)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "r0_act_grip")
    lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_act")
    cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    pads = [g for g in range(m.ngeom) if GC.PAD_RE.search(m.geom(g).name or "")]
    closed = m.actuator_ctrlrange[grip][0] if kind == "pg2" else m.actuator_ctrlrange[grip][1]
    opened = m.actuator_ctrlrange[grip][1] if kind == "pg2" else -0.1
    mu = float(max(m.geom_friction[pads[0]][0], m.geom_friction[cg][0]) if m.geom_priority[pads[0]] == m.geom_priority[cg]
               else (m.geom_friction[pads[0]][0] if m.geom_priority[pads[0]] > m.geom_priority[cg] else m.geom_friction[cg][0]))
    # schedule: settle 0.2 s, close t_close, lift 10 cm over t_lift + hold 0.5 s, then up to 200 mass-ramp windows of 0.25 s
    t_lift = 0.5
    n_settle, n_close, n_lift, n_win = int(0.2 / dt), int(t_close / dt), int((t_lift + 0.5) / dt), int(0.25 / dt)
    script = _RigScript(dt, opened, closed, n_settle, n_close, n_lift, t_lift)
    meter = _RigMeter(m, pads, n_settle, n_close, n_lift, n_win)
    ep, = rollout(lambda sd: _RigEnv(m, d, kind, grip, lift), script, _rig_task(n_settle + n_close + n_lift + 200 * n_win),
                  [0], batch=1, hooks=[meter, *hooks])
    if ep.outcome == "crash":
        raise RuntimeError(f"grasp rig crashed ({ep.failure_reason}): {ep.metrics.get('note', '')}")
    r = ep.metrics["rig"]
    m0, slip_mass = r["m0"], r["slip_mass"]
    g = 9.81
    pred = None if slip_mass is None else mu * r["N_prev"] / g
    out = dict(version=info["version"], gripper=kind, friction_scale=fscale, cube_yaw_deg=round(math.degrees(yaw), 1), mu_pad_obj=round(mu, 4), cube_mass_kg=round(m0, 4),
               grip_force_limit=float(np.abs(m.actuator_forcerange[grip]).max()),
               n_fingers_touching=r["n_touching"], pen_after_close_mm=round(1000 * r["pen_close"], 2),
               pen_max_during_lift_mm=round(1000 * max(r["pens"]), 2), pen_hold_mm=round(1000 * r["pen_hold"], 2),
               normal_force_sum_hold_N=round(r["N_hold"], 2), lift_held=r["held"], lift_slip_mm=round(1000 * r["lift_slip"], 2),
               slip_mass_kg=None if slip_mass is None else round(slip_mass, 4),
               coulomb_pred_mass_kg=None if pred is None else round(pred, 4),
               slip_ratio=None if (pred is None or not pred) else round(slip_mass / pred, 3),
               slip_load_over_cube_weight=None if slip_mass is None else round(slip_mass / m0, 1))
    return out


def main(argv=None):
    """rrp suite grasp-rig [VERSION [GRIPPER [FRICTION_SCALE]]] -> one JSON line."""
    argv = list(sys.argv[1:] if argv is None else argv)
    v = argv[0] if len(argv) > 0 else "v2"
    k = argv[1] if len(argv) > 1 else "pg2"
    fs = float(argv[2]) if len(argv) > 2 else 1.0
    print(json.dumps(run(v, k, fs)))
