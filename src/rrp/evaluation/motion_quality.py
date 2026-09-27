"""Motion-quality metrics for every legged and arm evaluation row (W6). PRIVILEGED evaluation-only measurements
(simulator truth); nothing here feeds a policy, and recording never changes the rollout (read-only callbacks).

Primitives (unit-tested with known values, tests/unit/test_motion_quality.py):
  finite_diff(x, dt, n)          n-th forward difference / dt^n along time (axis 0)
  jerk_stats(q, dt)              RMS and peak |d3q/dt3| over time and joints
  chunk_boundary_steps(c, dt, b) largest per-tick change of commanded joint velocity at chunk-boundary ticks (+-1 tick)
                                 and anywhere (rad/s) — the D-102 chunk-seam roughness
  joint_limit_margin(q, lo, hi)  min over time/joints of min(q - lo, hi - q) / (hi - lo)  (0 = at a limit, 0.5 = centred)
  slip_ratio(slips, speeds)      mean loaded-foot contact-point slip / mean body speed (moving ticks)
  cost_of_transport(E, m, d)     E / (m g d)

Legged recorder (`LeggedMotionRecorder`, 50 Hz ticks + per physics substep):
  slip_ratio / slip_cp_mps       contact-point stance slip (LeggedBinding.stance; foot load > 2% body weight) over ticks
                                 where the true horizontal base speed > 0.1 m/s (waypoint episodes contain halts)
  cot                            sum |tau qdot| dt of the policy actuators (per substep) / (m g path length); path = sum of
                                 per-tick horizontal base displacements
  joint_jerk_rms / _peak         measured policy joints at 50 Hz (rad/s^3)
  peak_contact_force_bw          max over substeps and feet of the foot touch-sensor normal force / body weight
  joint_limit_margin_min         measured policy joints vs jnt_range
Arm recorder (`ArmMotionRecorder`, per control tick):
  joint_jerk_rms / _peak         measured arm joints (rad/s^3)
  chunk_vel_step_max / any_max   commanded arm joint-velocity step at chunk/packet boundaries (+-1 tick) / anywhere (rad/s)
  penetration_max_m              max depth of cube-robot contacts (-dist), as rrp.evaluation.teacher_quality
  joint_limit_margin_min         measured arm joints vs jnt_range
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

MQ_VERSION = "rrp.evaluation.motion_quality/v1"
G = 9.81


def finite_diff(x, dt: float, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    for _ in range(n):
        x = np.diff(x, axis=0) / dt
    return x


def jerk_stats(q, dt: float) -> dict:
    j = finite_diff(q, dt, 3)
    if j.size == 0:
        return dict(joint_jerk_rms=None, joint_jerk_peak=None)
    return dict(joint_jerk_rms=float(np.sqrt(np.mean(j ** 2))), joint_jerk_peak=float(np.max(np.abs(j))))


def chunk_boundary_steps(cmd, dt: float, boundary_ticks, window: int = 1) -> dict:
    """cmd[k] = commanded joints executed at tick k; boundary_ticks = ticks k at which a new chunk/packet starts.
    v[k] = (cmd[k+1] - cmd[k]) / dt; dv[k] = max_j |v[k+1] - v[k]| is the velocity step AT tick k+1. A boundary at
    tick b counts the steps at ticks b - window .. b + window."""
    c = np.asarray(cmd, float)
    if c.ndim != 2 or len(c) < 3:
        return dict(chunk_vel_step_max=None, vel_step_any_max=None, n_boundaries=len(list(boundary_ticks)))
    v = np.diff(c, axis=0) / dt
    dv = np.max(np.abs(np.diff(v, axis=0)), axis=1)        # index i -> tick i + 1
    at = [dv[t - 1] for b in boundary_ticks for t in range(b - window, b + window + 1) if 1 <= t <= len(dv)]
    return dict(chunk_vel_step_max=float(max(at)) if at else None, vel_step_any_max=float(dv.max()),
                n_boundaries=len(list(boundary_ticks)))


def joint_limit_margin(q, lo, hi) -> float | None:
    """Joints with hi <= lo (unlimited in MuJoCo) are skipped."""
    q = np.asarray(q, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    keep = hi > lo
    if q.size == 0 or not keep.any():
        return None
    q = q.reshape(-1, len(lo))[:, keep]
    lo, hi = lo[keep], hi[keep]
    return float(np.min(np.minimum(q - lo, hi - q) / (hi - lo)))


def slip_ratio(slips, speeds, min_speed: float = 0.02) -> float | None:
    if not len(slips) or not len(speeds):
        return None
    return float(np.mean(slips)) / max(float(np.mean(speeds)), min_speed)


def cost_of_transport(energy_J: float, mass_kg: float, dist_m: float, min_dist: float = 0.2) -> float | None:
    if dist_m < min_dist:
        return None
    return float(energy_J / (mass_kg * G * dist_m))


class LeggedMotionRecorder:
    """Read-only per-substep/per-tick recorder for a LeggedSession (attach via rrp.envs.perturb.install_legged)."""
    MOVING = 0.1

    def __init__(self, session):
        s = self.s = session
        b = self.b = s.binding
        m = s.model
        self.dt_sub = float(m.opt.timestep)
        self.mass = float(m.body_subtreemass[b.root_bid])
        self.w_load = 0.02 * self.mass * G
        self.touch_adr = []
        for n in s.robots[0].touch:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
            if sid >= 0:
                self.touch_adr.append(int(m.sensor_adr[sid]))
        self.touch_adr = np.array(self.touch_adr, int)
        self.on = False
        self._clear()

    def _clear(self):
        self.energy = 0.0
        self.path = 0.0
        self.q = []
        self.slips, self.speeds = [], []
        self.peak_touch = 0.0
        self.p_last = None
        self.ticks = 0

    def on_reset(self, done: bool):
        self.on = bool(done)
        self._clear()
        if done:
            d = self.s.data
            self.p_last = d.qpos[self.b.qa:self.b.qa + 2].copy()

    def on_substep(self, d, dt):
        if not self.on:
            return
        b = self.b
        self.energy += float(np.sum(np.abs(d.actuator_force[b.pol_act] * d.qvel[b.pol_dadr]))) * dt
        if len(self.touch_adr):
            self.peak_touch = max(self.peak_touch, float(np.max(d.sensordata[self.touch_adr])))

    def on_tick(self, d):
        if not self.on:
            return
        b = self.b
        self.ticks += 1
        p = d.qpos[b.qa:b.qa + 2].copy()
        if self.p_last is not None:
            self.path += float(np.linalg.norm(p - self.p_last))
        self.p_last = p
        self.q.append(d.qpos[b.pol_qadr].copy())
        spd = float(np.hypot(*d.qvel[b.da:b.da + 2]))
        if spd > self.MOVING:
            _fc, fn, slip, _bad = b.stance(d)
            self.slips += [float(x) for x in slip[fn > self.w_load]]
            self.speeds.append(spd)

    def summary(self) -> dict:
        b = self.b
        from rrp.envs.legged import TRACKER_HZ
        dt = 1.0 / TRACKER_HZ
        q = np.array(self.q) if self.q else np.zeros((0, b.n))
        out = dict(version=MQ_VERSION, family="legged", ticks=self.ticks, dt=dt,
                   slip_ratio=slip_ratio(self.slips, self.speeds),
                   slip_cp_mps=float(np.mean(self.slips)) if self.slips else None,
                   moving_ticks=len(self.speeds), path_m=self.path, energy_J=self.energy,
                   cot=cost_of_transport(self.energy, self.mass, self.path),
                   peak_contact_force_bw=(self.peak_touch / (self.mass * G)) if len(self.touch_adr) else None,
                   joint_limit_margin_min=joint_limit_margin(q, b.jlo, b.jhi))
        out.update(jerk_stats(q, dt))
        return out


class ArmMotionRecorder:
    """Per-control-tick recorder for a native arm Session (call tick() after each executed Session.step)."""

    def __init__(self, session):
        s = self.s = session
        m = s.model
        r = s.robots[0]
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in r.arm_joints])
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in r.arm_joints]
        self.lo, self.hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self.dt = float(s.dt)
        names = {l.name for l in r.spec.links}
        self.robot_body = np.array([m.body(b).name in names for b in range(m.nbody)])
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.q, self.cmd, self.boundaries = [], [], []
        self.pen = 0.0
        self.last_cmd = None

    def tick(self, cmd_groups: dict | None, boundary: bool):
        d, m = self.s.data, self.s.model
        k = len(self.q)
        if boundary:
            self.boundaries.append(k)
        if cmd_groups is not None and "arm" in cmd_groups:
            self.last_cmd = np.asarray(cmd_groups["arm"], float).copy()
        self.q.append(d.qpos[self.qadr].copy())
        self.cmd.append(self.last_cmd if self.last_cmd is not None else d.qpos[self.qadr].copy())
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == self.cube and self.robot_body[b2]) or (b2 == self.cube and self.robot_body[b1]):
                self.pen = max(self.pen, -float(c.dist))

    def summary(self) -> dict:
        q = np.array(self.q) if self.q else np.zeros((0, len(self.qadr)))
        out = dict(version=MQ_VERSION, family="arm", ticks=len(self.q), dt=self.dt, penetration_max_m=self.pen,
                   joint_limit_margin_min=joint_limit_margin(q, self.lo, self.hi))
        out.update(jerk_stats(q, self.dt))
        out.update(chunk_boundary_steps(np.array(self.cmd), self.dt, self.boundaries))
        return out
