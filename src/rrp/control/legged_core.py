"""Legged tracker core shared by training (vectorised envs) and deployment (LeggedSession).

Policy-facing command: base-velocity group [vx (m/s, body frame), vy (m/s), wz (rad/s)].
Tracker output: joint position targets for the body's `policy_actuators` (held actuators are
servoed to the default pose).

PUBLIC tracker observation (deployable; identical in training and deployment):
    gyro (IMU, body frame) * 0.25, gravity direction in IMU frame (from IMU orientation, roll/pitch
    only), command * scales, q - q_default, qdot * 0.05, previous action, gait clock sin/cos.
PRIVILEGED critic extras (training only, never fed to the actor):
    base linear velocity (body frame), base height, foot contact flags, friction scale, push flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

CMD_SCALE = np.array([2.0, 2.0, 0.25])


def quat_rotate_inv(q, v):
    """Rotate world vector v into the frame of unit quaternion q (w,x,y,z)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LeggedBinding:
    """Resolves a legged body's indices inside a compiled model (robot namespaced by prefix)."""

    def __init__(self, model: mujoco.MjModel, meta: dict, prefix: str = "r0_"):
        L = meta["legged"]
        self.model, self.meta, self.L, self.prefix = model, meta, L, prefix
        nid = lambda t, n: mujoco.mj_name2id(model, t, prefix + n)
        O = mujoco.mjtObj
        self.root_bid = nid(O.mjOBJ_BODY, L["root_body"])
        fj = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
              and model.jnt_bodyid[j] == self.root_bid]
        assert len(fj) == 1, "root free joint not found"
        self.qa = int(model.jnt_qposadr[fj[0]])
        self.da = int(model.jnt_dofadr[fj[0]])
        self.pol_act = np.array([nid(O.mjOBJ_ACTUATOR, a) for a in L["policy_actuators"]])
        self.held_act = np.array([nid(O.mjOBJ_ACTUATOR, a) for a in L["held_actuators"]], dtype=int)
        assert (self.pol_act >= 0).all() and (self.held_act >= 0).all()
        pj = model.actuator_trnid[self.pol_act, 0]
        self.pol_qadr = model.jnt_qposadr[pj]
        self.pol_dadr = model.jnt_dofadr[pj]
        hj = model.actuator_trnid[self.held_act, 0] if len(self.held_act) else np.zeros(0, int)
        self.held_qadr = model.jnt_qposadr[hj] if len(hj) else np.zeros(0, int)
        jn = lambda j: model.joint(j).name[len(prefix):]
        self.q0 = np.array([L["default_pose"][jn(j)] for j in pj])
        self.q0_held = np.array([L["default_pose"][jn(j)] for j in hj]) if len(hj) else np.zeros(0)
        self.lo = model.actuator_ctrlrange[self.pol_act, 0].copy()
        self.hi = model.actuator_ctrlrange[self.pol_act, 1].copy()
        self.effort = model.actuator_forcerange[self.pol_act, 1].copy()
        self.jlo = model.jnt_range[pj, 0].copy()
        self.jhi = model.jnt_range[pj, 1].copy()
        self.n = len(self.pol_act)
        self.foot_bids = [nid(O.mjOBJ_BODY, f) for f in L["foot_bodies"]]
        self.nf = len(self.foot_bids)
        self.floor = mujoco.mj_name2id(model, O.mjOBJ_GEOM, "floor")
        self.body_is_foot = np.full(model.nbody, -1)
        for i, b in enumerate(self.foot_bids):
            self.body_is_foot[b] = i
        # robot bodies (for "non-foot body touches floor" terminations)
        self.robot_bodies = np.array([b for b in range(model.nbody) if model.body(b).name.startswith(prefix)])
        self.is_robot_body = np.zeros(model.nbody, bool)
        self.is_robot_body[self.robot_bodies] = True
        # bodies allowed to touch the ground: feet and their descendants (toes etc.)
        self.allowed_ground = np.zeros(model.nbody, bool)
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p in self.foot_bids:
                    self.allowed_ground[b] = True
                    break
                p = model.body_parentid[p]
        self.kind = L["kind"]
        self.biped = self.kind in ("humanoid", "biped")
        self.action_scale = float(L["action_scale"])
        self.period = float(L["gait_period"])
        self.cmd_ranges = L["command_ranges"]
        self.min_h = float(L["min_height_frac"]) * self.nominal_height()
        self.tilt_limit = float(L["tilt_limit"])
        imu = L["imu"]
        self.imu_quat = self._sensor_adr(prefix + imu["quat"])
        self.imu_gyro = self._sensor_adr(prefix + imu["gyro"])

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else None

    def nominal_height(self) -> float:
        L = self.L
        if L.get("nominal_height") is not None:
            return float(L["nominal_height"])
        d = mujoco.MjData(self.model)
        self.set_default(d, z=1.0)
        mujoco.mj_kinematics(self.model, d)
        fb = set(self.foot_bids)
        low = min(d.geom_xpos[g][2] - self.model.geom_rbound[g] for g in range(self.model.ngeom)
                  if self.model.geom_bodyid[g] in fb)
        L["nominal_height"] = float(1.0 - low)
        return L["nominal_height"]

    obs_dim = property(lambda self: 3 + 3 + 3 + 3 * self.n + 2)
    priv_dim = property(lambda self: 3 + 1 + self.nf + 2)

    def set_default(self, d: mujoco.MjData, z: float | None = None, xy=(0.0, 0.0), yaw: float = 0.0,
                    noise: float = 0.0, rng=None):
        d.qpos[self.qa:self.qa + 2] = xy
        d.qpos[self.qa + 2] = self.nominal_height() + 0.005 if z is None else z
        d.qpos[self.qa + 3:self.qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        q = self.q0 + (rng.uniform(-noise, noise, self.n) if (noise and rng is not None) else 0)
        d.qpos[self.pol_qadr] = np.clip(q, self.jlo, self.jhi)
        if len(self.held_qadr):
            d.qpos[self.held_qadr] = self.q0_held
        d.qvel[:] = 0
        d.ctrl[self.pol_act] = np.clip(self.q0, self.lo, self.hi)
        if len(self.held_act):
            d.ctrl[self.held_act] = self.q0_held

    # ---------------------------------------------------------------- public observation
    def imu(self, d: mujoco.MjData):
        """(quat, gyro) as the IMU reports them. IMU frame == root frame (identity site), so these
        equal the free-joint quaternion and local angular velocity (asserted in tests)."""
        return d.qpos[self.qa + 3:self.qa + 7], d.qvel[self.da + 3:self.da + 6]

    def public_obs(self, d: mujoco.MjData, cmd, last_action, phase) -> np.ndarray:
        quat, gyro = self.imu(d)
        g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
        return np.concatenate([gyro * 0.25, g, np.asarray(cmd) * CMD_SCALE, d.qpos[self.pol_qadr] - self.q0,
                               d.qvel[self.pol_dadr] * 0.05, last_action,
                               [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]]).astype(np.float32)

    def targets(self, action) -> np.ndarray:
        return np.clip(self.q0 + self.action_scale * np.asarray(action), self.lo, self.hi)

    # ---------------------------------------------------------------- privileged quantities
    def base_lin_vel_body(self, d):
        return quat_rotate_inv(d.qpos[self.qa + 3:self.qa + 7], d.qvel[self.da:self.da + 3])

    def contacts(self, d) -> tuple[np.ndarray, bool]:
        """(foot contact flags, bad_contact: a non-foot robot body touches the floor)."""
        fc = np.zeros(self.nf, bool)
        bad = False
        m = self.model
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor:
                b = m.geom_bodyid[c.geom2]
            elif c.geom2 == self.floor:
                b = m.geom_bodyid[c.geom1]
            else:
                continue
            f = self.body_is_foot[b]
            if f >= 0:
                fc[f] = True
            elif self.allowed_ground[b]:
                p = b
                while self.body_is_foot[p] < 0:
                    p = m.body_parentid[p]
                fc[self.body_is_foot[p]] = True
            elif self.is_robot_body[b]:
                bad = True
        return fc, bad

    def tilt(self, d) -> float:
        g = quat_rotate_inv(d.qpos[self.qa + 3:self.qa + 7], np.array([0, 0, -1.0]))
        return float(math.acos(max(-1.0, min(1.0, -g[2]))))


@dataclass
class RewardCfg:
    track_lin: float = 1.5
    track_ang: float = 0.75
    lin_z: float = -2.0
    ang_xy: float = -0.05
    orient: float = -2.0
    torque: float = -0.02
    action_rate: float = -0.02
    limits: float = -2.0
    air_time: float = 1.0
    stand_still: float = -0.3
    alive: float = 0.2
    contact_phase: float = 0.0
    height: float = 0.0
    feet_slip: float = -0.05
    termination: float = -5.0
    stand_contact: float = 0.5     # zero command: all feet down (stance transition)
    sigma: float = 0.25

    @staticmethod
    def for_kind(kind: str) -> "RewardCfg":
        if kind in ("humanoid", "biped"):
            # v2 (after v1 converged to a stable non-walking stander): sharper tracking kernel, more
            # tracking weight, less alive bonus so standing still under a walk command is not optimal
            return RewardCfg(orient=-5.0, alive=0.3, contact_phase=0.4, height=-20.0, air_time=1.0,
                             stand_still=-0.5, termination=-10.0, sigma=0.1, track_lin=2.5, track_ang=1.0)
        return RewardCfg()


class LeggedEnv:
    """N independent MjData sharing one model; 50 Hz tracker rate, PD on physics substeps."""

    def __init__(self, module_factory, n_envs: int, seed: int, *, control_dt: float = 0.02,
                 episode_s: float = 20.0, friction_scale: float = 1.0, push: bool = True, obs_noise: float = 1.0):
        from rrp.morphology.legged import standalone_model
        mod = module_factory()
        self.model, _, self.meta = standalone_model(mod)
        if friction_scale != 1.0:
            self.model.geom_friction[:, 0] *= friction_scale
        self.friction_scale = friction_scale
        self.b = LeggedBinding(self.model, self.meta)
        self.cfg = RewardCfg.for_kind(self.b.kind)
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.data = [mujoco.MjData(self.model) for _ in range(n_envs)]
        self.dt = control_dt
        self.substeps = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.max_steps = int(episode_s / control_dt)
        self.push = push
        self.obs_noise = obs_noise
        nA, nf = self.b.n, self.b.nf
        self.cmd = np.zeros((n_envs, 3))
        self.last_a = np.zeros((n_envs, nA))
        self.phase = np.zeros(n_envs)
        self.t = np.zeros(n_envs, int)
        self.air = np.zeros((n_envs, nf))
        self.cmd_timer = np.zeros(n_envs, int)
        self.push_flag = np.zeros(n_envs)
        self.ep_ret = np.zeros(n_envs)
        self.stats = []
        for i in range(n_envs):
            self._reset(i)

    def _sample_cmd(self, i):
        r = self.b.cmd_ranges
        c = np.array([self.rng.uniform(*r["vx"]), self.rng.uniform(*r["vy"]), self.rng.uniform(*r["wz"])])
        u = self.rng.random()
        if u < 0.15:
            c[:] = 0
        elif u < 0.45:        # forward + turn (the waypoint-teacher regime)
            c[1] = 0
        if np.linalg.norm(c[:2]) < 0.05:
            c[:2] = 0
        if abs(c[2]) < 0.05:
            c[2] = 0
        self.cmd[i] = c
        self.cmd_timer[i] = int(self.rng.integers(150, 300))

    def _reset(self, i):
        d = self.data[i]
        mujoco.mj_resetData(self.model, d)
        self.b.set_default(d, yaw=self.rng.uniform(-math.pi, math.pi), noise=0.05, rng=self.rng)
        mujoco.mj_forward(self.model, d)
        self.last_a[i] = 0
        self.phase[i] = self.rng.random()
        self.t[i] = 0
        self.air[i] = 0
        self.push_flag[i] = 0
        self.ep_ret[i] = 0
        self._sample_cmd(i)

    def obs(self, i):
        d = self.data[i]
        o = self.b.public_obs(d, self.cmd[i], self.last_a[i], self.phase[i])
        if self.obs_noise:
            nz = np.concatenate([np.full(3, 0.05), np.full(3, 0.03), np.zeros(3), np.full(self.b.n, 0.01),
                                 np.full(self.b.n, 0.05), np.zeros(self.b.n), np.zeros(2)]) * self.obs_noise
            o = o + (self.rng.standard_normal(o.shape) * nz).astype(np.float32)
        return o

    def priv(self, i, fc=None):
        d = self.data[i]
        if fc is None:
            fc, _ = self.b.contacts(d)
        return np.concatenate([self.b.base_lin_vel_body(d), [d.qpos[self.b.qa + 2] - self.b.nominal_height()],
                               fc.astype(float), [self.friction_scale - 1.0, self.push_flag[i]]]).astype(np.float32)

    def observe_all(self):
        return np.stack([self.obs(i) for i in range(self.n)]), np.stack([self.priv(i) for i in range(self.n)])

    def step(self, actions: np.ndarray):
        b, cfg, m = self.b, self.cfg, self.model
        O, P = [], []
        R = np.zeros(self.n)
        D = np.zeros(self.n, bool)
        T = np.zeros(self.n, bool)   # time-out (bootstrap)
        for i in range(self.n):
            d = self.data[i]
            a = np.clip(actions[i], -5, 5)
            d.ctrl[b.pol_act] = b.targets(a)
            tau2 = 0.0
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
            tau2 = float(np.sum((d.actuator_force[b.pol_act] / b.effort) ** 2))
            self.phase[i] = (self.phase[i] + self.dt / b.period) % 1.0
            self.t[i] += 1
            self.cmd_timer[i] -= 1
            if self.push and self.rng.random() < self.dt / 5.0:
                kick = self.rng.uniform(-0.4, 0.4, 2) * (0.5 if b.biped else 1.0)
                d.qvel[b.da:b.da + 2] += kick
                self.push_flag[i] = 1.0
            else:
                self.push_flag[i] *= 0.9
            fc, bad = b.contacts(d)
            v = b.base_lin_vel_body(d)
            quat, w = b.imu(d)
            g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
            c = self.cmd[i]
            r = cfg.track_lin * math.exp(-float(np.sum((c[:2] - v[:2]) ** 2)) / cfg.sigma)
            r += cfg.track_ang * math.exp(-float((c[2] - w[2]) ** 2) / cfg.sigma)
            r += cfg.lin_z * v[2] ** 2 + cfg.ang_xy * float(np.sum(w[:2] ** 2))
            r += cfg.orient * float(np.sum(g[:2] ** 2))
            r += cfg.torque * tau2 / b.n
            r += cfg.action_rate * float(np.sum((a - self.last_a[i]) ** 2)) / b.n * 4
            q = d.qpos[b.pol_qadr]
            span = b.jhi - b.jlo
            soft_lo, soft_hi = b.jlo + 0.05 * span, b.jhi - 0.05 * span
            r += cfg.limits * float(np.sum(np.clip(soft_lo - q, 0, None) + np.clip(q - soft_hi, 0, None)))
            moving = np.linalg.norm(c[:2]) > 0.05 or abs(c[2]) > 0.05
            first = fc & (self.air[i] > 0)
            r += cfg.air_time * float(np.sum((self.air[i] - 0.5 * b.period) * first)) * moving
            self.air[i] = np.where(fc, 0.0, self.air[i] + self.dt)
            if not moving:
                r += cfg.stand_still * float(np.sum(np.abs(q - b.q0))) / b.n * 4
                r += cfg.stand_contact * float(np.mean(fc))
            if cfg.contact_phase and b.nf == 2:
                if moving:
                    want = np.array([self.phase[i] < 0.55, self.phase[i] >= 0.45])  # left stance / right stance
                else:
                    want = np.array([True, True])
                r += cfg.contact_phase * float(np.mean(fc == want))
            if cfg.height:
                h = d.qpos[b.qa + 2]
                r += cfg.height * max(0.0, 0.93 * b.nominal_height() - h) ** 2
            # foot slip: feet in contact should not slide (privileged foot velocity)
            if cfg.feet_slip:
                slip = 0.0
                for k, fb in enumerate(b.foot_bids):
                    if fc[k]:
                        vel = np.zeros(6)
                        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb, vel, 0)
                        slip += float(np.sum(vel[3:5] ** 2))
                r += cfg.feet_slip * slip
            r += cfg.alive
            h = d.qpos[b.qa + 2]
            fell = bad or h < b.min_h or b.tilt(d) > b.tilt_limit or not np.isfinite(d.qpos).all()
            if fell:
                r += cfg.termination
            self.last_a[i] = a
            self.ep_ret[i] += r
            R[i] = r
            timeout = self.t[i] >= self.max_steps
            if fell or timeout:
                self.stats.append(dict(ret=float(self.ep_ret[i]), len=int(self.t[i]), fell=bool(fell)))
                D[i] = True
                T[i] = timeout and not fell
                self._reset(i)
                fc = None
            elif self.cmd_timer[i] <= 0:
                self._sample_cmd(i)
            O.append(self.obs(i))
            P.append(self.priv(i, fc))
        return np.stack(O), np.stack(P), R, D, T

    def pop_stats(self):
        s, self.stats = self.stats, []
        return s
