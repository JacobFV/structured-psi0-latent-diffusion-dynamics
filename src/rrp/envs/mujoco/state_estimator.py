"""Base-state estimator for legged bodies (D-126 roadmap #27): base velocity from IMU + leg kinematics + contact.

Deployable inputs ONLY (every argument of `BaseStateEstimator.update` is something a real robot measures):
  - IMU orientation quaternion (w, x, y, z; body -> world), gyro (body frame, rad/s) and accelerometer
    (specific force in the body frame, m/s^2; MuJoCo convention: reads +g upward at rest);
  - joint encoders (positions, velocities) of the leg joints;
  - per-foot touch readings (contact = touch > threshold).
It never reads the simulator's base pose or base velocity (the free joint). The kinematics adapter
(`LegKinematics`) evaluates foot positions/velocities in the ROOT frame on a private MjData whose free joint is
pinned at the identity pose, from the encoder values only (the robot model is known at deployment).

Filter (`bse-1`, a complementary filter; a linear Kalman filter with fixed gain in the limit):
  predict:  v_w <- v_w + (R f + g_w) dt                       (IMU strapdown, world frame, g_w = [0, 0, -g])
  measure:  v_leg_b = mean over stance feet of -(v_foot_rel_b + omega_b x r_foot_b)   (stance feet do not move)
  correct:  v_w <- v_w + k (R v_leg_b - v_w),  k = dt / (tau + dt)   if any foot is in stance, else no correction
  position: p_w <- p_w + v_w dt                                (dead reckoning; drifts, reported as such)
Outputs: base linear velocity in the world frame and in the body frame, the dead-reckoned position, and the
number of stance feet used. Version string ESTIMATOR_VERSION is recorded wherever the estimate is used.

Option (LeggedSession / legged eval): `base_state_source` = "truth_noise" (default, unchanged: the declared noisy
localization sensor; speed from 1 s differences) | "estimator" (this filter: speed from 1 s differences of the
dead-reckoned position; NodeState.base_vel_estimate filled with the body-frame velocity).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ESTIMATOR_VERSION = "bse-1"
BASE_STATE_SOURCES = ("truth_noise", "estimator")
GRAVITY = 9.81


def quat_to_rot(q) -> np.ndarray:
    """Rotation matrix body -> world of a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def leg_odometry(foot_pos_b, foot_vel_b, gyro_b, stance) -> tuple[np.ndarray | None, int]:
    """Body-frame base velocity implied by the stance feet (feet in stance are assumed static in the world).

    foot_pos_b [nf, 3]: foot positions relative to the base, body frame; foot_vel_b [nf, 3]: foot velocities
    relative to the base due to joint motion (J_q qd), body frame; gyro_b [3]; stance [nf] bool.
    Returns (v_b or None when no foot is in stance, number of stance feet)."""
    st = np.asarray(stance, bool)
    n = int(st.sum())
    if n == 0:
        return None, 0
    r = np.asarray(foot_pos_b, float)[st]
    v = np.asarray(foot_vel_b, float)[st]
    w = np.asarray(gyro_b, float)
    vb = -(v + np.cross(w[None, :], r))
    return vb.mean(0), n


@dataclass
class EstimatorConfig:
    tau_s: float = 0.1                 # complementary time constant toward leg odometry (s)
    touch_threshold: float = 1.0       # touch sensor reading (N) above which a foot counts as in stance
    gravity: float = GRAVITY
    version: str = ESTIMATOR_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class BaseStateEstimator:
    """Complementary IMU + leg-odometry base velocity filter (see module docstring). Pure numpy."""

    def __init__(self, cfg: EstimatorConfig | None = None):
        self.cfg = cfg or EstimatorConfig()
        self.reset()

    def reset(self, v_w=None, p_w=None):
        self.v_w = np.zeros(3) if v_w is None else np.asarray(v_w, float).copy()
        self.p_w = np.zeros(3) if p_w is None else np.asarray(p_w, float).copy()
        self.R = np.eye(3)
        self.n_stance = 0
        self.t = 0.0

    def predict(self, quat, acc_b, dt: float):
        self.R = quat_to_rot(quat)
        a_w = self.R @ np.asarray(acc_b, float) + np.array([0.0, 0.0, -self.cfg.gravity])
        self.v_w = self.v_w + a_w * dt

    def correct(self, v_leg_b, dt: float):
        if v_leg_b is None:
            return
        k = dt / (self.cfg.tau_s + dt)
        self.v_w = self.v_w + k * (self.R @ np.asarray(v_leg_b, float) - self.v_w)

    def update(self, *, quat, gyro, acc, foot_pos_b, foot_vel_b, touch, dt: float) -> np.ndarray:
        """One filter step from deployable measurements; returns the world-frame velocity estimate."""
        self.predict(quat, acc, dt)
        stance = np.asarray(touch, float) > self.cfg.touch_threshold
        v_leg, self.n_stance = leg_odometry(foot_pos_b, foot_vel_b, gyro, stance)
        self.correct(v_leg, dt)
        self.p_w = self.p_w + self.v_w * dt
        self.t += dt
        return self.v_w

    @property
    def v_b(self) -> np.ndarray:
        return self.R.T @ self.v_w

    def state(self) -> dict:
        return dict(v_w=self.v_w.tolist(), p_w=self.p_w.tolist(), R=self.R.tolist(), n_stance=self.n_stance, t=self.t)

    def load(self, st: dict):
        self.v_w, self.p_w = np.array(st["v_w"], float), np.array(st["p_w"], float)
        self.R, self.n_stance, self.t = np.array(st["R"], float), int(st["n_stance"]), float(st["t"])


class LegKinematics:
    """Foot positions/velocities relative to the root, in the root frame, from ENCODER values only.

    A private MjData of the same model: free joint pinned at the origin with identity orientation and zero
    velocity; joint positions/velocities are set from the encoders (qadr/dadr = the robot's joint addresses),
    then foot-site positions (site_xpos) and velocities (site Jacobian @ qd) are the body-frame quantities."""

    def __init__(self, model, *, free_qadr: int, free_dadr: int, joint_qadr, joint_dadr, foot_site_ids):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.qa, self.da = int(free_qadr), int(free_dadr)
        self.qadr, self.dadr = np.asarray(joint_qadr, int), np.asarray(joint_dadr, int)
        self.sites = list(foot_site_ids)
        self._jac = np.zeros((3, model.nv))

    @classmethod
    def from_binding(cls, binding, joint_qadr, joint_dadr):
        return cls(binding.model, free_qadr=binding.qa, free_dadr=binding.da, joint_qadr=joint_qadr,
                   joint_dadr=joint_dadr, foot_site_ids=binding.foot_sids)

    def feet(self, q_enc, qd_enc) -> tuple[np.ndarray, np.ndarray]:
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0
        d.qpos[self.qa:self.qa + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[self.qadr] = q_enc
        d.qvel[self.dadr] = qd_enc
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        pos = np.array([d.site_xpos[s] for s in self.sites])
        vel = np.zeros_like(pos)
        for i, s in enumerate(self.sites):
            mj.mj_jacSite(m, d, self._jac, None, s)
            vel[i] = self._jac @ d.qvel
        return pos, vel
