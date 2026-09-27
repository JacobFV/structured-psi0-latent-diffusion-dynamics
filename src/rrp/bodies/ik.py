"""Damped-least-squares IK on a private MjData copy (kinematics only; never touches the sim)."""
from __future__ import annotations

import mujoco
import numpy as np


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def rot_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Axis-angle vector rotating current to desired (world frame)."""
    Re = R_des @ R_cur.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Re.reshape(-1))
    if q[0] < 0:
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    ang = 2 * np.arctan2(s, q[0])
    return v / s * ang


def down_rotation(yaw: float) -> np.ndarray:
    """TCP frame with z pointing down (-Z world) and x rotated by yaw about vertical."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class IKSolver:
    def __init__(self, model: mujoco.MjModel, site: str, joint_names: list[str], damping: float = 0.05,
                 rot_weight: float = 0.35):
        self.model = model
        self.data = mujoco.MjData(model)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.sid < 0:
            raise ValueError(f"site {site} missing")
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self.damping = damping
        self.rot_weight = rot_weight

    def fk(self, qpos_full: np.ndarray, q_arm: np.ndarray | None = None):
        self.data.qpos[:] = qpos_full
        if q_arm is not None:
            self.data.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.sid].copy(), self.data.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, qpos_full: np.ndarray, q_init: np.ndarray, pos: np.ndarray, R: np.ndarray | None,
              iters: int = 60, tol: float = 1e-3, seeds: list | None = None) -> tuple[np.ndarray, float]:
        """DLS from q_init; if the result misses by >1 cm, retry from heuristic seeds."""
        q, e = self._solve(qpos_full, q_init, pos, R, iters, tol)
        for s in (seeds or []):
            if e < 0.01:
                break
            q2, e2 = self._solve(qpos_full, np.clip(np.asarray(s, float), self.lo, self.hi), pos, R, iters * 2, tol)
            if e2 < e:
                q, e = q2, e2
        return q, e

    def _solve(self, qpos_full, q_init, pos, R, iters, tol):
        q = q_init.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        err_n = np.inf
        for _ in range(iters):
            p, Rc = self.fk(qpos_full, q)
            ep = pos - p
            er = rot_error(Rc, R) * self.rot_weight if R is not None else np.zeros(3)
            e = np.concatenate([ep, er])
            err_n = float(np.linalg.norm(ep))
            if err_n < tol and (R is None or np.linalg.norm(er) < 0.02):
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.sid)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr] * self.rot_weight if R is not None
                           else np.zeros((3, len(self.dadr)))])
            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            step = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + step, self.lo, self.hi)
        return q, err_n
