"""Damped-least-squares IK on a private MjData copy (kinematics only; never touches the sim).

Limit-aware option (D-126 #8, D-114 (3); default OFF): `solve(..., limit_margin=m)` with m > 0 (a fraction of each
joint's range, the unit of the D-112 joint-limit-margin gate) switches to
  * joint-weighted DLS: a joint inside the margin band gets weight w_i = max(d_i / m, 0.05) (d_i = normalized distance
    to its nearest limit), so the primary task moves it less the closer it is to its limit;
  * a null-space term (I - J+ J) g pushing every joint inside the band back toward the band edge (the secondary task
    never disturbs the primary Cartesian task to first order);
  * solution selection that prefers, among solutions within ok_tol (2 mm), the one with the larger margin (up to m).
With limit_margin = 0 every code path is the historical one (byte-identical solutions).
"""
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

    def margin(self, q: np.ndarray) -> float:
        """Smallest normalized distance of q to a joint limit (fraction of the range; joints with hi <= lo skipped)."""
        keep = self.hi > self.lo
        if not keep.any():
            return 1.0
        q = np.asarray(q, float)[keep]
        lo, hi = self.lo[keep], self.hi[keep]
        return float(np.min(np.minimum(q - lo, hi - q) / (hi - lo)))

    def solve(self, qpos_full: np.ndarray, q_init: np.ndarray, pos: np.ndarray, R: np.ndarray | None,
              iters: int = 60, tol: float = 1e-3, seeds: list | None = None,
              limit_margin: float = 0.0, ok_tol: float = 2e-3) -> tuple[np.ndarray, float]:
        """DLS from q_init; if the result misses by >1 cm, retry from heuristic seeds.
        limit_margin > 0: limit-aware solve (module docstring). The historical solution is computed first and kept when
        it is within ok_tol and outside the margin band; otherwise limit-aware solves (warm from it, cold from q_init,
        then the seeds) compete, ordered by (error < ok_tol, margin up to limit_margin, error). So the limit-aware solve
        never returns a solution that misses by more than ok_tol when the historical one did not."""
        if limit_margin <= 0:
            q, e = self._solve(qpos_full, q_init, pos, R, iters, tol)
            for s in (seeds or []):
                if e < 0.01:
                    break
                q2, e2 = self._solve(qpos_full, np.clip(np.asarray(s, float), self.lo, self.hi), pos, R, iters * 2, tol)
                if e2 < e:
                    q, e = q2, e2
            return q, e
        m = float(limit_margin)

        def key(q_, e_):     # within ok_tol first, then margin (capped at m), then error
            return (e_ >= ok_tol, -min(self.margin(q_), m), e_)
        q, e = self._solve(qpos_full, q_init, pos, R, iters, tol)            # the historical solution first
        if not (e < ok_tol and self.margin(q) >= m):
            for start in (q, q_init):                                       # limit-aware, warm and cold start
                q2, e2 = self._solve(qpos_full, start, pos, R, iters, tol, m)
                if key(q2, e2) < key(q, e):
                    q, e = q2, e2
        for s in (seeds or []):
            if e < ok_tol and self.margin(q) >= m:
                break
            q2, e2 = self._solve(qpos_full, np.clip(np.asarray(s, float), self.lo, self.hi), pos, R, iters * 2, tol, m)
            if key(q2, e2) < key(q, e):
                q, e = q2, e2
        return q, e

    def _limit_terms(self, q: np.ndarray, m: float):
        """(weights w, null-space gradient g) of the limit-aware solve for margin band m (fraction of range)."""
        rng = np.maximum(self.hi - self.lo, 1e-9)
        limited = self.hi > self.lo
        d_lo, d_hi = (q - self.lo) / rng, (self.hi - q) / rng
        d = np.where(limited, np.minimum(d_lo, d_hi), 1.0)
        w = np.where(d < m, np.maximum(d / m, 0.05), 1.0)
        # push toward the band edge, proportional to the depth into the band (rad)
        g = np.where(limited & (d_lo < m), (m - d_lo) * rng, 0.0) - np.where(limited & (d_hi < m), (m - d_hi) * rng, 0.0)
        return w, g

    def _solve(self, qpos_full, q_init, pos, R, iters, tol, limit_margin: float = 0.0):
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
            if err_n < tol and (R is None or np.linalg.norm(er) < 0.02) and (
                    limit_margin <= 0 or self.margin(q) >= limit_margin - 1e-6):
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.sid)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr] * self.rot_weight if R is not None
                           else np.zeros((3, len(self.dadr)))])
            if limit_margin > 0:
                w, g = self._limit_terms(q, limit_margin)
                JW = J * w[None, :]                                  # J W, W = diag(w)
                A = JW @ J.T + (self.damping ** 2) * np.eye(6)
                Jp = JW.T @ np.linalg.inv(A)                         # weighted damped pseudo-inverse W J^T (J W J^T + l2 I)^-1
                dq = Jp @ e + (np.eye(len(q)) - Jp @ J) @ g
            else:
                JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
                dq = J.T @ np.linalg.solve(JJt, e)
            step = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + step, self.lo, self.hi)
        if limit_margin > 0:                                         # error of the returned q (not the last iterate's)
            err_n = float(np.linalg.norm(pos - self.fk(qpos_full, q)[0]))
        return q, err_n
