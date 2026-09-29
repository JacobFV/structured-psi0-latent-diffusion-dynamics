"""GPU task environments for W13 privileged RL experts (D-138 P2), built on rrp.envs.warp_tracker_env.

WarpStepsEnv (task h_steps, rrp.envs.humanoid_scenes): the staircase is part of the world; per-world step height
h = h_frac x L with h_frac ~ U(0, level x 0.30) (curriculum level, set_level). The command layer is SCRIPTED (heading
controller toward +x at 0.6 vx_max: label scripted_command); the learned expert maps (tracker observation + PRIVILEGED height
scan) -> joint targets (label privileged_teacher:rl_expert). Success: base x > x_end + 0.3 L without a fall.
Privileged extra observation (expert only, never a student input): ground height minus (base z - L) at an 11 x 3 grid in
the yaw frame (x -0.3 L .. 1.2 L, y -0.25 L .. 0.25 L), divided by L; plus h_frac.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from rrp.envs.humanoid_scenes import STEPS_N, STEPS_PLATFORM, STEPS_TREAD, STEPS_X0, task_model
from rrp.envs.warp_tracker_env import WarpTrackerEnv

SCAN_X = np.linspace(-0.3, 1.2, 11)
SCAN_Y = np.array([-0.25, 0.0, 0.25])
H_MAX = 0.30


def task_adapted_model(key: str, task: str, params=None):
    """(model, meta, binding, adaptations): task scene + the declared `no_self_collision` adaptation, with every scene
    ground geom treated like the floor (collides with the robot)."""
    import mujoco
    from rrp.envs.legged_core import LeggedBinding
    m, _, meta = task_model(key, task, params)
    b = LeggedBinding(m, meta)
    from rrp.envs.mjx_legged import ADAPTATIONS, _leg_cross_collision
    _leg_cross_collision(m, ground=b.ground)
    return m, meta, LeggedBinding(m, meta), [dict(name="leg_cross_collision", description=ADAPTATIONS["leg_cross_collision"][0]
                                                  + " (ground = floor + scene geoms)")]


class WarpStepsEnv(WarpTrackerEnv):
    def __init__(self, body, nworld: int, seed: int = 1, level: float = 0.0, **kw):
        kw.setdefault("cmd_mix", "default")
        super().__init__(body, nworld, seed, model_fn=lambda k: task_adapted_model(k, "h_steps"),
                         extra_batch=("geom_pos", "geom_size", "geom_aabb", "geom_rbound"), **kw)
        import mujoco
        m = self.m
        names = [n for n in self.meta["scene"]["ground"]]
        self.step_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
        self.L = self.nominal_h                                   # (N,) leg length per world
        self.level = float(level)
        self.h = torch.zeros(self.N, device=self.dev)
        self.x_end = torch.zeros(self.N, device=self.dev)
        wp = self.wp
        self.g_pos, self.g_size = wp.to_torch(self.mw.geom_pos), wp.to_torch(self.mw.geom_size)
        self.g_aabb, self.g_rb = wp.to_torch(self.mw.geom_aabb), wp.to_torch(self.mw.geom_rbound)
        sx, sy = np.meshgrid(SCAN_X, SCAN_Y, indexing="ij")
        self.scan_xy = torch.as_tensor(np.stack([sx.ravel(), sy.ravel()], -1), device=self.dev, dtype=torch.float32)
        self.prev_x = torch.zeros(self.N, device=self.dev)
        self._layout(torch.ones(self.N, dtype=torch.bool, device=self.dev))
        self.reset_all()
        self.obs_dim = self.b.obs_dim + int(self.extra_obs().shape[1])

    def observe(self):
        return torch.cat([super().observe(), self.extra_obs()], -1)

    def set_level(self, level: float) -> float:
        self.level = float(min(1.0, max(0.0, level)))
        return self.level

    def _layout(self, mask):
        """Resample step heights for masked worlds and write the batched geoms."""
        if not hasattr(self, "g_pos"):
            return
        n, L = self.N, self.L
        hf = self._u(n) * self.level * H_MAX
        self.h = torch.where(mask, hf * L, self.h)
        h, t = self.h, STEPS_TREAD * L
        hw = self.g_size[:, self.step_gids[0], 1].clone()
        x = STEPS_X0 * L
        k = 0
        rows = []
        for i in range(STEPS_N):
            rows.append((x + (i + 0.5) * t, 0.5 * t, (i + 1) * h))
        x2 = x + STEPS_N * t
        rows.append((x2 + 0.5 * STEPS_PLATFORM * L, 0.5 * STEPS_PLATFORM * L, (STEPS_N + 1) * h))
        x3 = x2 + STEPS_PLATFORM * L
        for i in range(STEPS_N):
            rows.append((x3 + (i + 0.5) * t, 0.5 * t, (STEPS_N - i) * h))
        self.x_end = torch.where(mask, x3 + STEPS_N * t, self.x_end)
        M = mask[:, None]
        for gid, (cx, hx, top) in zip(self.step_gids, rows):
            hz = (0.5 * top).clamp_min(1e-4)
            pos = torch.stack([cx, torch.zeros_like(cx), hz], -1)
            size = torch.stack([hx, hw, hz], -1)
            self.g_pos[:, gid] = torch.where(M, pos, self.g_pos[:, gid])
            self.g_size[:, gid] = torch.where(M, size, self.g_size[:, gid])
            shp = self.g_aabb[:, gid].shape
            new = torch.cat([torch.zeros(n, 3, device=self.dev), size], -1).reshape(shp)
            Mx = mask.view(-1, *([1] * (len(shp) - 1)))
            self.g_aabb[:, gid] = torch.where(Mx, new, self.g_aabb[:, gid])
            self.g_rb[:, gid] = torch.where(mask, size.norm(dim=-1), self.g_rb[:, gid])
        self._rows = rows

    def height_at(self, xw):
        """Ground height at world x (N, P) for each world's staircase."""
        z = torch.zeros_like(xw)
        for cx, hx, top in self._rows:
            z = torch.where((xw - cx[:, None]).abs() <= hx[:, None], torch.maximum(z, top[:, None]), z)
        return z

    def _reset(self, mask):
        super()._reset(mask)
        if hasattr(self, "g_pos"):
            self._layout(mask)
            yaw = self._u(self.N, lo=-0.3, hi=0.3)
            q = self.qpos
            q[:, self.qa] = torch.where(mask, self._u(self.N, lo=0.0, hi=0.4) * self.L, q[:, self.qa])
            q[:, self.qa + 1] = torch.where(mask, self._u(self.N, lo=-0.2, hi=0.2) * self.L, q[:, self.qa + 1])
            q[:, self.qa + 3] = torch.where(mask, torch.cos(yaw / 2), q[:, self.qa + 3])
            q[:, self.qa + 6] = torch.where(mask, torch.sin(yaw / 2), q[:, self.qa + 6])
            self.prev_x = torch.where(mask, q[:, self.qa], self.prev_x)
            self._command()

    def _yaw(self):
        q = self.qpos[:, self.qa + 3:self.qa + 7]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _command(self):
        """Scripted heading controller (scripted_command): walk +x at 0.6 vx_max, steer yaw and lateral offset to 0."""
        yaw = self._yaw()
        yerr = -self.qpos[:, self.qa + 1]
        head = torch.atan2(torch.sin(0.5 * torch.atan(yerr / self.L) - yaw), torch.cos(0.5 * torch.atan(yerr / self.L) - yaw))
        vx = 0.6 * self.rng_cmd[:, 0, 1]
        self.cmd = torch.stack([vx, torch.zeros_like(vx), (1.5 * head).clamp(-0.5, 0.5)], -1)
        self.turn_cmd = torch.zeros_like(self.turn_cmd)
        self.cmd_timer = torch.full_like(self.cmd_timer, 1e6)

    def extra_obs(self):
        if not hasattr(self, "g_pos"):
            return torch.zeros(self.N, SCAN_X.size * SCAN_Y.size + 1, device=self.dev)
        yaw = self._yaw()
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        L = self.L[:, None]
        px = self.qpos[:, self.qa:self.qa + 1] + L * (c[:, None] * self.scan_xy[:, 0] - s_[:, None] * self.scan_xy[:, 1])
        hz = self.height_at(px)
        base = self.qpos[:, self.qa + 2:self.qa + 3] - L
        return torch.cat([(hz - base) / L, (self.h / self.L)[:, None]], -1)

    def task_step(self, fell):
        x = self.qpos[:, self.qa]
        prog = (x - self.prev_x) / self.dt
        self.prev_x = x.clone()
        succ = (x > self.x_end + 0.3 * self.L) & ~fell
        self._command()
        r = 0.5 * prog.clamp(-1, 1) + 10.0 * succ.float()
        return r, succ, succ
