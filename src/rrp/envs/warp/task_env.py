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

from rrp.envs.mujoco.humanoid_scenes import (BURY, SCAN_X, SCAN_Y, STEPS_N, STEPS_PLATFORM, STEPS_TREAD, STEPS_X0,
                                              task_model)
from rrp.envs.warp.tracker_env import WarpTrackerEnv

H_MAX = 0.30


def task_adapted_model(key: str, task: str, params=None):
    """(model, meta, binding, adaptations): task scene + the declared `no_self_collision` adaptation, with every scene
    ground geom treated like the floor (collides with the robot)."""
    import mujoco
    from rrp.envs.mujoco.legged_core import LeggedBinding
    m, _, meta = task_model(key, task, params)
    b = LeggedBinding(m, meta)
    from rrp.envs.warp.mjx_legged import ADAPTATIONS, _leg_cross_collision
    walls = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in (meta.get("scene") or {}).get("walls", [])}
    _leg_cross_collision(m, ground=b.ground | walls)     # walls collide with the robot like the ground, not with the floor
    return m, meta, LeggedBinding(m, meta), [dict(name="leg_cross_collision", description=ADAPTATIONS["leg_cross_collision"][0]
                                                  + " (ground = floor + scene geoms)")]


class WarpStepsEnv(WarpTrackerEnv):
    def __init__(self, body, nworld: int, seed: int = 1, level: float = 0.0, **kw):
        kw.setdefault("cmd_mix", "default")
        # steps = fixed-size boxes on mocap bodies, moved per world (runtime geom size changes broke mujoco_warp contacts:
        # a flush h=0 staircase tipped h1 r6 over at x ~ 1 m in 100% of episodes, while the same model compiled at h=0 did not)
        super().__init__(body, nworld, seed, model_fn=lambda k: task_adapted_model(k, "h_steps", {"h_frac": 0.0, "mocap_h_max": H_MAX}),
                         **kw)
        import mujoco
        m = self.m
        names = [n for n in self.meta["scene"]["ground"]]
        self.step_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
        self.L = self.nominal_h                                   # (N,) leg length per world
        self.level = float(level)
        self.h = torch.zeros(self.N, device=self.dev)
        self.x_end = torch.zeros(self.N, device=self.dev)
        self.mocap_pos = self.wp.to_torch(self.dw.mocap_pos)            # (N, nmocap, 3)
        self.step_mocap = [int(m.body_mocapid[m.geom_bodyid[g]]) for g in self.step_gids]
        self.box_hz = float(0.5 * (BURY + (STEPS_N + 1) * H_MAX))        # x L
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
        """Resample step heights for masked worlds and move the mocap step boxes (tops at the new heights)."""
        if not hasattr(self, "mocap_pos"):
            return
        n, L = self.N, self.L
        hf = self._u(n) * self.level * H_MAX
        self.h = torch.where(mask, hf * L, self.h)
        h, t = self.h, STEPS_TREAD * L
        x = STEPS_X0 * L
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
        for mid, (cx, hx, top) in zip(self.step_mocap, rows):
            pos = torch.stack([cx, torch.zeros_like(cx), top - self.box_hz * L], -1)
            self.mocap_pos[:, mid] = torch.where(M, pos, self.mocap_pos[:, mid])
        self._rows = rows

    def height_at(self, xw):
        """Ground height at world x (N, P) for each world's staircase."""
        z = torch.zeros_like(xw)
        for cx, hx, top in self._rows:
            z = torch.where((xw - cx[:, None]).abs() <= hx[:, None], torch.maximum(z, top[:, None]), z)
        return z

    def _reset(self, mask):
        super()._reset(mask)
        if hasattr(self, "mocap_pos"):
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
        if not hasattr(self, "mocap_pos"):
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


# ------------------------------------------------------------------ L1 h_gap_sidestep
GAP_W = (1.6, 1.2)          # gap width / body width at level 0 .. 1


class WarpGapEnv(WarpTrackerEnv):
    """h_gap_sidestep expert env. Two mocap walls at x = GAP_X L leave a gap of width f x body width (f from 1.6 at level 0 to
    U(1.2, 1.6) at level 1) centred at y_c ~ U(-0.6, 0.6) L x level; afterwards turn in place to a final heading
    psi_f ~ U(-pi/2, pi/2) x level. SCRIPTED command layer (scripted_teacher): keep yaw 0, walk +x at 0.5 vx_max and sidestep
    (vy) onto the gap centre; past the wall, stop and turn to psi_f. Privileged expert inputs: gap centre in the yaw frame / L,
    wall distance / L, gap width / L, body width / L, sin/cos(psi_f - yaw), phase (7+1). Success: past the wall by 0.5 L and
    |psi_f - yaw| < 0.3 held 0.5 s. Wall contact: -2 per tick; > 0.2 s of contact ends the episode (failure wall_collision)."""

    def __init__(self, body, nworld: int, seed: int = 1, level: float = 0.0, **kw):
        kw.setdefault("cmd_mix", "default")
        super().__init__(body, nworld, seed, model_fn=lambda k: task_adapted_model(k, "h_gap_sidestep", {"mocap": True}), **kw)
        import mujoco
        from rrp.envs.mujoco.humanoid_scenes import body_width
        m = self.m
        self.L = self.nominal_h
        self.level = float(level)
        names = self.meta["scene"]["walls"]
        gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
        self.wall_mocap = [int(m.body_mocapid[m.geom_bodyid[g]]) for g in gids]
        self.is_wall = torch.zeros(m.ngeom, dtype=torch.bool, device=self.dev)
        self.is_wall[gids] = True
        keys = self.variant_keys
        bw = []
        for k in keys:
            mk, _, bk, _ = task_adapted_model(k, "h_gap_sidestep", {"mocap": True})
            bw.append(body_width(mk, bk))
        self.bw = torch.as_tensor(np.array(bw)[np.arange(self.N) % self.K], device=self.dev, dtype=torch.float32)
        self.mocap_pos = self.wp.to_torch(self.dw.mocap_pos)
        z = torch.zeros(self.N, device=self.dev)
        self.gap_w, self.gap_y, self.psi_f, self.phase2, self.wall_t, self.hold_t = z.clone(), z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
        self._layout(torch.ones(self.N, dtype=torch.bool, device=self.dev))
        self.reset_all()
        self.obs_dim = self.b.obs_dim + int(self.extra_obs().shape[1])

    def observe(self):
        return torch.cat([super().observe(), self.extra_obs()], -1)

    def set_level(self, level: float) -> float:
        self.level = float(min(1.0, max(0.0, level)))
        return self.level

    def _layout(self, mask):
        if not hasattr(self, "mocap_pos"):
            return
        n, L, lv = self.N, self.L, self.level
        f = GAP_W[0] - lv * self._u(n) * (GAP_W[0] - GAP_W[1])
        self.gap_w = torch.where(mask, f * self.bw, self.gap_w)
        self.gap_y = torch.where(mask, self._u(n, lo=-0.6, hi=0.6) * lv * L, self.gap_y)
        self.psi_f = torch.where(mask, self._u(n, lo=-math.pi / 2, hi=math.pi / 2) * lv, self.psi_f)
        from rrp.envs.mujoco.humanoid_scenes import GAP_X, WALL_HALF_Y, WALL_HALF_Z
        for mid, sg in zip(self.wall_mocap, (1.0, -1.0)):
            pos = torch.stack([GAP_X * L, self.gap_y + sg * (0.5 * self.gap_w + WALL_HALF_Y * L), WALL_HALF_Z * L], -1)
            self.mocap_pos[:, mid] = torch.where(mask[:, None], pos, self.mocap_pos[:, mid])
        for t_ in (self.phase2, self.wall_t, self.hold_t):
            t_.copy_(torch.where(mask, torch.zeros_like(t_), t_))

    def _reset(self, mask):
        super()._reset(mask)
        if hasattr(self, "mocap_pos"):
            self._layout(mask)
            yaw = self._u(self.N, lo=-0.2, hi=0.2)
            q = self.qpos
            q[:, self.qa] = torch.where(mask, torch.zeros_like(yaw), q[:, self.qa])
            q[:, self.qa + 1] = torch.where(mask, self._u(self.N, lo=-0.2, hi=0.2) * self.L, q[:, self.qa + 1])
            q[:, self.qa + 3] = torch.where(mask, torch.cos(yaw / 2), q[:, self.qa + 3])
            q[:, self.qa + 6] = torch.where(mask, torch.sin(yaw / 2), q[:, self.qa + 6])
            self._command()

    def _yaw(self):
        q = self.qpos[:, self.qa + 3:self.qa + 7]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _command(self):
        from rrp.envs.mujoco.humanoid_scenes import GAP_X
        x, y, yaw = self.qpos[:, self.qa], self.qpos[:, self.qa + 1], self._yaw()
        R = self.rng_cmd
        past = x > GAP_X * self.L + 0.5 * self.L
        self.phase2 = torch.where(past, torch.ones_like(self.phase2), self.phase2)
        wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))
        ey = self.gap_y - y
        vy_b = (-torch.sin(yaw) * 0 + torch.cos(yaw) * ey)                       # lateral error in the body frame (yaw ~ 0)
        c1 = torch.stack([0.5 * R[:, 0, 1], (1.5 * vy_b).clamp(R[:, 1, 0], R[:, 1, 1]), (1.5 * wrap(-yaw)).clamp(-0.5, 0.5)], -1)
        c2 = torch.stack([torch.zeros_like(x), torch.zeros_like(x), (1.5 * wrap(self.psi_f - yaw)).clamp(-0.5, 0.5)], -1)
        c2[:, 2] = torch.where(wrap(self.psi_f - yaw).abs() < 0.1, torch.zeros_like(c2[:, 2]), c2[:, 2])
        self.cmd = torch.where((self.phase2 > 0)[:, None], c2, c1)
        self.turn_cmd = (self.phase2 > 0) & (self.cmd[:, 2] != 0)
        self.cmd_timer = torch.full_like(self.cmd_timer, 1e6)

    def extra_obs(self):
        if not hasattr(self, "mocap_pos"):
            return torch.zeros(self.N, 8, device=self.dev)
        from rrp.envs.mujoco.humanoid_scenes import GAP_X
        x, y, yaw = self.qpos[:, self.qa], self.qpos[:, self.qa + 1], self._yaw()
        dx, dy = GAP_X * self.L - x, self.gap_y - y
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        L = self.L
        return torch.stack([(c * dx + s_ * dy) / L, (-s_ * dx + c * dy) / L, self.gap_w / L, self.bw / L,
                            torch.sin(self.psi_f - yaw), torch.cos(self.psi_f - yaw), self.phase2, dx / L], -1)

    def _wall_contact(self):
        N = self.N
        K = self.c_geom.shape[0]
        live = torch.arange(K, device=self.dev) < self.nacon[0]
        g = self.c_geom.long().clamp(0, self.m.ngeom - 1)
        w = self.c_world.long().clamp(0, N - 1)
        hit = live & (self.is_wall[g[:, 0]] ^ self.is_wall[g[:, 1]])
        return torch.zeros(N, device=self.dev).index_add_(0, w, hit.float()) > 0

    def task_step(self, fell):
        wc = self._wall_contact()
        self.wall_t = torch.where(wc, self.wall_t + self.dt, torch.zeros_like(self.wall_t))
        yaw = self._yaw()
        ok = (self.phase2 > 0) & (torch.atan2(torch.sin(self.psi_f - yaw), torch.cos(self.psi_f - yaw)).abs() < 0.3)
        self.hold_t = torch.where(ok, self.hold_t + self.dt, torch.zeros_like(self.hold_t))
        succ = (self.hold_t >= 0.5) & ~fell
        wall_fail = self.wall_t > 0.2
        self._command()
        r = -2.0 * wc.float() - 10.0 * wall_fail.float() + 10.0 * succ.float()
        return r, succ | wall_fail, succ
