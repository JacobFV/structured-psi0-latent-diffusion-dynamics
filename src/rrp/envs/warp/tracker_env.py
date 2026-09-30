"""GPU-batched legged tracker environment on MuJoCo Warp (W13 P1b, D-138): training-side twin of rrp.envs.legged_core.LeggedEnv.

Same body model (rrp.bodies.legged + contact_v2 floor + sourced limits) and the SAME public actor observation and action
convention as LeggedEnv / LeggedBinding.public_obs / LearnedTracker, so an actor trained here deploys unchanged in the C-MuJoCo
LeggedSession and is validated there (rrp.harness.eval.tracker_validation + gates.check_tracker, full self-collision).

Declared differences from LeggedEnv (recorded in actor meta `sim_engine` / `sim_adaptations` / `env_differences`):
  * physics: mujoco_warp (float32) with the explicit adaptation `no_self_collision` (robot geoms collide with the floor only);
    parity vs C MuJoCo on the same adapted model: max |dqpos| <= 5e-6 over 1 s (research/tracks/humanoid.md P1a);
  * randomisation per WORLD, resampled at each episode reset (LeggedEnv: per worker, every 500 ticks): floor mu, solref
    time constant / damping ratio, root mass x U(0.9, 1.1) and CoM shift +-2 cm, latency 0-4 substeps (contact_v2 ranges);
  * mechanical power for the power / CoT terms is the mean over substeps (as LeggedEnv);
  * slip: normal-force-weighted horizontal speed of the foot body AT each floor contact point (as LeggedBinding.stance).
Reward: rrp.envs.legged_core.RewardCfg weights (gait_v2 + overrides), the same terms and formulas as LeggedEnv.step.

Terrain (D-146): with `terrain_scan=True` the actor observation is proprio + the PUBLIC terrain scan (rrp.envs.mujoco.legged_core:
TERRAIN_SCAN_VERSION: 11 x 7 yaw-frame elevation cells, sigma 1 cm noise, 2 % dropout, one tick of latency), sampled from the
task's analytic ground height (`ground_z`; flat here) exactly as LeggedSession samples it with mj_ray; the critic block gets the
exact, current, noise-free scan (privileged). `extra_dim` is the width of that block (0 = blind actor).

Requires mujoco_warp + warp-lang (not rrp dependencies; peer: PYTHONPATH=src:~/work/ext/pylibs/mjwarp) and torch (CUDA).
"""
from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import torch

from rrp.envs.mujoco.legged_core import (CMD_SCALE, MIN_STOP_SHARE, SCAN_DIM, SCAN_DROPOUT, SCAN_DROPOUT_VALUE, SCAN_OFFSETS,
                                          SCAN_RANGE, SCAN_SIGMA, RewardCfg)
from rrp.envs.warp.model import build_model, default_data

ENV_VERSION = "warp_tracker_env_v1"
# declared physics adaptation for GPU training (C-MuJoCo validation always uses the full model). no_self_collision (r1 runs)
# let legs pass through each other: h1 r2b fell under lateral pushes in C MuJoCo (push no-fall 0.5). leg_cross_collision keeps
# left-vs-right leg contacts at the same GPU cost (t1 contended bench 13.7k vs 13.0k ticks/s at 1,024 worlds).
ADAPT = ("leg_cross_collision",)


def _wp():
    from rrp.envs.warp.model import _wp as f
    return f()


def quat_rot_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors v (N,3) into frames of unit quaternions q (N,4; w,x,y,z)."""
    w, x, y, z = q.unbind(-1)
    qv = torch.stack([-x, -y, -z], -1)
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + w[:, None] * t + torch.cross(qv, t, dim=-1)


MORPH_FIELDS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_subtreemass",
                "body_invweight0", "dof_invweight0", "geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_aabb", "geom_margin",
                "site_pos", "site_quat", "jnt_range", "qpos0", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                "actuator_ctrlrange", "dof_armature", "dof_damping")
_STREAM = {}


def _shared_stream(wp):
    """One torch stream shared by every env of the process (graph capture needs a non-default stream)."""
    if "t" not in _STREAM:
        wp.synchronize()
        _STREAM["t"] = torch.cuda.Stream()
        torch.cuda.set_stream(_STREAM["t"])
        _STREAM["w"] = wp.stream_from_torch(_STREAM["t"])
        wp.set_stream(_STREAM["w"])          # warp's current stream IS the shared stream (no stream switch inside captures)
    return _STREAM["w"]


PAYLOAD_ARM = 0.1          # m: lever arm of the training payload's added inertia


class WarpTrackerEnv:
    """`body`: one body key, or a list of keys with IDENTICAL topology (e.g. phum bodies of one topology): world w simulates
    variant w % K; per-variant model fields (MORPH_FIELDS, copied from each variant's own compiled model) and per-world
    morphology constants (nominal height, mass, torque limits, gait period, swing height, command ranges) are batched."""

    def __init__(self, body, nworld: int, seed: int = 1, *, reward_overrides: dict | None = None, episode_s: float = 20.0,
                 push: bool = True, obs_noise: float = 1.0, cmd_mix: str = "default", teacher_stop: float = MIN_STOP_SHARE,
                 turn_frac: float = 0.25, slow_frac: float = 0.0, randomize: bool = True, nconmax: int = 48,
                 njmax: int = 320, model_fn=None, extra_batch=(), clock_gate: bool = False, target_margin: float = 0.0,
                 land_vel: float = 0.0, force_cap: float = 0.0, force_cap_bw: float = 2.5, terrain_scan: bool = False,
                 upper_body: bool = False, upper_amp: float = 0.4, upper_speed: float = 1.5, payload_frac: float = 0.08,
                 require_payload: bool = True):
        wp, mjw = _wp()
        self.wp, self.mjw = wp, mjw
        self.dev = torch.device("cuda")
        keys = [body] if isinstance(body, str) else list(body)
        built = [(model_fn or (lambda k: build_model(k, "v2", ADAPT)))(k) for k in keys]
        self.m, self.meta, self.b, self.adaptations = built[0]
        m, b = self.m, self.b
        self.body, self.variant_keys, self.N, self.K = keys[0], keys, int(nworld), len(keys)
        for k, (mk, _, bk, _) in zip(keys, built):
            same = (mk.nq, mk.nv, mk.nu, mk.nbody, mk.ngeom, mk.nsite, mk.njnt) == (m.nq, m.nv, m.nu, m.nbody, m.ngeom, m.nsite, m.njnt)
            if not same or list(bk.pol_act) != list(b.pol_act) or list(bk.held_act) != list(b.held_act):
                raise ValueError(f"variant {k} does not share the topology of {keys[0]}")
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed))
        self.cfg0 = RewardCfg.for_kind(b.kind, "gait_v2")
        if reward_overrides:
            for k in reward_overrides:
                if not hasattr(self.cfg0, k):
                    raise KeyError(f"unknown reward term {k}")
            self.cfg0 = replace(self.cfg0, **{k: type(getattr(self.cfg0, k))(v) for k, v in reward_overrides.items()})
        self.cfg = self.cfg0.effective(0.0)
        self.alpha = 0.0
        self.cmd_mix, self.teacher_stop, self.turn_frac, self.slow_frac = cmd_mix, max(teacher_stop, MIN_STOP_SHARE), turn_frac, slow_frac
        self.turn_scale = 1.0
        # W13 clock gate (actor meta `clock_gate`): the gait-clock inputs are zeroed while the command is ~0 (a public
        # function of the command), so "stand" is an explicit mode instead of stepping on the clock (h1 r3/r4 stepped in place)
        self.clock_gate = bool(clock_gate)
        # W13 target_margin (actor meta): joint targets clipped to [lo + m span, hi - m span], deployed identically by
        # LeggedBinding.targets; land_vel: training-only penalty x sum over touchdown feet of the foot's downward speed^2
        self.target_margin, self.land_vel = float(target_margin), float(land_vel)
        # force_cap (training only): x sum over feet of max(0, normal force / (m g) - force_cap_bw) EVERY tick (the D-112 gate
        # checks the 20 ms-filtered per-foot peak <= 3.0 BW; touchdown-only impact terms did not bound it)
        self.force_cap, self.force_cap_bw = float(force_cap), float(force_cap_bw)
        self.push, self.obs_noise, self.randomize = push, obs_noise, randomize
        # U1 wholebody training (architecture 14.4): the `upper` group (the held joints: arms, waist, head) follows a RANDOM
        # scripted target trajectory (`random` source, training disturbance, not a policy output) and a payload (a random
        # mass on the two hand-side bodies) is drawn per episode. The actor sees the upper joint state (obs tail, public, as
        # LeggedBinding.upper_obs); the critic also sees the payload fraction and the target offset (privileged).
        self.upper_body = bool(upper_body)
        self.upper_amp, self.upper_speed, self.payload_frac = float(upper_amp), float(upper_speed), float(payload_frac)
        if self.upper_body and not len(b.held_act):
            raise ValueError(f"upper_body needs an upper group; body {keys[0]!r} has no held actuators")
        # require_payload=False (shared morphology tracker groups): a body with an upper group but no lateral hand links trains
        # with its upper targets and NO payload (recorded: upper_meta payload_bodies = [])
        if self.upper_body and require_payload and not len(b.payload_bodies()):
            raise ValueError(f"upper_body needs payload bodies (lateral hand links); body {keys[0]!r} has none")
        self.dt = 0.02
        self.substeps = max(1, int(round(self.dt / m.opt.timestep)))
        self.max_steps = int(episode_s / self.dt)
        d0 = default_data(m, b)
        self.d0 = d0
        # batched model fields: per-world randomisation and/or per-variant morphology
        bset = set(("geom_friction", "geom_solref", "body_mass", "body_inertia", "body_ipos") if randomize else ())
        if self.K > 1:
            bset |= set(MORPH_FIELDS)
        if self.upper_body:
            bset |= {"body_mass", "body_inertia"}
        bset |= set(extra_batch)
        self.mw = mjw.put_model(m, batch_sizes={k: self.N for k in sorted(bset)})
        vid_np = np.arange(self.N) % self.K
        if self.K > 1:
            for f in MORPH_FIELDS:
                arr = np.stack([getattr(mk, f) for mk, _, _, _ in built])[vid_np]
                dst = wp.to_torch(getattr(self.mw, f))
                if dst.numel() != arr.size:
                    raise ValueError(f"MORPH field {f}: warp shape {tuple(dst.shape)} vs {arr.shape}")
                dst.copy_(torch.as_tensor(arr, dtype=dst.dtype).reshape(dst.shape))
        self.dw = mjw.put_data(m, d0, nworld=self.N, nconmax=nconmax, njmax=njmax)
        T = lambda a: wp.to_torch(a)
        dw = self.dw
        self.qpos, self.qvel, self.ctrl = T(dw.qpos), T(dw.qvel), T(dw.ctrl)
        self.site_xpos, self.xpos, self.cvel, self.subtree_com = T(dw.site_xpos), T(dw.xpos), T(dw.cvel), T(dw.subtree_com)
        self.act_force = T(dw.actuator_force)
        self.qacc_ws = T(dw.qacc_warmstart)
        self.c_geom, self.c_pos, self.c_world = T(dw.contact.geom), T(dw.contact.pos), T(dw.contact.worldid)
        self.nacon = T(dw.nacon)
        self.cforce_wp = wp.zeros(dw.naconmax, dtype=wp.spatial_vectorf)
        self.cforce = T(self.cforce_wp)
        self.cids_wp = wp.array(np.arange(dw.naconmax, dtype=np.int32))
        f32 = lambda x: torch.as_tensor(np.asarray(x), device=self.dev, dtype=torch.float32)
        i64 = lambda x: torch.as_tensor(np.asarray(x), device=self.dev, dtype=torch.long)
        vw = lambda xs: f32(np.stack([np.asarray(x, dtype=np.float64) for x in xs])[vid_np])     # per-variant -> per-world
        ms, bs_ = [x[0] for x in built], [x[2] for x in built]
        d0s = [default_data(mk, bk) for mk, bk in zip(ms, bs_)]
        for bk, dk in zip(bs_, d0s):
            bk.foot_clearance(dk)                  # computes bk._foot_z0 (standing foot-site heights)
        rb = b.root_bid
        self.mass = vw([mk.body_subtreemass[rb] for mk in ms])
        self.nominal_h = vw([bk.nominal_height() for bk in bs_])
        self.mass0 = vw([mk.body_mass[rb] for mk in ms])
        self.inertia0 = vw([mk.body_inertia[rb] for mk in ms])
        self.ipos0 = vw([mk.body_ipos[rb] for mk in ms])
        self.qpos0 = vw([dk.qpos for dk in d0s])
        self.ctrl0 = vw([dk.ctrl for dk in d0s])
        self.period = vw([bk.period for bk in bs_])
        self.swing_h = vw([bk.swing_height for bk in bs_])
        self.min_h = vw([bk.min_h for bk in bs_])
        self.foot_z0 = vw([bk._foot_z0 for bk in bs_])
        self.q0, self.lo, self.hi = vw([bk.q0 for bk in bs_]), vw([bk.lo for bk in bs_]), vw([bk.hi for bk in bs_])
        self.jlo, self.jhi, self.effort = vw([bk.jlo for bk in bs_]), vw([bk.jhi for bk in bs_]), vw([bk.effort for bk in bs_])
        self.q0_held = vw([bk.q0_held for bk in bs_]) if len(b.held_act) else None
        cr = [bk.cmd_ranges for bk in bs_]
        self.rng_cmd = vw([[c["vx"], c["vy"], c["wz"]] for c in cr])          # (N, 3, 2)
        if randomize:
            self.m_fric, self.m_solref = T(self.mw.geom_friction), T(self.mw.geom_solref)
            self.m_mass, self.m_inertia, self.m_ipos = T(self.mw.body_mass), T(self.mw.body_inertia), T(self.mw.body_ipos)
        elif self.upper_body:
            self.m_mass, self.m_inertia = T(self.mw.body_mass), T(self.mw.body_inertia)
        if self.upper_body:
            self.held_lo = vw([bk.held_lo for bk in bs_])
            self.held_hi = vw([bk.held_hi for bk in bs_])
            self.pl_bodies = i64(b.payload_bodies())
            self.pl_mass0 = vw([mk.body_mass[b.payload_bodies()] for mk in ms])                # (N, npb)
            self.pl_inertia0 = vw([mk.body_inertia[b.payload_bodies()] for mk in ms])          # (N, npb, 3)
            self.has_payload = bool(len(self.pl_bodies))
        self.pol_act, self.pol_qadr, self.pol_dadr = i64(b.pol_act), i64(b.pol_qadr), i64(b.pol_dadr)
        self.held_act = i64(b.held_act) if len(b.held_act) else None
        self.held_qadr, self.held_dadr = i64(b.held_qadr), i64(b.held_dadr)
        self.qa, self.da, self.nA, self.nf = b.qa, b.da, b.n, b.nf
        self.scale = float(b.action_scale)
        self.foot_sids = i64(b.foot_sids)
        self.foot_bids = i64(b.foot_bids)
        # contact geometry tables
        floor_g = sorted(b.ground)
        self.is_floor = torch.zeros(m.ngeom, dtype=torch.bool, device=self.dev)
        self.is_floor[floor_g] = True
        foot_of_body = np.full(m.nbody, -1)
        for bb in range(m.nbody):
            if b.allowed_ground[bb]:
                p = bb
                while b.body_is_foot[p] < 0:
                    p = m.body_parentid[p]
                foot_of_body[bb] = b.body_is_foot[p]
        self.geom_body = i64(m.geom_bodyid)
        self.foot_of_body = i64(foot_of_body)
        self.robot_body = torch.as_tensor(b.is_robot_body, device=self.dev)
        self.body_root = i64(m.body_rootid)
        self.floor_gid = floor_g[0]
        self.ground_gids = floor_g
        self.tilt_limit = float(b.tilt_limit)
        # reference-gait indices (bipeds)
        self.ref_idx = None
        if b.biped and self.meta["legged"].get("pitch_actuators"):
            pi = b.pitch_idx()
            self.ref_idx = (i64(pi[0]), i64(pi[1]))
        import re
        acts = self.meta["legged"]["policy_actuators"]
        f = lambda side, pat: [k for k, a in enumerate(acts) if re.search(side, a, re.I) and re.search(pat, a, re.I)]
        try:
            if self.ref_idx is None:
                self.ref_idx = (i64([f("left", "hip_pitch")[0], f("left", "knee")[0], f("left", "ankle_pitch|ankle$")[0]]),
                                i64([f("right", "hip_pitch")[0], f("right", "knee")[0], f("right", "ankle_pitch|ankle$")[0]]))
        except IndexError:
            pass
        N, nA, nf = self.N, self.nA, self.nf
        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.cmd, self.last_a, self.last_a2 = z(N, 3), z(N, nA), z(N, nA)
        self.phase, self.t, self.cmd_timer = z(N), z(N), z(N)
        self.air, self.apex, self.stance_t = z(N, nf), z(N, nf), z(N, nf)
        self.push_flag, self.ep_ret, self.lat = z(N), z(N), torch.zeros(N, dtype=torch.long, device=self.dev)
        self.turn_cmd = torch.zeros(N, dtype=torch.bool, device=self.dev)
        self.fric_scale = torch.ones(N, device=self.dev)
        if self.upper_body:
            self.up_tgt, self.up_goal, self.up_timer, self.payload = self.q0_held.clone(), self.q0_held.clone(), z(N), z(N)
        self._gm_reset()
        # one mjw.step captured as a CUDA graph. Graph capture needs a non-default stream: the process's torch stream is set
        # to a dedicated stream that warp shares, so torch writes (ctrl, resets) and warp steps are ordered without host syncs
        self.stream = _shared_stream(wp)
        with wp.ScopedStream(self.stream):
            mjw.step(self.mw, self.dw)
            with wp.ScopedCapture() as cap:
                mjw.step(self.mw, self.dw)
        self.graph = cap.graph
        self.extra_dim = SCAN_DIM if terrain_scan else 0
        self.upper_dim = b.upper_dim if self.upper_body else 0
        self.obs_dim = b.obs_dim + self.extra_dim + self.upper_dim          # [public | terrain scan | upper joint state]
        self.priv_dim = b.priv_dim + 1 + self.extra_dim + ((1 + len(b.held_act)) if self.upper_body else 0)
        self._scan_off = torch.as_tensor(SCAN_OFFSETS, device=self.dev, dtype=torch.float32)
        self._scan_pub = torch.zeros(self.N, self.extra_dim, device=self.dev)
        self.reset_all()

    # ------------------------------------------------------------------ helpers
    def _u(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.dev)

    GM_KEYS = ("steps", "track_err", "cmd", "slip", "speed", "power", "cot_den", "turn_steps", "turn_cmd", "turn_w")

    def _gm_reset(self):
        # gate accumulators as device tensors (no host sync inside step); zeroed IN PLACE so a CUDA-graphed step keeps them
        if getattr(self, "gm", None) is None:
            self.gm = {k: torch.zeros((), device=self.dev) for k in self.GM_KEYS}
            self.ep_acc = {k: torch.zeros((), device=self.dev) for k in ("ret", "len", "n", "falls", "success")}
        else:
            for v in list(self.gm.values()) + list(self.ep_acc.values()):
                v.zero_()

    def set_alpha(self, a: float):
        self.alpha = float(min(1.0, max(0.0, a)))
        self.cfg = self.cfg0.effective(self.alpha)
        return self.cfg.weights()

    def _sample_cmd(self, mask: torch.Tensor):
        """Resample commands where `mask` (bool, N) is set; computed for every world and blended (no host sync)."""
        n = self.N
        R = self.rng_cmd
        c = torch.stack([self._u(n, lo=R[:, i, 0], hi=R[:, i, 1]) for i in range(3)], -1)
        u = self._u(n)
        c = torch.where((u < 0.15)[:, None], torch.zeros_like(c), c)
        fwd = (u >= 0.15) & (u < 0.45)
        c[:, 1] = torch.where(fwd, torch.zeros_like(c[:, 1]), c[:, 1])
        turn = (u >= 0.45) & (u < 0.45 + self.turn_frac) & self.b.biped
        sgn = torch.where(self._u(n) < 0.5, -1.0, 1.0)
        tz = sgn * self._u(n, lo=0.3, hi=1.0) * self.turn_scale * R[:, 2, 1]
        c = torch.where(turn[:, None], torch.stack([torch.zeros_like(tz), torch.zeros_like(tz), tz], -1), c)
        if self.cmd_mix.startswith("teacher"):      # W8 waypoint-teacher mix for 70% of commands (LeggedEnv._sample_teacher_mix)
            tm = self._u(n) < 0.7
            vm, wm = 0.6 * R[:, 0, 1], 0.8 * R[:, 2, 1]
            ut = self._u(n)
            zero = torch.zeros(n, device=self.dev)
            st = (ut >= self.teacher_stop) & (ut < 0.5)
            ar = (ut >= 0.5) & (ut < 0.85)
            pt = ut >= 0.85
            vx = torch.where(st, self._u(n, lo=0.3, hi=1.0) * vm, torch.where(ar, self._u(n, lo=0.2, hi=1.0) * vm, zero))
            wz = torch.where(st, self._u(n, lo=-0.1, hi=0.1), torch.where(ar, sgn * self._u(n, lo=0.3, hi=1.0) * wm,
                                                                          torch.where(pt, sgn * self._u(n, lo=0.6, hi=1.0) * wm, zero)))
            ct = torch.stack([vx, zero, wz], -1)
            c = torch.where(tm[:, None], ct, c)
            turn = torch.where(tm, pt, turn)
        small = c[:, :2].norm(dim=-1) < 0.05
        c[:, :2] = torch.where(small[:, None], torch.zeros_like(c[:, :2]), c[:, :2])
        c[:, 2] = torch.where(c[:, 2].abs() < 0.05, torch.zeros_like(c[:, 2]), c[:, 2])
        if self.slow_frac and self.b.biped:
            nr = c[:, :2].norm(dim=-1)
            sl = (nr > 0.05) & (self._u(n) < self.slow_frac)
            k = self._u(n, lo=0.05, hi=0.2) / nr.clamp_min(1e-6)
            c = torch.where(sl[:, None], c * torch.stack([k, k, torch.full_like(k, 0.3)], -1), c)
        self.cmd = torch.where(mask[:, None], c, self.cmd)
        self.turn_cmd = torch.where(mask, turn & (c[:, 2] != 0) & (c[:, 0] == 0), self.turn_cmd)
        tm_ = torch.randint(150, 300, (n,), generator=self.gen, device=self.dev).float()
        self.cmd_timer = torch.where(mask, tm_, self.cmd_timer)

    def _reset(self, mask: torch.Tensor):
        """Reset the worlds where `mask` (bool, N) is set (default pose, random yaw, joint noise, new randomisation)."""
        n = self.N
        M = mask[:, None]
        q = self.qpos0.clone()
        yaw = self._u(n, lo=-math.pi, hi=math.pi)
        q[:, self.qa + 2] = self.nominal_h + 0.005
        q[:, self.qa + 3] = torch.cos(yaw / 2)
        q[:, self.qa + 4:self.qa + 6] = 0
        q[:, self.qa + 6] = torch.sin(yaw / 2)
        q[:, self.pol_qadr] = torch.clamp(self.q0 + self._u(n, self.nA, lo=-0.05, hi=0.05), self.jlo, self.jhi)
        self.qpos.copy_(torch.where(M, q, self.qpos))
        self.qvel.copy_(torch.where(M, torch.zeros_like(self.qvel), self.qvel))
        self.qacc_ws.copy_(torch.where(M, torch.zeros_like(self.qacc_ws), self.qacc_ws))
        self.ctrl.copy_(torch.where(M, self.ctrl0, self.ctrl))
        for name in ("last_a", "last_a2", "apex", "stance_t", "air"):
            setattr(self, name, torch.where(M, torch.zeros_like(getattr(self, name)), getattr(self, name)))
        zero = torch.zeros_like(self.t)
        self.t = torch.where(mask, zero, self.t)
        self.push_flag = torch.where(mask, zero, self.push_flag)
        self.ep_ret = torch.where(mask, zero, self.ep_ret)
        self.phase = torch.where(mask, self._u(n), self.phase)
        if self.randomize:
            rb = self.b.root_bid
            mu = self._u(n, lo=0.4, hi=1.25)
            fr = torch.stack([mu, 0.02 * mu, 0.001 * mu], -1)
            sr = torch.stack([self._u(n, lo=0.006, hi=0.012), self._u(n, lo=0.8, hi=1.2)], -1)
            for g in self.ground_gids:                   # floor + task-scene ground share one draw per episode
                self.m_fric[:, g] = torch.where(M, fr, self.m_fric[:, g])
                self.m_solref[:, g] = torch.where(M, sr, self.m_solref[:, g])
            sc = self._u(n, lo=0.9, hi=1.1)
            self.m_mass[:, rb] = torch.where(mask, self.mass0 * sc, self.m_mass[:, rb])
            self.m_inertia[:, rb] = torch.where(M, self.inertia0 * sc[:, None], self.m_inertia[:, rb])
            self.m_ipos[:, rb] = torch.where(M, self.ipos0 + self._u(n, 3, lo=-0.02, hi=0.02), self.m_ipos[:, rb])
            self.fric_scale = torch.where(mask, mu / 0.9, self.fric_scale)
            self.lat = torch.where(mask, torch.randint(0, 5, (n,), generator=self.gen, device=self.dev), self.lat)
        if self.upper_body:
            self._reset_upper(mask)
        self._sample_cmd(mask)
        self._scan_sync(mask)

    def _reset_upper(self, mask):
        """Episode start of the masked worlds: the upper joints hold the default pose, a payload in [0, payload_frac x mass] is
        split over the hand-side bodies (added as a point mass PAYLOAD_ARM from the link centre: mass and inertia)."""
        n, M = self.N, mask[:, None]
        if self.has_payload:
            pl = self._u(n) * self.payload_frac * self.mass
            self.payload = torch.where(mask, pl, self.payload)
            npb = self.pl_mass0.shape[1]
            add = self.payload[:, None] / npb
            self.m_mass[:, self.pl_bodies] = torch.where(M, self.pl_mass0 + add, self.m_mass[:, self.pl_bodies])
            inertia = self.pl_inertia0 + (add * PAYLOAD_ARM ** 2)[:, :, None]     # a point load PAYLOAD_ARM from the link centre (com unchanged)
            self.m_inertia[:, self.pl_bodies] = torch.where(M[:, :, None], inertia, self.m_inertia[:, self.pl_bodies])
        self.up_tgt = torch.where(M, self.q0_held, self.up_tgt)
        self.up_goal = torch.where(M, self.q0_held, self.up_goal)
        self.up_timer = torch.where(mask, torch.zeros_like(self.up_timer), self.up_timer)

    def set_upper_ramp(self, amp: float, payload_frac: float) -> None:
        """Curriculum hook (HS2): the upper-body goal amplitude (fraction of each joint's half range) applies from the next
        target draw, the payload cap from the next episode reset (a running episode keeps its payload)."""
        self.upper_amp, self.payload_frac = float(amp), float(payload_frac)

    def _upper_targets(self):
        """One 50 Hz tick of the random upper-body target trajectory: every 1-3 s a new goal (20% rest at the default pose, else
        the default plus a uniform offset of upper_amp x the joint half-range), the applied target slews toward it at
        upper_speed rad/s and stays inside the actuator range. Computed for every world and blended (no host sync)."""
        n = self.N
        new = self.up_timer <= 0
        span = (self.held_hi - self.held_lo) * 0.5
        off = self.upper_amp * (2 * self._u(n, len(self.q0_held[0])) - 1) * span
        goal = torch.clamp(self.q0_held + off, self.held_lo, self.held_hi)
        goal = torch.where((self._u(n) < 0.2)[:, None], self.q0_held, goal)
        self.up_goal = torch.where(new[:, None], goal, self.up_goal)
        tm = torch.randint(50, 150, (n,), generator=self.gen, device=self.dev).float()
        self.up_timer = torch.where(new, tm, self.up_timer) - 1
        lim = self.upper_speed * self.dt
        self.up_tgt = torch.clamp(self.up_tgt + (self.up_goal - self.up_tgt).clamp(-lim, lim), self.held_lo, self.held_hi)
        return self.up_tgt

    def upper_meta(self) -> dict:
        """Recorded in the actor meta: how the upper-body targets and the payload were generated (a random training source)."""
        return dict(source="random", targets="goal every 1-3 s (20% rest), slew-limited, clipped to the actuator range",
                    amp=self.upper_amp, speed=self.upper_speed, payload_frac=self.payload_frac,
                    payload_bodies=[int(i) for i in (self.b.payload_bodies() if self.has_payload else [])], obs="q - q0, qdot * 0.05 (noise 0.01 / 0.05)")

    def reset_all(self):
        self._reset(torch.ones(self.N, dtype=torch.bool, device=self.dev))
        return self.observe()

    # ------------------------------------------------------------------ observations
    def _gravity_body(self):
        quat = self.qpos[:, self.qa + 3:self.qa + 7]
        g = torch.zeros(self.N, 3, device=self.dev)
        g[:, 2] = -1
        return quat_rot_inv(quat, g)

    def observe(self):
        gyro = self.qvel[:, self.da + 3:self.da + 6]
        if not hasattr(self, "_cs"):
            self._cs = torch.as_tensor(CMD_SCALE, device=self.dev, dtype=torch.float32)
            nA = self.nA
            self._nz = torch.cat([torch.full((3,), 0.05), torch.full((3,), 0.03), torch.zeros(3), torch.full((nA,), 0.01),
                                  torch.full((nA,), 0.05), torch.zeros(nA), torch.zeros(2)]).to(self.dev)
        o = torch.cat([gyro * 0.25, self._gravity_body(), self.cmd * self._cs,
                       self.qpos[:, self.pol_qadr] - self.q0, self.qvel[:, self.pol_dadr] * 0.05, self.last_a,
                       self.clock_obs()], -1)
        if self.obs_noise:
            o = o + torch.randn(o.shape, generator=self.gen, device=self.dev) * self._nz * self.obs_noise
        o = torch.cat([o, self.extra_obs()], -1) if self.extra_dim else o
        return torch.cat([o, self.upper_obs()], -1) if self.upper_body else o

    def upper_obs(self):
        """The public upper-body joint state (numpy twin: LeggedBinding.upper_obs), with the same encoder noise as the legs."""
        q = self.qpos[:, self.held_qadr] - self.q0_held
        qd = self.qvel[:, self.held_dadr] * 0.05
        if self.obs_noise:
            q = q + torch.randn(q.shape, generator=self.gen, device=self.dev) * 0.01 * self.obs_noise
            qd = qd + torch.randn(qd.shape, generator=self.gen, device=self.dev) * 0.05 * self.obs_noise
        return torch.cat([q, qd], -1)

    def clock_obs(self):
        c = torch.stack([torch.sin(2 * math.pi * self.phase), torch.cos(2 * math.pi * self.phase)], -1)
        if self.clock_gate:
            mv = ((self.cmd[:, :2].norm(dim=-1) > 0.05) | (self.cmd[:, 2].abs() > 0.05)).float()
            c = c * mv[:, None]
        return c

    # ------------------------------------------------------------------ public terrain scan (D-146)
    def _yaw(self):
        q = self.qpos[:, self.qa + 3:self.qa + 7]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def ground_z(self, xw, yw):
        """Ground height (m) at world (x, y), each (N, P). Flat floor here; task envs override with their scene."""
        return torch.zeros_like(xw)

    def terrain_exact(self):
        """(N, SCAN_DIM) exact scan of the current pose: nominal_h - clip(base_z - ground_z, 0, SCAN_RANGE) per cell (the
        numpy twin is rrp.envs.mujoco.legged_core.TerrainScan.exact, which casts mj_ray)."""
        yaw = self._yaw()
        c, s_ = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        o = self._scan_off
        px = self.qpos[:, self.qa:self.qa + 1] + c * o[:, 0] - s_ * o[:, 1]
        py = self.qpos[:, self.qa + 1:self.qa + 2] + s_ * o[:, 0] + c * o[:, 1]
        rng = (self.qpos[:, self.qa + 2:self.qa + 3] - self.ground_z(px, py)).clamp(0.0, SCAN_RANGE)
        return self.nominal_h[:, None] - rng

    def _scan_model(self, exact):
        """The declared sensor model (numpy twin: legged_core.scan_sensor_model): N(0, SCAN_SIGMA) noise, SCAN_DROPOUT of the
        cells read SCAN_DROPOUT_VALUE. Independent of `obs_noise` (the scan noise is the sensor's, not a training knob)."""
        v = exact + torch.randn(exact.shape, generator=self.gen, device=self.dev) * SCAN_SIGMA
        keep = torch.rand(exact.shape, generator=self.gen, device=self.dev) >= SCAN_DROPOUT
        return torch.where(keep, v, torch.full_like(v, SCAN_DROPOUT_VALUE))

    def _scan_sync(self, mask):
        """Episode start of the masked worlds: the published scan is the (noisy) scan of the new pose."""
        if self.extra_dim:
            self._scan_pub = torch.where(mask[:, None], self._scan_model(self.terrain_exact()), self._scan_pub)

    def extra_obs(self):
        """The PUBLIC extra actor inputs: the terrain scan of the pre-step pose (one tick of latency); (N, 0) when blind."""
        return self._scan_pub

    def task_priv(self):
        """Task-specific PRIVILEGED critic inputs (targets, layout); none for the plain tracker."""
        return torch.zeros(self.N, 0, device=self.dev)

    def task_step(self, fell):
        """Task hook after physics: (extra reward (N), task done (N, bool), success (N, bool)). Plain tracker: nothing."""
        z = torch.zeros(self.N, device=self.dev)
        return z, torch.zeros_like(fell), torch.zeros_like(fell)

    def _base_vel_body(self):
        quat = self.qpos[:, self.qa + 3:self.qa + 7]
        return quat_rot_inv(quat, self.qvel[:, self.da:self.da + 3])

    def privileged(self, fc):
        p = torch.cat([self._base_vel_body(), (self.qpos[:, self.qa + 2] - self.nominal_h)[:, None], fc.float(),
                       (self.fric_scale - 1)[:, None], self.push_flag[:, None],
                       torch.full((self.N, 1), self.alpha, device=self.dev)], -1)
        if self.extra_dim:
            p = torch.cat([p, self.terrain_exact()], -1)                              # the critic keeps the exact scan
        if self.upper_body:                                                           # payload fraction + upper target offset
            p = torch.cat([p, (self.payload / self.mass)[:, None], self.up_tgt - self.q0_held], -1)
        return p

    # ------------------------------------------------------------------ contacts
    def _stance(self):
        """(fc, fn, slip, bad) per world, as LeggedBinding.stance; over all contact slots with a validity mask (no host sync)."""
        N, nf = self.N, self.nf
        self.mjw.contact_force(self.mw, self.dw, self.cids_wp, False, self.cforce_wp)   # warp's stream is the shared stream
        K = self.c_geom.shape[0]
        live = torch.arange(K, device=self.dev) < self.nacon[0]
        g = self.c_geom.long().clamp(0, self.m.ngeom - 1)
        w = self.c_world.long().clamp(0, N - 1)
        f1, f2 = self.is_floor[g[:, 0]], self.is_floor[g[:, 1]]
        other = torch.where(f1, g[:, 1], g[:, 0])
        valid = live & (f1 ^ f2)
        bdy = self.geom_body[other]
        foot = self.foot_of_body[bdy]
        isfoot = valid & (foot >= 0)
        isbad = valid & (foot < 0) & self.robot_body[bdy]
        bad = torch.zeros(N, device=self.dev).index_add_(0, w, isbad.float()) > 0
        n = self.cforce[:, 0].clamp_min(0) * isfoot
        cv = self.cvel[w, bdy]                                        # (ang(3), lin(3)) at subtree_com of the root
        com = self.subtree_com[w, self.body_root[bdy]]
        vp = cv[:, 3:6] + torch.cross(cv[:, 0:3], self.c_pos - com, dim=-1)
        hs = vp[:, :2].norm(dim=-1)
        flat = w * nf + foot.clamp_min(0)
        cnt = torch.zeros(N * nf, device=self.dev).index_add_(0, flat, isfoot.float()).view(N, nf)
        fn = torch.zeros(N * nf, device=self.dev).index_add_(0, flat, n).view(N, nf)
        sv = torch.zeros(N * nf, device=self.dev).index_add_(0, flat, n * hs).view(N, nf)
        fc = cnt > 0
        slip = torch.where(fn > 1e-9, sv / fn.clamp_min(1e-9), torch.zeros_like(fn))
        return fc, fn, slip, bad

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        cfg, N = self.cfg, self.N
        a = actions.clamp(-5, 5)
        if self.target_margin:
            sp_ = self.hi - self.lo
            new_t = torch.clamp(self.q0 + self.scale * a, self.lo + self.target_margin * sp_, self.hi - self.target_margin * sp_)
        else:
            new_t = torch.clamp(self.q0 + self.scale * a, self.lo, self.hi)
        if self.extra_dim:       # the scan the actor sees after this step is the one taken from the pre-step pose
            self._scan_pub = self._scan_model(self.terrain_exact())
        old_full = self.ctrl.clone()
        if self.held_act is not None and not self.upper_body:
            old_full[:, self.held_act] = self.q0_held
        new_full = old_full.clone()
        new_full[:, self.pol_act] = new_t
        if self.upper_body:
            new_full[:, self.held_act] = self._upper_targets()
        pw = torch.zeros(N, device=self.dev)
        dadr = self.pol_dadr
        maxlat = 4                                      # latency draws are 0-4 substeps: afterwards every world has new targets
        for kk in range(self.substeps):
            if kk == 0:
                self.ctrl.copy_(torch.where((self.lat <= 0)[:, None], new_full, old_full))
            elif kk <= maxlat:
                self.ctrl.copy_(torch.where((self.lat <= kk)[:, None], new_full, old_full))
            if getattr(self, "_raw_step", False):     # inside a torch CUDA-graph capture: launch the kernels themselves
                self.mjw.step(self.mw, self.dw)
            else:
                self.wp.capture_launch(self.graph, stream=self.stream)
            pw += (self.act_force[:, self.pol_act] * self.qvel[:, dadr]).abs().sum(-1)
        pw /= self.substeps
        tau2 = ((self.act_force[:, self.pol_act] / self.effort) ** 2).sum(-1)
        self.phase = (self.phase + self.dt / self.period) % 1.0
        self.t += 1
        self.cmd_timer -= 1
        if self.push:
            pm = self._u(N) < self.dt / 5.0
            kick = self._u(N, 2, lo=-0.4, hi=0.4) * (0.5 if self.b.biped else 1.0)
            self.qvel[:, self.da:self.da + 2] += kick * pm[:, None]
            self.push_flag = torch.where(pm, torch.ones_like(self.push_flag), self.push_flag * 0.9)
        fc, fn, slip_v, bad = self._stance()
        v = self._base_vel_body()
        w = self.qvel[:, self.da + 3:self.da + 6]
        g = self._gravity_body()
        c = self.cmd
        nA = self.nA
        r = cfg.track_lin * torch.exp(-((c[:, :2] - v[:, :2]) ** 2).sum(-1) / cfg.sigma)
        r += cfg.track_ang * torch.exp(-(c[:, 2] - w[:, 2]) ** 2 / (cfg.sigma_ang or cfg.sigma))
        r += cfg.lin_z * v[:, 2] ** 2 + cfg.ang_xy * (w[:, :2] ** 2).sum(-1)
        r += cfg.orient * (g[:, :2] ** 2).sum(-1)
        r += cfg.torque * tau2 / nA
        r += cfg.action_rate * ((a - self.last_a) ** 2).sum(-1) / nA * 4
        if cfg.smooth:
            r += cfg.smooth * ((a - 2 * self.last_a + self.last_a2) ** 2).sum(-1) / nA * 4
        q = self.qpos[:, self.pol_qadr]
        span = self.jhi - self.jlo
        r += cfg.limits * ((self.jlo + 0.05 * span - q).clamp_min(0) + (q - self.jhi + 0.05 * span).clamp_min(0)).sum(-1)
        if cfg.limit_margin:
            keep = (span > 0).float()
            mg = torch.minimum(q - self.jlo, self.jhi - q) / span.clamp_min(1e-9)
            pen = ((cfg.limit_margin_m0 - mg).clamp_min(0) / cfg.limit_margin_m0) ** 2 * keep
            agg = {"mean": pen.sum(-1) / keep.sum(-1).clamp_min(1), "max": pen.max(-1).values,
                   "sum": pen.sum(-1)}[cfg.limit_margin_agg]
            r += cfg.limit_margin * agg
        cspd = c[:, :2].norm(dim=-1)
        moving = (cspd > 0.05) | (c[:, 2].abs() > 0.05)
        mf = moving.float()
        first = fc & (self.air > 0)
        per = self.period[:, None]
        r += cfg.air_time * ((self.air - 0.5 * per) * first).sum(-1) * mf
        clr = self.site_xpos[:, self.foot_sids, 2] - self.foot_z0
        if cfg.clearance:
            swing = (~fc) & (self.air < 0.6 * per)
            clr_n = (clr / self.swing_h[:, None]).clamp(0, 1)
            if self.b.biped:
                r += cfg.clearance * (clr_n * swing).sum(-1) * mf
            else:
                r += cfg.clearance * torch.where(swing.any(-1), (clr_n * swing).sum(-1) / swing.sum(-1).clamp_min(1), 0) * mf
        if self.force_cap:
            r += self.force_cap * (fn / (self.mass[:, None] * 9.81) - self.force_cap_bw).clamp_min(0).sum(-1)
        if self.land_vel:
            fb = self.foot_bids
            cvf = self.cvel[:, fb]                                                   # (N, nf, 6) ang, lin at subtree com
            com = self.subtree_com[:, self.body_root[fb]]
            vz = (cvf[..., 3:6] + torch.cross(cvf[..., 0:3], self.xpos[:, fb] - com, dim=-1))[..., 2]
            self._vz_prev = getattr(self, "_vz_prev", torch.zeros_like(vz))
            r += self.land_vel * ((self._vz_prev.clamp_max(0) ** 2) * first).sum(-1)
            self._vz_prev = vz
        if cfg.clearance_floor:
            self.apex = torch.where(fc, self.apex, torch.maximum(self.apex, clr))
            fl = cfg.floor_frac * self.swing_h[:, None]
            short = ((fl - self.apex) / fl).clamp(0, 1) ** 2
            r += cfg.clearance_floor * (short * first).sum(-1) * mf
            self.apex = torch.where(fc, torch.zeros_like(self.apex), self.apex)
        if cfg.slip:
            if self.b.biped:
                r += cfg.slip * slip_v.sum(-1)
            else:
                r += cfg.slip * torch.where(fc.any(-1), (slip_v * fc).sum(-1) / fc.sum(-1).clamp_min(1) * 2.0, 0)
        r += cfg.power * pw / (self.mass * 9.81 * cspd.clamp_min(0.25))
        if cfg.impact:
            r += cfg.impact * ((fn / (self.mass[:, None] * 9.81) - 1.0).clamp(0, 3) * first).sum(-1)
        mv = (cspd > 0.05).float()
        gm = self.gm
        spd = v[:, :2].norm(dim=-1)
        sl = torch.where(fc.any(-1), (slip_v * fc).sum(-1) / fc.sum(-1).clamp_min(1), torch.zeros_like(spd))
        gm["steps"] += mv.sum()
        gm["track_err"] += ((c[:, :2] - v[:, :2]).norm(dim=-1) * mv).sum()
        gm["cmd"] += (cspd * mv).sum()
        gm["slip"] += (sl * mv).sum()
        gm["speed"] += (spd * mv).sum()
        gm["power"] += (pw * mv).sum()
        gm["cot_den"] += (self.mass * 9.81 * spd * mv).sum()
        self.air = torch.where(fc, torch.zeros_like(self.air), self.air + self.dt)
        st = ~moving
        r += st.float() * (cfg.stand_still * (q - self.q0).abs().sum(-1) / nA * 4 + cfg.stand_vel * v[:, :2].norm(dim=-1)
                           + cfg.stand_contact * fc.float().mean(-1))
        pure_turn = self.turn_cmd | ((cspd < 0.05) & (c[:, 2].abs() > 0.05))
        if cfg.yaw_slip:
            fw = self.cvel[:, self.foot_bids, 2].abs()                # world-frame yaw rate of each foot body
            r += cfg.yaw_slip * (fw * fc).sum(-1)
        self.stance_t = torch.where(fc, self.stance_t + self.dt, torch.zeros_like(self.stance_t))
        sp = torch.sin(2 * math.pi * self.phase)
        if cfg.ref_step and self.ref_idx is not None:
            aL, aR = cfg.ref_amp * (-sp).clamp_min(0), cfg.ref_amp * sp.clamp_min(0)
            qr = self.q0.clone()
            for idx, a_ in ((self.ref_idx[0], aL), (self.ref_idx[1], aR)):
                qr[:, idx] += torch.stack([-a_, 2 * a_, -a_], -1)
            sel = torch.cat(self.ref_idx)
            dn = (q[:, sel] - qr[:, sel]).norm(dim=-1)
            r += cfg.ref_step * (torch.exp(-2 * dn) - 0.2 * dn.clamp_max(0.5)) * mf
        if cfg.stance_cap:
            cap = cfg.stance_cap_frac * per
            r += cfg.stance_cap * ((self.stance_t - cap) / per).clamp(0, 1).sum(-1) * mf
        wr = w[:, 2] * torch.sign(c[:, 2]) / c[:, 2].abs().clamp_min(1e-6)
        if cfg.turn_lin:
            app = pure_turn
            if cfg.yaw_lin_all:
                arc = c[:, 2].abs() > 0.05
                if cfg.yaw_lin_all_max:
                    arc = arc & (c[:, 2].abs() <= cfg.yaw_lin_all_max)
                app = app | arc
            r += cfg.turn_lin * wr.clamp(-0.5, cfg.yaw_progress_cap) * app.float()
        if cfg.yaw_overshoot:
            r += cfg.yaw_overshoot * (wr - 1.0).clamp_min(0) * (c[:, 2].abs() > 0.05).float()
        if cfg.ref_gait == "clock" and self.nf == 2:
            h_ref = cfg.ref_lift_frac * self.swing_h[:, None] * torch.stack([(-sp).clamp_min(0), sp.clamp_min(0)], -1) * mf[:, None]
            if cfg.ref_lift:
                r += cfg.ref_lift * torch.exp(-((clr - h_ref) / cfg.ref_lift_sigma) ** 2).mean(-1)
            if cfg.ref_contact:
                r += cfg.ref_contact * (fc == (h_ref <= 1e-9)).float().mean(-1)
        want = torch.stack([self.phase < 0.55, self.phase >= 0.45], -1)
        if cfg.turn_step and self.nf == 2:
            r += cfg.turn_step * (fc == want).float().mean(-1) * pure_turn.float()
        ptf = pure_turn.float()
        self.gm["turn_steps"] += ptf.sum()
        self.gm["turn_cmd"] += (c[:, 2].abs() * ptf).sum()
        self.gm["turn_w"] += (w[:, 2] * torch.sign(c[:, 2]) * ptf).sum()
        if cfg.contact_phase and self.nf == 2:
            wnt = torch.where(moving[:, None], want, torch.ones_like(want))
            r += cfg.contact_phase * (fc == wnt).float().mean(-1)
        h = self.qpos[:, self.qa + 2]
        if cfg.height:
            r += cfg.height * (0.93 * self.nominal_h - h).clamp_min(0) ** 2
        r += cfg.alive
        tilt = torch.acos((-g[:, 2]).clamp(-1, 1))
        fell = bad | (h < self.min_h) | (tilt > self.tilt_limit) | ~torch.isfinite(self.qpos).all(-1)
        r += cfg.termination * fell.float()
        r_task, tdone, succ = self.task_step(fell)
        r = r + r_task
        self.ep_acc["success"] += succ.float().sum()
        self.last_a2 = self.last_a
        self.last_a = a
        self.ep_ret += r
        timeout = (self.t >= self.max_steps) & ~tdone
        done = fell | timeout | tdone
        df = done.float()
        self.ep_acc["ret"] += (self.ep_ret * df).sum()
        self.ep_acc["len"] += (self.t * df).sum()
        self.ep_acc["n"] += df.sum()
        self.ep_acc["falls"] += fell.float().sum()
        priv_fc = fc & ~done[:, None]
        self._reset(done)
        self._sample_cmd((self.cmd_timer <= 0) & ~done)
        return self.observe(), self.privileged(priv_fc), r, done, timeout & ~fell & ~tdone

    def pop_stats(self) -> dict:
        """Episode stats since the last call plus the raw gate accumulators (summed by the trainer over a window)."""
        e = {k: float(v) for k, v in self.ep_acc.items()}
        out = dict(episodes=int(e["n"]), ret_sum=e["ret"], len_sum=e["len"], falls=int(e["falls"]),
                   successes=int(e.get("success", 0)), gm={k: float(v) for k, v in self.gm.items()})
        self._gm_reset()
        return out

    # ------------------------------------------------------------------ StateView (D-144 R8)
    def state_view(self, index: int):
        """`rrp.envs.base.StateView` for world `index` (capability "privileged_truth"; labels only, never a policy
        input): a host numpy snapshot of `self.m` (already CPU-side; built by `build_model`) and this step's body
        positions / contacts for that one world, via `warp_state_view_from_arrays` (the pure, CPU-testable part)."""
        xpos = self.xpos[index].detach().cpu().numpy()
        c_geom = self.c_geom.detach().cpu().numpy()
        c_pos = self.c_pos.detach().cpu().numpy()
        c_world = self.c_world.detach().cpu().numpy()
        t = float(self.t[index].detach().cpu())
        return warp_state_view_from_arrays(self.m, xpos, c_geom, c_pos, c_world, index, t)


def _warp_entities(m, xpos) -> list:
    """`m`: mujoco.MjModel (host, CUDA not needed); `xpos`: [nbody, 3] world body positions of ONE world."""
    from rrp.envs.base import EntityState
    out = []
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        pid = int(m.body_parentid[b])
        parent = None if pid == b else (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, pid) or f"body{pid}")
        out.append(EntityState(id=name, kind="body", name=name, pos=np.asarray(xpos[b], np.float64), quat=None,
                               vel=None, extent=None, mass=float(m.body_mass[b]), friction=None, material=None,
                               parent=parent, assembly=None, body=0, visible=True, attrs=None))
    return out


def _warp_contacts(m, c_geom, c_pos, c_world, index: int, t: float) -> list:
    """Contact slots of ONE world (`c_world[i] == index`; a warp `-1` geom id marks an unused slot). No contact normal
    / force is read by `WarpTrackerEnv` today (only `.geom`, `.pos`, `.worldid`; `_stance` computes force separately
    with `contact_force`), so `ContactState.normal` is zeros (unknown), not fabricated."""
    from rrp.envs.base import ContactState
    out = []
    for i in range(c_world.shape[0]):
        if int(c_world[i]) != index:
            continue
        g1, g2 = int(c_geom[i, 0]), int(c_geom[i, 1])
        if g1 < 0 or g2 < 0:
            continue
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"  # noqa: E731
        out.append(ContactState(a=name(b1), b=name(b2), pos=np.asarray(c_pos[i], np.float64), normal=np.zeros(3),
                                force=None, time=t))
    return out


class WarpStateView:
    """`rrp.envs.base.StateView` over one WarpTrackerEnv world, built from host numpy arrays (D-144 R8)."""

    def __init__(self, m, xpos, c_geom, c_pos, c_world, index: int, t: float):
        self.caps = frozenset({"poses", "contacts"})
        self.time = t
        self.gravity = np.asarray(m.opt.gravity, np.float64)
        self._entities = _warp_entities(m, xpos)
        self._contacts = _warp_contacts(m, c_geom, c_pos, c_world, index, t)

    def entities(self) -> list:
        return list(self._entities)

    def contacts(self) -> list:
        return list(self._contacts)

    def joints(self):
        from rrp.envs.base import CapabilityError
        raise CapabilityError("warp tracker StateView has no joints yet (D-144 R8 exposes bodies / contacts only)")

    def camera(self, name: str):
        from rrp.envs.base import CapabilityError
        raise CapabilityError("warp tracker StateView has no camera")

    def ui_tree(self):
        from rrp.envs.base import CapabilityError
        raise CapabilityError("warp tracker StateView has no ui_tree")

    def token_entity(self, token_set: str, slot) -> str | None:
        return None


def warp_state_view_from_arrays(m, xpos, c_geom, c_pos, c_world, index: int, t: float = 0.0) -> WarpStateView:
    """Pure builder (the CPU-testable part of `WarpTrackerEnv.state_view`): `m` a host `mujoco.MjModel`
    (`rrp.envs.warp.model.build_model`; no CUDA / mujoco_warp needed to build it), `xpos` [nbody, 3] for world
    `index`, `c_geom`/`c_pos`/`c_world` the flat per-slot contact pool of a batch (any number of worlds >= 1)."""
    return WarpStateView(m, xpos, c_geom, c_pos, c_world, index, t)


def window_metrics(recs: list) -> dict:
    """AlphaGate metrics from pop_stats records (same definitions as rrp.harness.train.reward_schedule.window_metrics)."""
    gm = {k: sum(r["gm"][k] for r in recs) for k in recs[0]["gm"]} if recs else {}
    eps = sum(r["episodes"] for r in recs)
    out = dict(steps=gm.get("steps", 0), episodes=eps, fall_rate=(sum(r["falls"] for r in recs) / eps) if eps else 0.0)
    if gm.get("turn_cmd", 0) > 0:
        out["turn_ratio"] = gm["turn_w"] / gm["turn_cmd"]
    if gm.get("steps"):
        out.update(track_rel_err=gm["track_err"] / max(gm["cmd"], 1e-9), slip_ratio=gm["slip"] / max(gm["speed"], 1e-9),
                   cot=gm["power"] / max(gm["cot_den"], 1e-9))
    return out


class GraphedStep:
    """Captures env.step(actions) (warp physics graphs + all torch reward/reset ops, sync-free) into ONE torch CUDA graph.
    Use: gs = GraphedStep(env); obs, priv, r, d, t = gs(actions). Outputs are graph-owned buffers (clone to keep).
    Randomness comes from the env's torch generator, registered with the graph."""

    def __init__(self, env, warmup: int = 3):
        self.env = env
        n_act = env.nA
        self.a = torch.zeros(env.N, n_act, device=env.dev)
        for _ in range(warmup):
            env.step(self.a)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        gens = [e.gen for e in getattr(env, "envs", [env])] + ([env.gen] if hasattr(env, "envs") else [])
        for gn in gens:
            self.g.register_generator_state(gn)
        subs = getattr(env, "envs", [env])
        for e in subs:
            e._raw_step = True
        try:
            wp = subs[0].wp
            with torch.cuda.graph(self.g, stream=torch.cuda.current_stream()):
                wp.capture_begin(stream=subs[0].stream, external=True)     # tell warp the capture is torch's
                try:
                    self.out = env.step(self.a)
                finally:
                    wp.capture_end(stream=subs[0].stream)
        finally:
            for e in subs:
                e._raw_step = False
        torch.cuda.synchronize()

    def __call__(self, actions):
        self.a.copy_(actions)
        self.g.replay()
        return self.out


class MorphMultiEnv:
    """Several WarpTrackerEnv groups (a body, or K same-topology variants) behind one morphology-conditioned interface
    (rrp.envs.morph_obs, obs format morph_v1, 14 canonical slot actions). Duck-types WarpTrackerEnv for the PPO trainer.

    `upper_body=True` (HS2, obs format morph_v2 = morph_v1 + a canonical upper block of UP_MAX padded slots, see morph_obs):
    every group whose body has an upper group trains with the random upper-body targets and payload; a group without one
    (e.g. a legs-only phum topology) contributes an all-zero upper block. The per-group dims are checked: all variants of a
    group share the upper joint count, and every group yields the same obs / critic width."""

    def __init__(self, groups: list, seed: int = 1, obs_noise: float = 1.0, env_cls=None, **env_kw):
        from rrp.envs.mujoco.morph_obs import CTX_DIM, DYN_DIM, NS, OBS_DIM, UP_MAX, UP_PRIV_DIM, UPPER_DIM, MorphSpec
        self.upper = bool(env_kw.get("upper_body"))
        self.envs, self.specs, self.slices, self.names = [], [], [], []
        self.obs_noise = obs_noise
        n0 = 0
        priv_dims = set()
        for gi, (keys, nw) in enumerate(groups):
            keys = [keys] if isinstance(keys, str) else list(keys)
            specs = []
            for k in keys:
                m, meta, b, _ = build_model(k, "v2", ADAPT)
                specs.append(MorphSpec(m, b, meta))
            s0 = specs[0]
            for sp in specs[1:]:
                if not (np.array_equal(sp.idx, s0.idx) and np.array_equal(sp.sign, s0.sign)):
                    raise ValueError(f"group {keys[0]}: variants disagree on slot mapping / signs")
                if sp.n_up != s0.n_up:
                    raise ValueError(f"group {keys[0]}: variants disagree on the upper joint count ({sp.n_up} vs {s0.n_up})")
            gkw = dict(env_kw, upper_body=self.upper and s0.n_up > 0, require_payload=False) if self.upper else env_kw
            e = (env_cls or WarpTrackerEnv)(keys, int(nw), seed=seed + 101 * gi, obs_noise=0.0, **gkw)
            dev = e.dev
            has_up = bool(gkw["upper_body"]) if self.upper else False
            if has_up and int(e.upper_dim) != 2 * s0.n_up:
                raise ValueError(f"group {keys[0]}: env upper block {e.upper_dim} != 2 x {s0.n_up} upper joints")
            # critic width: the env's [base | payload fraction | n_up offsets] is padded to [base | payload | UP_MAX offsets];
            # a group without an upper group gets payload 0 + zero offsets, so every group has the same width
            priv_dims.add(int(e.priv_dim) + ((UP_MAX - s0.n_up if has_up else UP_PRIV_DIM) if self.upper else 0))
            vid = np.arange(e.N) % e.K
            g = dict(idx=torch.as_tensor(s0.idx[s0.present], device=dev), slots=torch.as_tensor(np.nonzero(s0.present)[0], device=dev),
                     sign=torch.as_tensor(s0.sign[s0.present], device=dev, dtype=torch.float32),
                     ctx=torch.as_tensor(np.stack([sp.ctx for sp in specs])[vid], device=dev),
                     last=torch.zeros(e.N, NS, device=dev), cs=torch.as_tensor(CMD_SCALE, device=dev, dtype=torch.float32))
            g["has_up"], g["n_up"] = has_up, (s0.n_up if has_up else 0)
            if has_up:
                g["up_ctx"] = torch.as_tensor(np.stack([sp.up_ctx.ravel() for sp in specs])[vid], device=dev)
            g["pm"] = torch.zeros(NS, device=dev)
            g["pm"][g["slots"]] = 1.0
            self.envs.append(e)
            self.specs.append(g)
            self.slices.append(slice(n0, n0 + e.N))
            self.names.append(keys[0] if len(keys) == 1 else f"{keys[0]}+{len(keys) - 1}")
            n0 += e.N
        self.N, self.nA, self.nf = n0, NS, 2
        self.extra_dim = int(self.envs[0].extra_dim)
        self.upper_dim = UPPER_DIM if self.upper else 0     # morph_v2: padded upper joint state + static slot descriptors
        self.obs_dim, self.dyn_dim, self.ctx_dim = OBS_DIM + self.extra_dim + self.upper_dim, DYN_DIM, CTX_DIM
        if len(priv_dims) != 1:
            raise ValueError(f"groups disagree on the critic width {sorted(priv_dims)} (the shared critic needs one)")
        self.priv_dim = priv_dims.pop()
        self.dev = self.envs[0].dev
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed) + 7)
        self.b = self.envs[0].b
        self.meta = self.envs[0].meta
        self.adaptations = self.envs[0].adaptations
        self.cfg0 = self.envs[0].cfg0
        self.dt = self.envs[0].dt
        self._priv = [self._canon_priv(e, g, e.privileged(torch.zeros(e.N, e.nf, dtype=torch.bool, device=self.dev)))
                      for e, g in zip(self.envs, self.specs)]
        for p in self._priv:
            assert p.shape[-1] == self.priv_dim, (p.shape, self.priv_dim)

    def _canon_priv(self, e, g, p):
        """Critic block at the canonical width (see __init__): pad the upper target offsets to UP_MAX, or zeros for a group
        with no upper body (payload fraction 0, offsets 0)."""
        if not self.upper:
            return p
        from rrp.envs.mujoco.morph_obs import UP_MAX, UP_PRIV_DIM
        if g["has_up"]:
            return torch.cat([p, torch.zeros(p.shape[0], UP_MAX - g["n_up"], device=p.device)], -1)
        return torch.cat([p, torch.zeros(p.shape[0], UP_PRIV_DIM, device=p.device)], -1)

    def set_upper_ramp(self, amp: float, payload_frac: float) -> None:
        for e in self.envs:
            if getattr(e, "upper_body", False):
                e.set_upper_ramp(amp, payload_frac)
        self.upper_amp, self.payload_frac = float(amp), float(payload_frac)

    def upper_meta(self) -> dict:
        rec = dict(groups={n: (e.upper_meta() if getattr(e, "upper_body", False) else None) for n, e in zip(self.names, self.envs)})
        ref = next((e for e in self.envs if getattr(e, "upper_body", False)), None)
        if ref is not None:
            rec.update(amp=ref.upper_amp, speed=ref.upper_speed, payload_frac=ref.payload_frac)
        return rec

    def _upper_block(self, e, g):
        """morph_v2 upper block of one group: padded q - q0 and qdot x 0.05 (encoder noise as the legs), then the static descriptors."""
        from rrp.envs.mujoco.morph_obs import UP_CTX_DIM, UP_MAX
        if not g["has_up"]:
            return torch.zeros(e.N, 2 * UP_MAX + UP_CTX_DIM, device=self.dev)
        q = e.qpos[:, e.held_qadr] - e.q0_held
        qd = e.qvel[:, e.held_dadr] * 0.05
        if self.obs_noise:
            q = q + torch.randn(q.shape, generator=self.gen, device=self.dev) * 0.01 * self.obs_noise
            qd = qd + torch.randn(qd.shape, generator=self.gen, device=self.dev) * 0.05 * self.obs_noise
        pad = UP_MAX - g["n_up"]
        z = torch.zeros(e.N, pad, device=self.dev)
        return torch.cat([q, z, qd, z, g["up_ctx"]], -1)

    def set_alpha(self, a: float):
        w = None
        for e in self.envs:
            w = e.set_alpha(a)
        self.alpha = a
        return w

    def _slot(self, g, x):
        out = torch.zeros(x.shape[0], self.nA, device=self.dev)
        out[:, g["slots"]] = g["sign"] * x[:, g["idx"]]
        return out

    def _group_obs(self, e, g):
        from rrp.envs.mujoco.morph_obs import NS
        gyro = e.qvel[:, e.da + 3:e.da + 6] * 0.25
        grav = e._gravity_body()
        cmd = e.cmd * g["cs"]
        q = self._slot(g, e.qpos[:, e.pol_qadr] - e.q0)
        qd = self._slot(g, e.qvel[:, e.pol_dadr] * 0.05)
        if self.obs_noise:
            pm = g["pm"]
            nz = lambda shape, s: torch.randn(shape, generator=self.gen, device=self.dev) * s * self.obs_noise
            gyro = gyro + nz(gyro.shape, 0.05)
            grav = grav + nz(grav.shape, 0.03)
            q = q + nz(q.shape, 0.01) * pm
            qd = qd + nz(qd.shape, 0.05) * pm
        clock = e.clock_obs()
        o = torch.cat([gyro, grav, cmd, q, qd, g["last"], clock, g["ctx"], e.extra_obs()], -1)
        return torch.cat([o, self._upper_block(e, g)], -1) if self.upper else o

    def observe(self):
        return torch.cat([self._group_obs(e, g) for e, g in zip(self.envs, self.specs)], 0)

    def privileged(self, fc=None):
        return torch.cat(self._priv, 0)

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        R, D, T = [], [], []
        for i, (e, g, sl) in enumerate(zip(self.envs, self.specs, self.slices)):
            a_slot = actions[sl].clamp(-5, 5)
            a = torch.zeros(e.N, e.nA, device=self.dev)
            a[:, g["idx"]] = g["sign"] * a_slot[:, g["slots"]]
            _, priv, r, d, tmo = e.step(a)
            pm = g["pm"]
            g["last"] = torch.where(d[:, None], torch.zeros_like(a_slot), a_slot * pm)
            priv = self._canon_priv(e, g, priv)
            self._priv[i] = priv
            R.append(r)
            D.append(d)
            T.append(tmo)
        return self.observe(), torch.cat(self._priv, 0), torch.cat(R), torch.cat(D), torch.cat(T)

    def pop_stats(self) -> dict:
        recs = [e.pop_stats() for e in self.envs]
        out = dict(episodes=sum(r["episodes"] for r in recs), ret_sum=sum(r["ret_sum"] for r in recs),
                   len_sum=sum(r["len_sum"] for r in recs), falls=sum(r["falls"] for r in recs),
                   successes=sum(r.get("successes", 0) for r in recs),
                   gm={k: sum(r["gm"][k] for r in recs) for k in recs[0]["gm"]},
                   per_group={n: dict(episodes=r["episodes"], falls=r["falls"], successes=r.get("successes", 0),
                                      track_rel_err=(r["gm"]["track_err"] / r["gm"]["cmd"]) if r["gm"]["cmd"] else None)
                              for n, r in zip(self.names, recs)})
        return out


class WarpEnv:
    """`rrp.envs.base.Env` (env_id "warp/legged", capability "batched") over a WarpTrackerEnv-like engine (WarpTrackerEnv,
    WarpStepsEnv, WarpGapEnv, MorphMultiEnv). Commands: BatchCommand(groups={"legs": [N, nA] tensor}) of normalized
    residual joint targets; StepResult.reward = (r, done, timeout) tensors; truth() = the privileged critic block of the
    last step (never part of the actor observation). Commands are rrp.envs.base.BatchCommand. The engine keeps its own
    API for the PPO loop."""

    def __init__(self, engine, task: str = "locomotion"):
        self.engine, self.task = engine, task
        self._priv = None
        self._t = 0.0

    @property
    def spec(self):
        from rrp.envs.base import ActionSpace, BodyInfo, EnvSpec
        e = self.engine
        keys = list(getattr(e, "variant_keys", None) or getattr(e, "names", None) or [getattr(e, "body", "?")])
        dt = float(getattr(e, "dt", 0.02))
        return EnvSpec(env_id="warp/legged", backend="mujoco_warp", task=self.task,
                       bodies=[BodyInfo(robot=0, family=getattr(getattr(e, "b", None), "kind", "legged"), key=k,
                                        robot_spec_hash=f"warp:{k}") for k in keys],
                       batch=int(e.N), control_hz=1.0 / dt,
                       action_spaces=[ActionSpace(group="legs", kind="joint_position", width=int(e.nA), rate_hz=1.0 / dt,
                                                  low=[-5.0] * int(e.nA), high=[5.0] * int(e.nA), units="normalized")],
                       capabilities=["batched", "vector_obs", "reward", "privileged_truth", "proprio"]
                       + (["terrain_scan"] if int(getattr(e, "extra_dim", 0)) else []),
                       provenance=dict(engine=type(e).__name__, obs_dim=int(e.obs_dim)))

    def _vobs(self, o):
        from rrp.envs.base import VectorObservation
        return VectorObservation(vec=o, layout=[("tracker_obs", int(o.shape[-1]))], time=self._t,
                                 robot_spec_hashes=[b.robot_spec_hash for b in self.spec.bodies])

    def reset(self, seed=None):
        self._t = 0.0
        return self._vobs(self.engine.reset_all())

    def observe(self):
        return self._vobs(self.engine.observe())

    def step(self, command):
        from rrp.envs.base import StepResult
        o, priv, r, done, timeout = self.engine.step(command.groups["legs"])
        self._priv = priv
        self._t += float(getattr(self.engine, "dt", 0.02))
        return StepResult(self._vobs(o), None, self._t, source=command.source, reward=(r, done, timeout))

    def truth(self):
        return self._priv

    def close(self):
        pass


def make_warp_env(*, task: str = "locomotion", body, seed: int = 1, nworld: int = 1024, **kw) -> WarpEnv:
    """env_id "warp/legged": task "locomotion" (WarpTrackerEnv), "h_steps" (WarpStepsEnv), "h_gap" (WarpGapEnv)."""
    if task == "locomotion":
        return WarpEnv(WarpTrackerEnv(body, nworld, seed, **kw), task)
    from rrp.envs.warp.task_env import WarpGapEnv, WarpStepsEnv
    cls = {"h_steps": WarpStepsEnv, "h_gap": WarpGapEnv}[task]
    return WarpEnv(cls(body, nworld, seed, **kw), task)
