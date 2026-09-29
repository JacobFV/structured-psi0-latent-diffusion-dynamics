"""MuJoCo Warp legged stepping: ENGINE BAKE-OFF PROTOTYPE (W13 P1a, D-138). Not used by training/collection/eval yet.

Steps the legged tracker physics (rrp.bodies.legged body on the contact_v1/v2 floor via `standalone_model`, bounded PD
servos addressed through LeggedBinding, 10 substeps per 50 Hz tick) with mujoco_warp on the GPU, batched over `nworld`
worlds, and compares it with the C MuJoCo reference on the SAME compiled model (same adaptations on both sides).

mujoco_warp / warp-lang are NOT rrp dependencies. On the peer they live in an isolated target dir:
    uv pip install --python <rrp venv python> --target ~/work/ext/pylibs/mjwarp --no-deps mujoco-warp==3.14.0 warp-lang
and are put on PYTHONPATH only for this module (`PYTHONPATH=src:~/work/ext/pylibs/mjwarp`).

Model adaptations are the explicit, recorded ones of rrp.envs.mjx_legged.ADAPTATIONS (default: none); results carry the
adaptation list and its effect on the C reference, so a changed physics is never silent.

CLI (JSON on stdout):
    python -m rrp.envs.warp.warp_legged --body t1 [--contact v2] [--adapt no_self_collision] [--nworld 1024,4096]
        [--parity-ticks 50,200] [--tp-ticks 50] [--cpu-ticks 200]
"""
from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np

from rrp.envs.warp.mjx_legged import build_model, c_rollout, default_data, open_loop_targets, substeps_of

PROTOTYPE_VERSION = "warp_legged_proto_v0"
SOURCE_LABEL = "prototype:mujoco_warp_bakeoff (not a training/eval path)"
INSTALL_HINT = ("rrp.envs.warp_legged needs mujoco_warp + warp-lang (not rrp dependencies); peer: "
                "PYTHONPATH=src:$HOME/work/ext/pylibs/mjwarp")


def _wp():
    try:
        import warp as wp
        import mujoco_warp as mjw
    except ImportError as e:
        raise ImportError(f"{INSTALL_HINT} ({e})") from e
    wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
    return wp, mjw


def stand_targets(b, t: int, amp: float = 0.05) -> np.ndarray:
    """Gentle open-loop targets that keep a humanoid standing (small clock-phased oscillation about the default pose), so
    a parity rollout is not dominated by a chaotic fall."""
    return open_loop_targets(b, t, amp=amp)


class WarpLegged:
    """nworld copies of one legged model on the GPU; one 50 Hz tick = `substeps` mjw.step calls, CUDA-graph captured."""

    def __init__(self, body: str, nworld: int, contact: str = "v2", adapt=(), nconmax: int = 64, njmax: int = 384,
                 graph: bool = True):
        wp, mjw = _wp()
        self.wp, self.mjw = wp, mjw
        self.m, self.meta, self.b, self.adaptations = build_model(body, contact, adapt)
        self.body, self.nworld, self.substeps = body, int(nworld), substeps_of(self.m)
        self.d0 = default_data(self.m, self.b)
        self.mw = mjw.put_model(self.m)
        self.dw = mjw.put_data(self.m, self.d0, nworld=self.nworld, nconmax=nconmax, njmax=njmax)
        b = self.b
        self.pol = np.asarray(b.pol_act)
        self.held = np.asarray(b.held_act) if len(b.held_act) else None
        self.ctrl_host = np.tile(self.d0.ctrl, (self.nworld, 1)).astype(np.float32)
        if self.held is not None:
            self.ctrl_host[:, self.held] = b.q0_held
        self.graph = None
        if graph:
            self._tick_raw()                                   # compile kernels before capture
            with wp.ScopedCapture() as cap:
                self._tick_raw()
            self.graph = cap.graph

    def _tick_raw(self):
        for _ in range(self.substeps):
            self.mjw.step(self.mw, self.dw)

    def reset(self):
        self.wp.copy(self.dw.qpos, self.wp.array(np.tile(self.d0.qpos, (self.nworld, 1)).astype(np.float32)))
        self.wp.copy(self.dw.qvel, self.wp.array(np.zeros((self.nworld, self.m.nv), np.float32)))

    def step(self, targets: np.ndarray):
        """targets (nworld, n_policy_actuators). Holds them for one tick."""
        self.ctrl_host[:, self.pol] = targets
        self.wp.copy(self.dw.ctrl, self.wp.array(self.ctrl_host))
        if self.graph is not None:
            self.wp.capture_launch(self.graph)
        else:
            self._tick_raw()

    def qpos(self) -> np.ndarray:
        return self.dw.qpos.numpy()

    def ncon_max(self) -> int:
        return int(self.dw.nacon.numpy().max()) if hasattr(self.dw, "nacon") else -1


def parity(body: str, ticks: int, contact: str = "v2", adapt=(), amp: float = 0.05) -> dict:
    env = WarpLegged(body, 1, contact=contact, adapt=adapt)
    b, m = env.b, env.m
    fn = (lambda t: stand_targets(b, t, amp))
    ref = c_rollout(m, b, ticks, fn)
    env.reset()
    got = [env.qpos()[0].copy()]
    for t in range(ticks):
        env.step(np.asarray(fn(t))[None])
        got.append(env.qpos()[0].copy())
    got = np.array(got)
    err = np.abs(got - ref)
    h0 = float(ref[0, 2])
    return dict(body=body, ticks=ticks, amp=amp, substeps=env.substeps, max_dqpos=float(err.max()),
                max_dqpos_joints=float(err[:, 7:].max()), max_dbase_height=float(err[:, 2].max()),
                c_final_height=float(ref[-1, 2]), warp_final_height=float(got[-1, 2]),
                c_fell=bool(ref[:, 2].min() < 0.55 * h0), warp_fell=bool(got[:, 2].min() < 0.55 * h0),
                adaptations=env.adaptations)


def throughput(body: str, nworld: int, ticks: int, contact: str = "v2", adapt=()) -> dict:
    t0 = time.time()
    env = WarpLegged(body, nworld, contact=contact, adapt=adapt)
    build_s = time.time() - t0
    b = env.b
    env.reset()
    tg = [np.tile(open_loop_targets(b, t), (nworld, 1)).astype(np.float32) for t in range(ticks)]
    env.step(tg[0])
    env.wp.synchronize()
    t0 = time.time()
    for t in range(ticks):
        env.step(tg[t])
    env.wp.synchronize()
    wall = time.time() - t0
    q = env.qpos()
    return dict(body=body, nworld=nworld, ticks=ticks, build_capture_s=build_s, wall_s=wall,
                env_ticks_per_s=nworld * ticks / wall, finite=bool(np.isfinite(q).all()), ncon_max=env.ncon_max(),
                adaptations=env.adaptations)


def cpu_ticks_per_s(body: str, ticks: int, contact: str = "v2", adapt=()) -> dict:
    m, _, b, done = build_model(body, contact, adapt)
    d = default_data(m, b)
    ns = substeps_of(m)
    t0 = time.time()
    for t in range(ticks):
        d.ctrl[b.pol_act] = open_loop_targets(b, t)
        for _ in range(ns):
            mujoco.mj_step(m, d)
    wall = time.time() - t0
    return dict(body=body, ticks=ticks, one_thread_ticks_per_s=ticks / wall, adaptations=done)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--contact", default="v2")
    ap.add_argument("--adapt", default="")
    ap.add_argument("--nworld", default="1024,4096")
    ap.add_argument("--parity-ticks", default="50,200")
    ap.add_argument("--tp-ticks", type=int, default=50)
    ap.add_argument("--cpu-ticks", type=int, default=200)
    a = ap.parse_args(argv)
    adapt = [x for x in a.adapt.split(",") if x]
    out = dict(version=PROTOTYPE_VERSION, source=SOURCE_LABEL, body=a.body, contact=a.contact, adapt=adapt,
               cpu=cpu_ticks_per_s(a.body, a.cpu_ticks, a.contact, adapt), parity=[], throughput=[])
    for T in [int(x) for x in a.parity_ticks.split(",") if x]:
        out["parity"].append(parity(a.body, T, a.contact, adapt))
    for n in [int(x) for x in a.nworld.split(",") if x]:
        out["throughput"].append(throughput(a.body, n, a.tp_ticks, a.contact, adapt))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
