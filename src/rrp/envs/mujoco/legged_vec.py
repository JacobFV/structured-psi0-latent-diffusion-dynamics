"""Torch-free vectorised legged env pool (spawned CPU worker processes)."""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


# ------------------------------------------------------------------ workers
def _worker(conn, body, n_envs, seed, friction, kw):
    from rrp.envs.mujoco.legged_core import LeggedEnv
    from rrp.bodies.legged import legged_body
    env = LeggedEnv(lambda: legged_body(body), n_envs, seed, friction_scale=friction, **kw)
    conn.send(("spec", dict(obs_dim=env.b.obs_dim, priv_dim=env.priv_dim, contact=env.meta["contact_model"],
                            actuator_limits=env.meta.get("actuator_limits"),
                            reward=env.cfg0.version, reward_weights0=env.cfg.weights(), act_dim=env.b.n, dt=env.dt,
                            substeps=env.substeps, kind=env.b.kind,
                            **({"reward_options": env.cfg0.options()} if env.cfg0.options() else {}),
                            **({"actuator_speed_estimated": env.act.speed_estimated} if env.act is not None else {}),
                            **({"terrain": dict(env.terrain, version=env.meta.get("terrain", {}).get("version"))}
                               if env.terrain else {}))))
    while True:
        msg, payload = conn.recv()
        if msg == "reset":
            conn.send(env.observe_all())
        elif msg == "step":
            o, p, r, d, t = env.step(payload)
            conn.send((o, p, r, d, t, env.pop_stats()))
        elif msg == "alpha":
            env.set_alpha(payload)
            conn.send(("ok", env.cfg.weights()))
        elif msg == "turn":
            env.set_turn_scale(payload)
            conn.send(("ok", env.turn_scale))
        elif msg == "cmdmix":
            mix, _, stop = str(payload).partition(":")
            env.cmd_mix = mix
            if stop:
                env.teacher_stop = float(stop)
            conn.send(("ok", env.cmd_mix))
        elif msg == "refff":
            env.ref_ff = float(payload)
            conn.send(("ok", env.ref_ff))
        elif msg == "refffvmax":        # D-126 #13
            env.ref_ff_vmax = None if payload is None else float(payload)
            conn.send(("ok", env.ref_ff_vmax))
        elif msg == "terrain":          # D-126 #15: curriculum level -> this worker's amplitude (m)
            conn.send(("ok", env.set_terrain_amp(float(payload))))
        elif msg == "slowfrac":
            env.slow_frac = float(payload)
            conn.send(("ok", env.slow_frac))
        elif msg == "turnvx":
            env.turn_vx = float(payload)
            conn.send(("ok", env.turn_vx))
        elif msg == "turnfrac":
            env.turn_frac = float(payload)
            conn.send(("ok", env.turn_frac))
        elif msg == "close":
            conn.close()
            return


class VecPool:
    def __init__(self, body, workers, envs, seed, env_kw=None):
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"     # no BLAS thread pools spinning in N worker processes
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        frng = np.random.default_rng(seed + 999)
        for w in range(workers):
            a, b = ctx.Pipe()
            fr = float(frng.uniform(0.6, 1.25)) if workers > 1 else 1.0
            kw = env_kw or {}
            if kw.get("terrain"):       # D-126 #15: per-worker terrain pattern and amplitude fraction (spread over the level)
                kw = dict(kw, terrain=dict(kw["terrain"], seed=seed * 1000 + w, frac=(w + 1) / workers))
            p = ctx.Process(target=_worker, args=(b, body, envs, seed * 1000 + w, fr, kw), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.spec = [c.recv()[1] for c in self.conns][0]
        self.envs = envs

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        res = [c.recv() for c in self.conns]
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])

    def step(self, actions):
        for k, c in enumerate(self.conns):
            c.send(("step", actions[k * self.envs:(k + 1) * self.envs]))
        res = [c.recv() for c in self.conns]
        stats = [s for r in res for s in r[5]]
        return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
                np.concatenate([r[2] for r in res]), np.concatenate([r[3] for r in res]),
                np.concatenate([r[4] for r in res]), stats)

    def set_alpha(self, alpha: float) -> dict:
        for c in self.conns:
            c.send(("alpha", float(alpha)))
        return [c.recv()[1] for c in self.conns][0]

    def set_turn_scale(self, scale: float) -> float:
        for c in self.conns:
            c.send(("turn", float(scale)))
        return [c.recv()[1] for c in self.conns][0]

    def set_cmd_mix(self, mix: str) -> str:
        for c in self.conns:
            c.send(("cmdmix", mix))
        return [c.recv()[1] for c in self.conns][0]

    def set_ref_ff(self, amp: float) -> float:
        for c in self.conns:
            c.send(("refff", float(amp)))
        return [c.recv()[1] for c in self.conns][0]

    def set_ref_ff_vmax(self, vmax) -> float | None:
        for c in self.conns:
            c.send(("refffvmax", vmax))
        return [c.recv()[1] for c in self.conns][0]

    def set_terrain(self, level: float) -> list:
        """Curriculum level in [0, 1] -> per-worker amplitudes (m)."""
        for c in self.conns:
            c.send(("terrain", float(level)))
        return [c.recv()[1] for c in self.conns]

    def set_slow_frac(self, frac: float) -> float:
        for c in self.conns:
            c.send(("slowfrac", float(frac)))
        return [c.recv()[1] for c in self.conns][0]

    def set_turn_vx(self, vx: float) -> float:
        for c in self.conns:
            c.send(("turnvx", float(vx)))
        return [c.recv()[1] for c in self.conns][0]

    def set_turn_frac(self, frac: float) -> float:
        for c in self.conns:
            c.send(("turnfrac", float(frac)))
        return [c.recv()[1] for c in self.conns][0]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


