"""Torch-free vectorised legged env pool (spawned CPU worker processes)."""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


# ------------------------------------------------------------------ workers
def _worker(conn, body, n_envs, seed, friction, kw):
    from rrp.control.legged_core import LeggedEnv
    from rrp.morphology.legged import legged_body
    env = LeggedEnv(lambda: legged_body(body), n_envs, seed, friction_scale=friction, **kw)
    conn.send(("spec", dict(obs_dim=env.b.obs_dim, priv_dim=env.b.priv_dim, act_dim=env.b.n, dt=env.dt,
                            substeps=env.substeps, kind=env.b.kind)))
    while True:
        msg, payload = conn.recv()
        if msg == "reset":
            conn.send(env.observe_all())
        elif msg == "step":
            o, p, r, d, t = env.step(payload)
            conn.send((o, p, r, d, t, env.pop_stats()))
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
            p = ctx.Process(target=_worker, args=(b, body, envs, seed * 1000 + w, fr, env_kw or {}), daemon=True)
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


