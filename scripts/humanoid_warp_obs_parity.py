"""W13 P1b check (peer, GPU): the warp env's public observation equals LeggedBinding.public_obs (the C-MuJoCo deployment
observation LearnedTracker uses) on the same states, and a few random-action ticks stay finite. Prints JSON."""
import json
import sys

import mujoco
import numpy as np
import torch

from rrp.envs.warp_tracker_env import WarpTrackerEnv

body = sys.argv[1] if len(sys.argv) > 1 else "t1"
gate = len(sys.argv) > 2 and sys.argv[2] == "gate"
env = WarpTrackerEnv(body, 64, seed=3, obs_noise=0.0, clock_gate=gate)
env.cmd[:16] = 0                                   # some standing worlds (clock gated when gate)
for _ in range(5):
    env.step(torch.randn(env.N, env.nA, device="cuda") * 0.3)
o = env.observe().cpu().numpy()
m, b = env.m, env.b
d = mujoco.MjData(m)
err = 0.0
for w in list(range(8)) + list(range(16, 24)):
    d.qpos[:] = env.qpos[w].cpu().numpy()
    d.qvel[:] = env.qvel[w].cpu().numpy()
    mujoco.mj_forward(m, d)
    ref = b.public_obs(d, env.cmd[w].cpu().numpy(), env.last_a[w].cpu().numpy(), float(env.phase[w]), gate)
    err = max(err, float(np.abs(ref - o[w]).max()))
st = env.pop_stats()
print(json.dumps(dict(body=body, obs_dim=env.obs_dim, max_obs_err=err, finite=bool(np.isfinite(o).all()), episodes=st["episodes"])))
assert err < 1e-4, err
