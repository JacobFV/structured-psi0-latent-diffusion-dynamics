"""W13 P1c check (peer, GPU): MorphMultiEnv (GPU, morph_v1) observation == MorphSpec.obs (numpy, C-MuJoCo deployment) on the
same states for a menagerie body and a phum variant group. Prints JSON."""
import json

import mujoco
import numpy as np
import torch

from rrp.envs.mjx_legged import build_model
from rrp.envs.morph_obs import MorphSpec
from rrp.envs.warp_tracker_env import ADAPT, MorphMultiEnv

groups = [[["op3"], 16], [["phum_0", "phum_17"], 16]]
env = MorphMultiEnv(groups, seed=3, obs_noise=0.0)
for _ in range(5):
    env.step(torch.randn(env.N, env.nA, device="cuda") * 0.3)
o = env.observe().cpu().numpy()
err = {}
for (keys, _), e, g, sl in zip(groups, env.envs, env.specs, env.slices):
    for w in range(4):
        k = keys[w % len(keys)]
        m, meta, b, _ = build_model(k, "v2", ADAPT)
        sp = MorphSpec(m, b, meta)
        d = mujoco.MjData(m)
        d.qpos[:] = e.qpos[w].cpu().numpy()
        d.qvel[:] = e.qvel[w].cpu().numpy()
        mujoco.mj_forward(m, d)
        ref = sp.obs(b, d, e.cmd[w].cpu().numpy(), g["last"][w].cpu().numpy(), float(e.phase[w]))
        err[k] = max(err.get(k, 0.0), float(np.abs(ref - o[sl][w]).max()))
print(json.dumps(dict(obs_dim=env.obs_dim, max_err=err, finite=bool(np.isfinite(o).all()))))
assert max(err.values()) < 1e-4, err
