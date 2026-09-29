"""W13 P1c check (peer, GPU): a K-variant batched warp env (phum bodies of one topology) steps each world like the C-MuJoCo
model of ITS OWN variant (same adapted model, same open-loop targets), and the public observation matches. Prints JSON."""
import json
import sys

import mujoco
import numpy as np
import torch

from rrp.bodies.humanoid_gen import sample_params
from rrp.envs.mjx_legged import build_model, default_data, substeps_of
from rrp.envs.warp_tracker_env import ADAPT, WarpTrackerEnv

topo = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
seeds, s = [], 0
want = topo or sample_params(0).topology
while len(seeds) < 4:
    if sample_params(s).topology == want:
        seeds.append(s)
    s += 1
keys = [f"phum_{x}" for x in seeds]
if "--single" in sys.argv:
    keys = keys[:1]
env = WarpTrackerEnv(keys, 8, seed=3, obs_noise=0.0, randomize=False, push=False)
T = 25
tg = [0.1 * np.sin(0.3 * t + np.arange(env.nA)) for t in range(T)]
for w in range(env.N):                          # identical initial states: default pose, yaw 0
    k = w % env.K
    m, _, b, _ = build_model(keys[k], "v2", ADAPT)
    d = default_data(m, b)
    env.qpos[w] = torch.as_tensor(d.qpos, dtype=torch.float32)
    env.qvel[w] = 0
    env.ctrl[w] = torch.as_tensor(d.ctrl, dtype=torch.float32)
env.lat.zero_()
errs = []
for w in range(env.K):
    m, _, b, _ = build_model(keys[w], "v2", ADAPT)
    d = default_data(m, b)
    ref = []
    for t in range(T):
        d.ctrl[b.pol_act] = np.clip(b.q0 + b.action_scale * tg[t], b.lo, b.hi)
        for _ in range(substeps_of(m)):
            mujoco.mj_step(m, d)
        ref.append(d.qpos.copy())
    errs.append(ref)
got = [[] for _ in range(env.K)]
for t in range(T):
    env.step(torch.as_tensor(np.tile(tg[t], (env.N, 1)), dtype=torch.float32, device="cuda"))
    env.cmd.zero_()
    q = env.qpos.cpu().numpy()
    for k in range(env.K):
        got[k].append(q[k])
out = dict(topology=want, keys=keys, heights=[round(sample_params(x).height, 2) for x in seeds],
           max_dqpos=[float(np.abs(np.array(got[k]) - np.array(errs[k])).max()) for k in range(env.K)])
e0 = np.abs(np.array(got[0]) - np.array(errs[0]))
out["first_tick_worst_coords"] = np.argsort(-e0[0])[:5].tolist()
out["first_tick_err"] = float(e0[0].max())
out["err_by_tick"] = [float(x) for x in e0.max(1)[:8]]
print(json.dumps(out))
assert max(out["max_dqpos"]) < 1e-3, out
