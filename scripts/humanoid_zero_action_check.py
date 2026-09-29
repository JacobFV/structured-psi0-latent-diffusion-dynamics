"""W13 diag (peer): zero-action (default-pose PD hold) survival in the warp env vs the C LeggedEnv, same body. Prints JSON."""
import json
import sys

import numpy as np
import torch

body = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 150
from rrp.envs.warp_tracker_env import WarpTrackerEnv
env = WarpTrackerEnv(body, 256, seed=5, push=False)
env.cmd.zero_()
env.cmd_timer.fill_(10_000)
first_done = torch.full((env.N,), T, device="cuda")
for t in range(T):
    _, _, r, d, _ = env.step(torch.zeros(env.N, env.nA, device="cuda"))
    env.cmd.zero_()
    first_done = torch.where(d & (first_done == T), torch.full_like(first_done, t), first_done)
w = dict(warp_mean_first_done=float(first_done.float().mean()), warp_frac_survive=float((first_done == T).float().mean()))
from functools import partial
from rrp.bodies.legged import legged_body
from rrp.envs.legged_core import LeggedEnv
ce = LeggedEnv(partial(legged_body, body), 16, 5, contact="v2", push=False)
fd = np.full(16, T)
for t in range(T):
    ce.cmd[:] = 0
    ce.cmd_timer[:] = 10_000
    _, _, _, d, _ = ce.step(np.zeros((16, ce.b.n)))
    fd = np.where(d & (fd == T), t, fd)
w.update(c_mean_first_done=float(fd.mean()), c_frac_survive=float((fd == T).mean()), body=body, ticks=T)
print(json.dumps(w))
