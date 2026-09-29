"""W13 P2 smoke (peer, GPU): WarpStepsEnv geometry + scan sanity at level 1, a few random ticks. Prints JSON."""
import json, sys
import torch
from rrp.envs.warp_task_env import WarpStepsEnv
e = WarpStepsEnv(sys.argv[1] if len(sys.argv) > 1 else "t1", 64, seed=2, level=1.0)
e._reset(torch.ones(e.N, dtype=torch.bool, device="cuda"))
x = e.extra_obs()
for _ in range(20):
    o, p, r, d, t = e.step(torch.zeros(e.N, e.nA, device="cuda"))
print(json.dumps(dict(obs_dim=e.obs_dim, obs=list(o.shape), h_over_L=[round(float(v), 3) for v in (e.h / e.L)[:5]],
                      x_end=[round(float(v), 2) for v in e.x_end[:3]], scan0=[round(float(v), 3) for v in x[0, :11]],
                      step0_pos=e.mocap_pos[0, e.step_mocap[1]].tolist(), finite=bool(torch.isfinite(o).all()),
                      stats=e.pop_stats()["episodes"])))
