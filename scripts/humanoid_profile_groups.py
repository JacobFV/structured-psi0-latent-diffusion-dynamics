"""W13 P1c profiling (peer, GPU): per-group step time of MorphMultiEnv vs a single-body env. Prints JSON."""
import json, sys, time
import torch
from rrp.envs.warp_tracker_env import MorphMultiEnv, WarpTrackerEnv
from rrp.training.humanoid_recipes import SHARED_POOL_V1, phum_groups

def timeit(env, n=10, act_dim=None):
    a = torch.zeros(env.N, act_dim or env.nA, device="cuda")
    env.step(a); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        env.step(a)
    torch.cuda.synchronize()
    return (time.time() - t0) / n

out = {}
for k in ["t1", "phum"]:
    if k == "t1":
        e = WarpTrackerEnv("t1", 512)
    else:
        e = WarpTrackerEnv(phum_groups()[0][0][0], 512) if False else WarpTrackerEnv(phum_groups()[0][0], 512)
    out[f"single_{k}_512_s_per_tick"] = timeit(e)
    # break down: physics only
    t0 = time.time()
    for _ in range(10):
        for kk in range(e.substeps):
            e.wp.capture_launch(e.graph, stream=e.stream)
    torch.cuda.synchronize()
    out[f"single_{k}_physics_s_per_tick"] = (time.time() - t0) / 10
    t0 = time.time()
    for _ in range(10):
        e._stance()
    torch.cuda.synchronize()
    out[f"single_{k}_stance_s"] = (time.time() - t0) / 10
print(json.dumps(out), flush=True)
m = MorphMultiEnv([[[b], 512] for b in SHARED_POOL_V1] + phum_groups())
out["multi_s_per_tick"] = timeit(m)
print(json.dumps(out))
