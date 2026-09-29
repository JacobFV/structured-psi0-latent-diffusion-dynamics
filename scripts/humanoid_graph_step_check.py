"""W13 dev (peer, GPU): CUDA-graphed env.step vs eager: speed and sanity (finite, episodes/falls accumulate). Prints JSON."""
import json, time, sys
import torch
from rrp.envs.warp_tracker_env import GraphedStep, MorphMultiEnv, WarpTrackerEnv
from rrp.training.humanoid_recipes import SHARED_POOL_V1, phum_groups
out = {}
def bench(env, tag, n=20):
    a = torch.randn(env.N, env.nA, device="cuda") * 0.2
    for _ in range(3): env.step(a)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): env.step(a)
    torch.cuda.synchronize(); out[f"{tag}_eager"] = (time.time() - t0) / n
    gs = GraphedStep(env)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): o = gs(a)
    torch.cuda.synchronize(); out[f"{tag}_graph"] = (time.time() - t0) / n
    out[f"{tag}_finite"] = bool(torch.isfinite(o[0]).all())
    for _ in range(200): gs(a)
    st = env.pop_stats(); out[f"{tag}_episodes_200"] = st["episodes"]; out[f"{tag}_falls"] = st["falls"]
bench(WarpTrackerEnv("t1", 1024), "t1_1024")
print(json.dumps(out), flush=True)
bench(MorphMultiEnv([[[b], 1024] for b in SHARED_POOL_V1] + phum_groups(2, 32, 2048)), "shared")
print(json.dumps(out))
