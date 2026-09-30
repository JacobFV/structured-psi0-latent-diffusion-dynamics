"""W13 P2 smoke (peer GPU): WarpGapEnv at level 1 with a tracker actor: wall contacts, successes, finite obs."""
import json, sys
import torch
from rrp.envs.warp.task_env import WarpGapEnv
from rrp.envs.mujoco.tracker_nets import mlp
body, actor = sys.argv[1], sys.argv[2]
st = torch.load(actor, map_location="cuda", weights_only=False); m = st["meta"]
net = mlp(m["obs_dim"], tuple(m["hidden"]), m["act_dim"]).cuda(); net.load_state_dict(st["actor"])
mean, std = st["obs_mean"].cuda(), (st["obs_var"].cuda() + 1e-8).sqrt()
out = {}
for lv in (0.0, 1.0):
    e = WarpGapEnv(body, 256, seed=5, level=lv, clock_gate=bool(m.get("clock_gate")), push=False)
    for t in range(1000):
        o = e.observe()[:, :m["obs_dim"]]
        e.step(net(((o - mean) / std).clamp(-5, 5)))
    s = e.pop_stats()
    out[f"level{lv}"] = dict(episodes=s["episodes"], falls=s["falls"], successes=s["successes"], obs_dim=e.obs_dim,
                             gap_w_over_bw=float((e.gap_w / e.bw).mean()), finite=bool(torch.isfinite(e.observe()).all()))
print(json.dumps(out))
