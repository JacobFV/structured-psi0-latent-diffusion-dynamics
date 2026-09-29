"""W13 diag (peer GPU): run a trained tracker actor (deterministic mean, extra inputs zero-padded) in WarpStepsEnv at a level
and in the plain WarpTrackerEnv under the same scripted forward command; report fall times and progress."""
import json, sys
import torch
from rrp.envs.tracker_nets import mlp
from rrp.envs.warp_task_env import WarpStepsEnv
from rrp.envs.warp_tracker_env import WarpTrackerEnv
body, actor, level = sys.argv[1], sys.argv[2], float(sys.argv[3])
st = torch.load(actor, map_location="cuda", weights_only=False)
m = st["meta"]
net = mlp(m["obs_dim"], tuple(m["hidden"]), m["act_dim"]).cuda()
net.load_state_dict(st["actor"])
mean, std = st["obs_mean"].cuda(), (st["obs_var"].cuda() + 1e-8).sqrt()
out = {}
from rrp.envs.warp_task_env import task_adapted_model
chf = float([a for a in sys.argv if a.startswith("chf=")][0][4:]) if any(a.startswith("chf=") for a in sys.argv) else 0.15
envs = [("steps", WarpStepsEnv(body, 256, seed=4, level=level, clock_gate=bool(m.get("clock_gate")), push=False)),
        ("flat", WarpTrackerEnv(body, 256, seed=4, clock_gate=bool(m.get("clock_gate")), push=False)),
        ("static", WarpTrackerEnv(body, 256, seed=4, clock_gate=bool(m.get("clock_gate")), push=False,
                                  model_fn=lambda k: task_adapted_model(k, "h_steps", {"h_frac": 0.0}),
                                  extra_batch=("geom_pos", "geom_size", "geom_aabb", "geom_rbound")))]
for tag, env in envs:
    first = torch.full((env.N,), 500.0, device="cuda")
    if tag in ("static", "flat") or ("yaw0" in sys.argv):   # face +x from the origin
        env.qpos[:, env.qa:env.qa + 2] = 0.0
        env.qpos[:, env.qa + 3] = 1.0
        env.qpos[:, env.qa + 4:env.qa + 7] = 0.0
    if "nomove" in sys.argv and tag == "steps":
        env.qpos[:, env.qa] = 0.0
        env.qpos[:, env.qa + 1] = 0.0
    for t in range(500):
        if tag in ("flat", "static") or "fwd" in sys.argv:
            env.cmd[:, 0] = 0.6 * env.rng_cmd[:, 0, 1]; env.cmd[:, 1:] = 0; env.cmd_timer.fill_(1e6)
        o = env.observe()[:, :m["obs_dim"]]
        a = net(((o - mean) / std).clamp(-5, 5))
        orig_reset = env._reset
        cap = {}
        def rec_reset(mask, _o=orig_reset, _e=env):
            if mask.any():
                fc, fn, sl, bad = _e._stance()
                g = _e._gravity_body()
                h = _e.qpos[:, _e.qa + 2]
                tilt = torch.acos((-g[:, 2]).clamp(-1, 1))
                cap["bad"] = cap.get("bad", 0) + int((bad & mask).sum()); cap["low"] = cap.get("low", 0) + int(((h < _e.min_h) & mask).sum())
                cap["tilt"] = cap.get("tilt", 0) + int(((tilt > _e.tilt_limit) & mask).sum()); cap["x"] = cap.get("x", []) + _e.qpos[mask, _e.qa].tolist()[:5]
            return _o(mask)
        env._reset = rec_reset
        _, _, _, d, _ = env.step(a)
        env._reset = orig_reset
        for k, v in cap.items():
            out.setdefault(tag + "_why", {})
            out[tag + "_why"][k] = (out[tag + "_why"].get(k, 0) + v) if k != "x" else (out[tag + "_why"].get(k, []) + v)[:12]
        first = torch.where(d & (first == 500), torch.full_like(first, t), first)
    s = env.pop_stats()
    ov = env.dw.overflow.numpy() if hasattr(env.dw, "overflow") else None
    out[tag + "_contacts"] = dict(nacon=int(env.nacon[0]), per_world=float(env.nacon[0]) / env.N, naconmax=int(env.c_geom.shape[0]),
                                  overflow=(ov.tolist() if ov is not None else None), ngeom=int(env.m.ngeom))
    out[tag] = dict(mean_first_done=float(first.mean()), frac_no_done=float((first == 500).float().mean()), stats={k: s[k] for k in ("episodes", "falls", "successes") if k in s})
print(json.dumps(out))
