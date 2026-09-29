import torch, json
from rrp.envs.warp_task_env import WarpStepsEnv, task_adapted_model
from rrp.envs.warp_tracker_env import WarpTrackerEnv
s = WarpStepsEnv("h1", 4, seed=4, level=0.0)
t = WarpTrackerEnv("h1", 4, seed=4, model_fn=lambda k: task_adapted_model(k, "h_steps", {"h_frac": 0.0}),
                   extra_batch=("geom_pos", "geom_size", "geom_aabb", "geom_rbound"))
g = s.step_gids[1]
W = lambda e, f: e.wp.to_torch(getattr(e.mw, f))
print(json.dumps({f: [W(s, f)[0, g].tolist(), W(t, f)[0, g].tolist()] for f in ("geom_pos", "geom_size", "geom_aabb", "geom_rbound")}))
print("robot geom 5 aabb", W(s, "geom_aabb")[0, 20].tolist(), W(t, "geom_aabb")[0, 20].tolist())
print("m aabb row", t.m.geom_aabb[g].tolist(), "size", t.m.geom_size[g].tolist(), "pos", t.m.geom_pos[g].tolist())
