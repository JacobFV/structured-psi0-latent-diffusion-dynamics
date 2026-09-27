"""Fall frequency of a tracker for one command from the renderer's initial condition (yaw 0, joint noise 0.03, seeds s0..s0+n-1).
usage: [RRP_ACTUATOR_LIMITS=..] [RRP_ALLOW_LIMITS_MISMATCH=1] python scripts/contact_fall_rate.py <body> <actor> <vx> <wz> <T> <s0> <n> <out.json>"""
import json, sys, warnings
import mujoco, numpy as np
warnings.filterwarnings("ignore")
from rrp.bodies.legged import legged_body, standalone_model
from rrp.envs.legged_core import LeggedBinding
from rrp.envs.legged_tracker import LearnedTracker
body, actor, vx, wz, T, s0, n, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
m, _, meta = standalone_model(legged_body(body), contact="v2"); b = LeggedBinding(m, meta); tr = LearnedTracker(actor, b, body)
rows = []
for seed in range(s0, s0 + n):
    rng = np.random.default_rng(seed); d = mujoco.MjData(m); b.set_default(d, yaw=0.0, noise=0.03, rng=rng); mujoco.mj_forward(m, d); tr.reset(0.0)
    fell_t = None
    for k in range(int(T / 0.02)):
        d.ctrl[b.pol_act] = tr.act(d, np.array([vx, 0, wz]))
        for _ in range(10):
            mujoco.mj_step(m, d)
        fc, bad = b.contacts(d)
        if bad or d.qpos[b.qa + 2] < b.min_h or b.tilt(d) > b.tilt_limit:
            fell_t = k * 0.02; break
    rows.append(dict(seed=seed, fell_t=fell_t))
res = dict(body=body, actor=actor, cmd=[vx, 0, wz], T=T, n=n, falls=sum(r["fell_t"] is not None for r in rows), actuator_limits=meta.get("actuator_limits"), rows=rows)
json.dump(res, open(out, "w"), indent=1); print({k: v for k, v in res.items() if k != "rows"})
