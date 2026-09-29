"""W13 diag (peer CPU): per-joint minimum joint-limit margin of a tracker over the validation command scripts (C MuJoCo)."""
import json, sys
import mujoco
import numpy as np
from rrp.bodies.legged import legged_body, standalone_model
from rrp.envs.legged_core import LeggedBinding
from rrp.envs.legged_tracker import LearnedTracker
body, actor = sys.argv[1], sys.argv[2]
m, _, meta = standalone_model(legged_body(body), contact="v2")
b = LeggedBinding(m, meta)
names = meta["legged"]["policy_actuators"]
worst = {n: 1.0 for n in names}
tgt_worst = {n: 1.0 for n in names}
r = b.cmd_ranges
for cmd in ([0, 0, 0], [0.6 * r["vx"][1], 0, 0], [0, 0, 0.6 * r["wz"][1]], [0, 0, 0.8 * r["wz"][1]], [0.5 * r["vx"][1], 0, 0.4 * r["wz"][1]]):
    for seed in range(3):
        tr = LearnedTracker(actor, b, body)
        d = mujoco.MjData(m)
        b.set_default(d, noise=0.03, rng=np.random.default_rng(seed))
        mujoco.mj_forward(m, d)
        for k in range(400):
            t = tr.act(d, np.array(cmd, float))
            d.ctrl[b.pol_act] = t
            for _ in range(10):
                mujoco.mj_step(m, d)
            q = d.qpos[b.pol_qadr]
            span = b.jhi - b.jlo
            mg = np.minimum(q - b.jlo, b.jhi - q) / span
            tm = np.minimum(t - b.jlo, b.jhi - t) / span
            for i, n in enumerate(names):
                worst[n] = min(worst[n], float(mg[i]))
                tgt_worst[n] = min(tgt_worst[n], float(tm[i]))
bad = sorted(worst.items(), key=lambda x: x[1])[:5]
print(json.dumps(dict(body=body, worst_joint_margin=bad, target_margin_of_those={n: tgt_worst[n] for n, _ in bad},
                      ranges={n: [float(b.jlo[i]), float(b.jhi[i]), float(b.q0[i])] for i, n in enumerate(names) if n in dict(bad)})))
