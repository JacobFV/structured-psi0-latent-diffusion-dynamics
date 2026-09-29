"""W13 P2: C-MuJoCo evaluation of an h_steps expert (privileged_teacher:rl_expert with height scan) driven by the scripted
heading command, full self-collision, contact_v2. usage: python scripts/humanoid_steps_eval.py <body> <actor.pt> <seed0> <n> <out.json> [h_frac]"""
import json
import sys
import time

import numpy as np

import rrp.envs.legged as _L
from rrp.envs.humanoid_scenes import build_h_steps
from rrp.envs.legged import LeggedSession
from rrp.envs.legged_tracker import LearnedTracker
from rrp.teachers.humanoid import StepsHeadingTeacher

body, actor, s0, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
hf = float(sys.argv[6]) if len(sys.argv) > 6 else None
CUR = {}


def _load(key, binding, meta, kind="auto"):
    """The expert with its PRIVILEGED scan attached before the session's first (settling) tick."""
    from rrp.envs.humanoid_scenes import steps_scan_np
    tr = LearnedTracker(actor, binding, key)
    if int(tr.meta.get("extra_obs_dim") or 0):
        L, h = CUR["L"], CUR["h"]
        tr.extra_fn = lambda d: steps_scan_np(d.qpos[binding.qa:binding.qa + 7], L, h)
    return tr


_L.load_tracker = _load
rows = []
for seed in range(s0, s0 + n):
    sc = build_h_steps(body, seed, h_frac=hf)
    CUR.update(L=float(sc.meta["L"]), h=float(sc.meta["staircase"]["h"]))
    s = LeggedSession(sc, tracker_kind="learned", seed=seed)   # plain trackers (no scan input) run blind
    s.reset(seed)
    te = StepsHeadingTeacher(s)
    t0, status = time.time(), "timeout"
    for k in range(400):                            # 10 Hz commands, 40 s
        s.step(te.act())
        if s.fell:
            status = "fell"
            break
        if s.runtime.succeeded():
            status = "success"
            break
    x = float(s.data.qpos[s.binding.qa])
    rows.append(dict(seed=seed, status=status, h_frac=sc.meta["h_frac"], x=x, x_end=sc.meta["x_end"], sim_s=float(s.data.time),
                     wall_s=time.time() - t0))
    print(rows[-1], flush=True)
summ = dict(body=body, actor=actor, n=n, source="privileged_teacher:rl_expert + scripted_command",
            success=sum(r["status"] == "success" for r in rows), fell=sum(r["status"] == "fell" for r in rows), rows=rows)
json.dump(summ, open(out, "w"), indent=1)
print({k: v for k, v in summ.items() if k != "rows"})
