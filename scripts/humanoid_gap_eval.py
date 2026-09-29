"""W13 P2: C-MuJoCo evaluation of an h_gap_sidestep expert (privileged_teacher:rl_expert) or a blind tracker, driven by the
scripted GapTeacher, full self-collision. Privileged evaluator: success = 0.5 L past the wall with |psi_f - yaw| < 0.3 held
0.5 s; failures: fell | wall_collision (> 0.2 s of robot-wall contact) | timeout (30 s).
usage: python scripts/humanoid_gap_eval.py <body> <actor.pt> <seed0> <n> <out.json> [level]"""
import json
import math
import sys
import time

import mujoco
import numpy as np

import rrp.envs.legged as _L
from rrp.envs.humanoid_scenes import build_h_gap, gap_obs_np
from rrp.envs.legged import LeggedSession
from rrp.envs.legged_tracker import LearnedTracker
from rrp.teachers.humanoid import GapTeacher

body, actor, s0, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
level = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
CUR = {}


def _load(key, binding, meta, kind="auto"):
    tr = LearnedTracker(actor, binding, key)
    CUR["expert"] = bool(int(tr.meta.get("extra_obs_dim") or 0))
    if CUR["expert"]:
        tr.extra_fn = lambda d: gap_obs_np(d.qpos[binding.qa:binding.qa + 7], CUR["meta"], CUR["teacher"].phase2 if CUR.get("teacher") else 0.0)
    return tr


_L.load_tracker = _load
rows = []
for seed in range(s0, s0 + n):
    sc = build_h_gap(body, seed, level=level)
    CUR.update(meta=sc.meta, teacher=None)
    s = LeggedSession(sc, tracker_kind="learned", seed=seed)
    s.reset(seed)
    te = GapTeacher(s)
    CUR["teacher"] = te
    m = s.model
    walls = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, w) for w in sc.meta["walls"]}
    wall_t, hold_t, status, t0 = 0.0, 0.0, "timeout", time.time()
    for k in range(300):
        s.step(te.act())
        d = s.data
        hit = any((d.contact[i].geom1 in walls) ^ (d.contact[i].geom2 in walls) for i in range(d.ncon))
        wall_t = wall_t + 0.1 if hit else 0.0
        x, y, yaw = s.base_pose_truth()
        ok = te.phase2 and abs(math.atan2(math.sin(sc.meta["psi_f"] - yaw), math.cos(sc.meta["psi_f"] - yaw))) < 0.3
        hold_t = hold_t + 0.1 if ok else 0.0
        if s.fell:
            status = "fell"
            break
        if wall_t > 0.2:
            status = "wall_collision"
            break
        if hold_t >= 0.5:
            status = "success"
            break
    rows.append(dict(seed=seed, status=status, gap_ratio=sc.meta["gap_ratio"], y_c=sc.meta["y_c"], psi_f=sc.meta["psi_f"],
                     x=float(s.data.qpos[s.binding.qa]), sim_s=float(s.data.time), wall_s=time.time() - t0))
    print(rows[-1], flush=True)
from collections import Counter
summ = dict(body=body, actor=actor, n=n, level=level, source=("privileged_teacher:rl_expert + scripted_teacher command" if CUR.get("expert")
                                                              else "learned_tracker (blind) + scripted_teacher command"),
            success=sum(r["status"] == "success" for r in rows), status=dict(Counter(r["status"] for r in rows)), rows=rows)
json.dump(summ, open(out, "w"), indent=1)
print({k: v for k, v in summ.items() if k != "rows"})
