"""Is the re-anchored scripted teacher itself competent? (validity check for the ladder's re-anchored oracle route)
Runs PickPlaceTeacher on matched dev seeds; every K ticks its internal TCP reference and IK seed are reset to the
MEASURED arm (as ladder.ShadowTeacher.reanchor_now does). K=0: no re-anchoring (plain teacher). PRIVILEGED teacher."""
import json, sys
import numpy as np
from rrp.morphology.catalog import workbench_robots
from rrp.sim.scenario import BUILDERS
from rrp.sim.native import Session
from rrp.control.teachers import PickPlaceTeacher

robot, K, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows, sd = [], 3_000_000
while len(rows) < n:
    s = Session(BUILDERS["pick_place"](workbench_robots()[robot](), sd, n_distractors=sd % 3), seed=sd)
    t = PickPlaceTeacher(s)
    if not t.feasibility()["feasible"]:
        sd += 1; continue
    for k in range(300):
        if K and k % K == 0 and k > 0:
            t.tcp_cmd = s._fk_site(t.r, t.tcp_site)[0].copy()
            t.q_arm = s.data.qpos[t.r.qadr[:len(t.q_arm)]].copy()
        s.step(t.act())
        if t.done or s.runtime.succeeded():
            break
    rows.append(dict(seed=sd, success=bool(s.privileged_success()), phase=t.phase, steps=k + 1))
    sd += 1
res = dict(robot=robot, reanchor_every=K, n=len(rows), success=sum(r["success"] for r in rows),
           final_phases={p: sum(r["phase"] == p for r in rows) for p in {r["phase"] for r in rows}}, rows=rows)
open(out, "w").write(json.dumps(res, indent=1)); print(robot, K, res["success"], "/", res["n"], res["final_phases"])
