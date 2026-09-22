"""Privileged scripted-teacher reference on evaluation scenes (labelled; not a learned result)."""
import json, sys
from rrp.morphology.catalog import workbench_robots
from rrp.sim.scenario import build_pick_place
from rrp.sim.native import Session
from rrp.control.teachers import PickPlaceTeacher, run_teacher_episode
robots = sys.argv[1].split(","); n = int(sys.argv[2]); seed0 = int(sys.argv[3]); out = sys.argv[4]
R = workbench_robots()
summary = {}
with open(out, "w") as f:
    for rk in robots:
        robot = R[rk](); c = {"success": 0, "failure": 0, "infeasible": 0}
        for sd in range(seed0, seed0 + n):
            s = Session(build_pick_place(robot, sd, n_distractors=sd % 3), seed=sd)
            r = run_teacher_episode(s, PickPlaceTeacher(s), 600)
            k = "infeasible" if (r.failure_reason or "").startswith("infeasible") else ("success" if r.privileged_evaluator_success else "failure")
            c[k] += 1
            f.write(json.dumps(dict(robot=rk, seed=sd, outcome=k, source="scripted_teacher", privileged=True, steps=r.steps)) + "\n")
        summary[rk] = c; print(rk, c, flush=True)
json.dump(summary, open(out.replace(".jsonl", ".summary.json"), "w"), indent=1)
