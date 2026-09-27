"""Closed-loop W8 check: WaypointTeacher (scripted_teacher, privileged) driving a given t1/legged tracker on waypoint_contact.
usage: RRP_CONTACT_MODEL=v2 [RRP_ACTUATOR_LIMITS=...] python scripts/contact_waypoint_eval.py <body> <actor.pt> <seed0> <n> <out.json>"""
import json, sys, time
import numpy as np
from rrp.envs.legged import LeggedSession, build_waypoint_contact
from rrp.envs.legged_tracker import LearnedTracker
from rrp.teachers.legged import WaypointTeacher

body, actor, s0, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
import rrp.envs.legged as _L
_L.load_tracker = lambda key, binding, meta, kind="auto": LearnedTracker(actor, binding, key)   # evaluate THIS actor
rows = []
for seed in range(s0, s0 + n):
    sc = build_waypoint_contact(body, seed)
    s = LeggedSession(sc, tracker_kind="learned", seed=seed)
    assert s.tracker.meta.get("iter") == LearnedTracker(actor, s.binding, body).meta.get("iter")
    te = WaypointTeacher(s)
    t0, steps, cmds = time.time(), 0, []
    for k in range(1300):
        cmd = te.act()
        cmds.append(cmd.groups["base_velocity"])
        s.step(cmd)
        steps += 1
        if te.done or s.fell:
            break
    c = np.array(cmds)
    status = "success" if s.privileged_success() and not s.fell else ("fell" if s.fell else "failure")
    rows.append(dict(seed=seed, status=status, steps=steps, sim_s=steps * s.dt, pure_turn_frac=float(np.mean((c[:, 0] < 0.05) & (np.abs(c[:, 2]) > 0.05))),
                     contact_model=sc.meta.get("contact_model"), actuator_limits=s.binding.meta.get("actuator_limits")))
    print(rows[-1], flush=True)
summ = dict(body=body, actor=actor, n=n, success=sum(r["status"] == "success" for r in rows), fell=sum(r["status"] == "fell" for r in rows),
            mean_sim_s_success=float(np.mean([r["sim_s"] for r in rows if r["status"] == "success"] or [0])), rows=rows)
json.dump(summ, open(out, "w"), indent=1)
print({k: v for k, v in summ.items() if k != "rows"})
