"""Why do DART (exec-noise 0.08) episodes succeed less often with teacher v2 than v1? (W7 step 2 diagnosis)"""
import collections, sys
from rrp.bodies.catalog import workbench_robots
from rrp.envs.native import Session
from rrp.envs.scenario import BUILDERS
from rrp.data.collect import collect_teacher_episode
rk = sys.argv[1]; n = int(sys.argv[2])
robot = workbench_robots()[rk]()
for ver in ("v1", "v2"):
    c = collections.Counter(); ev = collections.Counter(); steps = []
    for sd in range(n):
        s = Session(BUILDERS["pick_place"](robot, sd, n_distractors=sd % 3), seed=sd)
        holder = {}
        from rrp.teachers import arm_smooth
        orig = arm_smooth.make_arm_teacher
        def mk(sess, v, **kw):
            t = orig(sess, v, **kw); holder["t"] = t; return t
        import rrp.data.collect as C
        arm_smooth.make_arm_teacher = mk
        rec = collect_teacher_episode(s, exec_noise=0.08, noise_seed=sd, teacher_version=ver)
        arm_smooth.make_arm_teacher = orig
        st = rec.public["meta"]["status"]; c[st] += 1; steps.append(rec.public["meta"]["steps"])
        t = holder.get("t")
        ph = rec.private["phases"][-1] if rec.private["phases"] else None
        ev[(st, ph, tuple(e["what"].split(" ")[0] for e in (getattr(t, "diag", {}) or {}).get("events", [])))] += 1
    print(rk, ver, dict(c), "mean steps", sum(steps) / len(steps))
    for k, v in ev.most_common(8):
        print("   ", v, k)
