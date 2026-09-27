import json, sys, collections
import numpy as np
for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f)]
    print("==", f, len(R), "succ", sum(r["privileged_success"] for r in R), collections.Counter(r["failed_stage"] for r in R))
    g = lambda k: np.mean([r[k] for r in R if r.get(k) is not None]) if any(r.get(k) is not None for r in R) else None
    print(" lab_err arm %.4f grip %.4f lab_step %.4f track_q %.4f track_tcp %.4f cmd_step %.4f min_d %.3f" % tuple(
        (g(k) or 0) for k in ("lab_err_arm", "lab_err_grip", "lab_step_arm", "track_q_rad", "track_tcp_m", "cmd_step_rad", "min_tcp_cube_m")))
    ph = collections.defaultdict(list)
    for r in R:
        for p, v in r["by_phase"].items():
            ph[p].append(v)
    print(" by teacher phase (n ticks, arm err, grip err):", {p: (sum(x["n"] for x in v), round(np.mean([x["lab_err_arm"] for x in v if x["lab_err_arm"] is not None] or [0]), 4), round(np.mean([x["lab_err_grip"] for x in v if x["lab_err_grip"] is not None] or [0]), 3)) for p, v in ph.items()})
    bj = collections.defaultdict(list)
    for r in R:
        for j, v in r["lab_err_by_j"].items():
            bj[int(j)].append(v)
    print(" arm err by j:", {j: round(np.mean(v), 4) for j, v in sorted(bj.items())})
    oc = [r["oracle_cmp"] for r in R if r.get("oracle_cmp")]
    if oc:
        print(" oracle cmp:", {k: round(np.mean([o[k] for o in oc if k in o]), 4) for k in oc[0]})
    print(" final teacher phases:", collections.Counter(r["final_teacher_phase"] for r in R), "events:", collections.Counter(json.dumps(r["events"]) for r in R).most_common(3))
