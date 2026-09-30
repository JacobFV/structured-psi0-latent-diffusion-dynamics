"""W7 step 2: motion smoothness of BC rollouts (v1-data vs v2-data experts) next to the two teachers, same bodies and seeds
(3,000,000-3,000,039; feasible only). argv: out.json then jsonl files (policy rows from teacher_quality --policy, teacher
rows from teacher_quality --versions)."""
import collections
import json
import sys

import numpy as np

out, files = sys.argv[1], sys.argv[2:]
rows = []
for f in files:
    for l in open(f):
        r = json.loads(l)
        if 3000000 <= r["seed"] < 3000040 and r["robot"] in ("panda_pg2", "parm6_tf3", "parm5s_tf3", "parm5l_pg2"):
            rows.append(r)
g = collections.defaultdict(list)
for r in rows:
    if r.get("feasible"):
        g[(r["source"], r["robot"])].append(r)
K = [("joint_cmd_jerk_peak", "{:.0f}"), ("joint_cmd_jerk_rms", "{:.0f}"), ("joint_meas_jerk_peak", "{:.0f}"),
     ("tcp_meas_jerk_peak", "{:.1f}"), ("jump", "{:.2f}"), ("time_s", "{:.1f}")]
res = {}
print("| source | body | success | jerk cmd peak med | jerk cmd rms med | jerk meas peak med | TCP jerk meas med (m/s^3) | max joint-velocity step (rad/s per tick) med | time med (s) |")
print("|---|---|---|---|---|---|---|---|---|")
for (src, b), rs in sorted(g.items()):
    e = dict(n=len(rs), success=sum(r["success"] for r in rs))
    for k, _ in K:
        kk = "vel_jump_any_max" if k == "jump" else k
        v = np.array([r[kk] for r in rs if r.get(kk) is not None], float)
        e[k] = float(np.median(v)) if v.size else None
    res[f"{src}|{b}"] = e
    print(f"| {src} | {b} | {e['success']}/{e['n']} | " + " | ".join(
        (fmt.format(e[k]) if e[k] is not None else "-") for k, fmt in K) + " |")
json.dump(res, open(out, "w"), indent=1)
