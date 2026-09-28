"""W12 diagnostic (privileged): where does a dual teacher's commanded joint jerk peak? One episode per option set;
prints per arm the jerk RMS and the ticks/phases of the 5 largest |d3q_cmd/dt3| values.
  python scripts/w12/jerk_diag.py handover panda_pg2__ur5e_pg2 0 '{"limit_aware": false}'"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from rrp.envs.motion_quality import finite_diff  # noqa: E402
from rrp.teachers.dual_smooth import make_dual_teacher  # noqa: E402
from rrp.teachers.dual_validate import make_session  # noqa: E402

task, pair, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
opts = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
ver = sys.argv[5] if len(sys.argv) > 5 else "v3"
s = make_session(task, pair, seed)
t = make_dual_teacher(task, s, ver, opts if ver == "v3" else None)
q = {e: [] for e in s.handles}
ph = []
for k in range(int(sys.argv[6]) if len(sys.argv) > 6 else 700):
    c = t.act()
    for e, h in s.handles.items():
        q[e].append(np.asarray(c[h.robot].groups[h.arm_group], float))
    ph.append(t.phase_label)
    s.step(c)
    if t.done:
        break
out = dict(task=task, pair=pair, seed=seed, opts=opts, ver=ver, success=bool(s.privileged_success()), steps=len(ph))
for e in q:
    j = np.abs(finite_diff(np.array(q[e]), s.dt, 3)).max(axis=1)
    v = np.abs(finite_diff(np.array(q[e]), s.dt, 1)).max(axis=1)
    top = np.argsort(j)[-5:][::-1]
    out[e] = dict(jerk_rms=float(np.sqrt(np.mean(finite_diff(np.array(q[e]), s.dt, 3) ** 2))),
                  vmax=float(v.max()), vsat=int((v > 2.9).sum()),
                  top=[(int(i), round(float(j[i]), 1), ph[min(i + 1, len(ph) - 1)]) for i in top])
print(json.dumps(out))
