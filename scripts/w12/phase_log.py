"""W12 diagnostic: phase transitions and final statuses of one dual teacher episode (privileged).
  python scripts/w12/phase_log.py handover panda_pg2__ur5e_pg2 0 v3"""
import json
import sys

sys.path.insert(0, "src")
from rrp.teachers.dual_smooth import make_dual_teacher  # noqa: E402
from rrp.teachers.dual_validate import make_session  # noqa: E402

task, pair, seed, ver = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
s = make_session(task, pair, seed)
t = make_dual_teacher(task, s, ver)
for k in range(900):
    s.step(t.act())
    if t.done:
        break
print(json.dumps(dict(steps=k + 1, success=bool(s.privileged_success()), log=[(round(x["t"], 2), x["arm"], x["frm"], x["to"]) for x in t.log],
                      statuses={e: v.status for e, v in s.runtime.instances.items()}, limits=getattr(t, "limits", None))))
