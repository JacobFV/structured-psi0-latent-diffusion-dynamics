"""Per-tick trace of one teacher episode (W7 diagnosis): phase, tcp, object rel. tcp, contacts (bodies, dist), grip."""
import sys, json
import numpy as np, mujoco
from rrp.evaluation.teacher_quality import run_quality_episode
rk, sd, ver = sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "v1"
row = run_quality_episode(rk, sd, ver, keep_trace=True)
T = row.pop("_trace", None)
print(json.dumps({k: v for k, v in row.items() if k not in ("feasibility",)}, default=float)[:1500])
if T:
    for k in range(len(T["phase"])):
        rel = T["tcp_R"][k].T @ (T["obj"][k] - T["tcp"][k])
        print(k, T["phase"][k], "q_cmd", np.round(T["q_cmd"][k], 2), "q", np.round(T["q"][k], 2), "grip %.3f" % T["grip"][k],
              "tcp", np.round(T["tcp"][k], 3), "cmdfk", np.round(T["tcp_cmd_fk"][k], 3), "rel", np.round(rel, 3),
              "pen %.3f" % T["pen_hand"][k], "nc", T["n_hand_contacts"][k], "ik %.4f" % T["ik"][k])
