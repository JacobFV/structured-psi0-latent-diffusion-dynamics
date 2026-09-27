"""Debug a v2 approach plan: per-tick joint velocity of the IK chain before stretching (W7)."""
import sys
import numpy as np
from rrp.bodies.catalog import workbench_robots
from rrp.envs.native import Session
from rrp.envs.scenario import BUILDERS
from rrp.teachers import arm_smooth as A
rk, sd = sys.argv[1], int(sys.argv[2])
s = Session(BUILDERS["pick_place"](workbench_robots()[rk](), sd, n_distractors=sd % 3), seed=sd)
t = A.SmoothPickPlaceTeacher(s)
print("limits v", np.round(t.v_joint, 2), "a", np.round(t.a_joint, 2), "a_act", t.diag["a_actuator"], t.diag["limits_error"])
print("plans", t.diag["plans"], "yaw", t.diag["yaw_choice"]); sys.exit(0)
p0, y0 = t._cart_start()
_, R0 = t.r.ik.fk(s.data.qpos.copy(), t.q_arm)
print("start tcp", np.round(p0, 3), "yaw0", round(y0, 3), "tool z axis", np.round(R0[:, 2], 3))
_, _, hover, grasp = t._grasp_targets()
plan = A.CartPlan(p0, y0)
plan.add(hover, t.yaw, 1.5, 0.0, "pregrasp")
ts = [(k + 1) * s.dt for k in range(int(1.5 / s.dt))]
pts, yaws = zip(*[plan.at(x) for x in ts])
qs, worst = t._ik_chain(t.q_arm.copy(), pts, yaws)
Q = np.vstack([t.q_arm, *qs])
v = np.diff(Q, axis=0) / s.dt
print("worst", worst)
for k in range(len(v)):
    print(k, np.round(v[k], 2), np.round(Q[k + 1], 2))
