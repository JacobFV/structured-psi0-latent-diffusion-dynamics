"""Run the v1 teacher for N ticks and list robot contacts (W7 diagnosis of blocked descents)."""
import sys
import mujoco
from rrp.bodies.catalog import workbench_robots
from rrp.envs.native import Session
from rrp.envs.scenario import BUILDERS
from rrp.teachers.arm import PickPlaceTeacher
rk, sd, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = Session(BUILDERS["pick_place"](workbench_robots()[rk](), sd, n_distractors=sd % 3), seed=sd)
t = PickPlaceTeacher(s)
for k in range(n):
    s.step(t.act())
m, d = s.model, s.data
print("phase", t.phase)
for c in range(d.ncon):
    con = d.contact[c]
    b1, b2 = m.body(m.geom_bodyid[con.geom1]).name, m.body(m.geom_bodyid[con.geom2]).name
    print(b1, m.geom(con.geom1).name, "|", b2, m.geom(con.geom2).name, "dist %.4f" % con.dist)
for o in ("cube", "distractor0", "distractor1"):
    try:
        print(o, d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o)])
    except Exception:
        pass
