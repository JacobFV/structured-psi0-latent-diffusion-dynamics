"""W12 phase-B diagnostic (research, privileged): why does the support_insert peg leave a parallel grip under
grasp_v2.1? Runs ONE scripted_teacher episode and logs, per control tick of the right arm from r_close until the peg
is lost or r_align, the grip actuator force, the pad<->peg contacts (count, summed normal force, max tangential/normal
ratio), the peg position along the tool axis and the peg tilt relative to the tool.

  python scripts/w12/grip_diag.py --pair parm5_pg2__parm5_pg2 --seed 0 --out artifacts/runs/w12_dualaudit/grip_diag.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="parm5_pg2__parm5_pg2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from rrp.physics.grasp_contact import model_grasp_version
    from rrp.teachers.dual import TEACHERS
    from rrp.teachers.dual_validate import make_session
    s = make_session("support_insert", a.pair, a.seed)
    t = TEACHERS["support_insert"](s)
    m, d = s.model, s.data
    h = s.handles["right"]
    peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
    names = {l.name for l in s.robots[h.robot].spec.links}
    rb = {b for b in range(m.nbody) if m.body(b).name in names}
    grip_act = [i for i in range(m.nu) if (m.actuator(i).name or "").startswith(h.prefix) and
                m.actuator(i).name.endswith("act_grip")]
    rows = []
    f6 = np.zeros(6)
    started = False
    for k in range(a.max_steps):
        cmds = t.act()
        s.step(cmds)
        ph = t.phase["right"]
        if ph == "r_close":
            started = True
        if not started:
            continue
        n_c, fn, ratio, geoms = 0, 0.0, 0.0, set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if not ((b1 == peg and b2 in rb) or (b2 == peg and b1 in rb)):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            n_c += 1
            fn += max(f6[0], 0.0)
            ratio = max(ratio, float(np.hypot(f6[1], f6[2]) / max(f6[0], 1e-9)))
            geoms.add(m.geom(c.geom2 if b1 == peg else c.geom1).name)
        tcp, R = s.tcp_pose("right")
        pp, pR = d.xpos[peg].copy(), d.xmat[peg].reshape(3, 3)
        rows.append(dict(k=k, t=round(float(d.time), 3), phase=ph, grip_cmd=float(t.arms["right"].grip),
                         grip_force=[float(d.actuator_force[i]) for i in grip_act],
                         grip_q=[float(d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]) for i in grip_act],
                         n_contacts=n_c, normal_force_N=round(fn, 3), max_tan_over_normal=round(ratio, 3),
                         contact_geoms=sorted(geoms),
                         peg_along_tool_m=round(float((pp - tcp) @ R[:, 2]), 4),
                         peg_tilt_deg=round(math.degrees(math.acos(abs(float(pR[:, 2] @ R[:, 2])))), 2),
                         held_truth="peg" in s.truth().held_by.get("right", [])))
        if ph in ("r_align", "r_insert") or (rows[-1]["n_contacts"] == 0 and len(rows) > 30):
            break
    out = dict(pair=a.pair, seed=a.seed, grasp_contact_version=model_grasp_version(m) or "grasp_v1",
               source="scripted_teacher (privileged diagnostic)", grasp_depth=t.grasp_depth,
               closed_value=h.closed_value, open_value=h.open_value, rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=0))
    for r in rows[::5]:
        print(r)


if __name__ == "__main__":
    main()
