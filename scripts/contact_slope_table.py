"""Slope stick/slide + creep table for contact_v1 vs contact_v2 (raw output for research/tracks/contact.md).
usage: PYTHONPATH=src python scripts/contact_slope_table.py artifacts/runs/contact_v2/slope_table.json"""
import json, math, sys, time
import mujoco, numpy as np
sys.path.insert(0, "tests/unit")
from test_contact_model import _slope_model  # noqa: E402


def run(contact, foot, ratio, noslip=0, mu=1.0, T=2.5):
    m = _slope_model(contact, foot, mu)
    m.opt.noslip_iterations = noslip
    ang = math.atan(ratio * mu)
    m.opt.gravity[:] = [9.81 * math.sin(ang), 0, -9.81 * math.cos(ang)]
    d = mujoco.MjData(m)
    v, pen = [], 0.0
    t0 = time.time()
    for _ in range(int(T / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.time > 0.5:
            v.append(d.qvel[0]); pen = max([pen] + [-d.contact[i].dist for i in range(d.ncon)])
    return dict(contact=contact, foot=foot, tan_over_mu=ratio, noslip=noslip, creep_mps=float(np.mean(v)),
                max_pen_mm=pen * 1e3, us_per_step=(time.time() - t0) / (T / m.opt.timestep) * 1e6)


rows = [run(c, f, r, ns) for f in ("box", "stool") for c, ns in (("v1", 0), ("v2", 0), ("v2", 5))
        for r in (0.5, 0.9, 1.2)]
for r in rows:
    print(r)
json.dump(dict(mujoco=mujoco.__version__, rows=rows), open(sys.argv[1], "w"), indent=1)
