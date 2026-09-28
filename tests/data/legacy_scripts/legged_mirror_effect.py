"""Mirror edits: lateral displacement and yaw change TOWARD the mirrored goal side (sign-normalized by the side of the
active waypoint at t_edit in the true body frame; privileged, scoring only), paired vs the unedited run."""
import json, math, sys
from pathlib import Path
import numpy as np

def side_and_eff(r):
    te = r["t_edit"]; tr = [x for x in r["trace"] if x["t"] >= te - 1e-6]
    if len(tr) < 2:
        return None
    x0, y0, a0 = tr[0]["pose"]; x1, y1, a1 = tr[-1]["pose"]
    ev = next((p["ev"] for p in r["packets"] if p["t"] >= te - 1e-6), 0)
    wp = r["waypoints"]["a" if ev == 0 else "b"]
    c, s = math.cos(a0), math.sin(a0)
    gy = -s * (wp[0] - x0) + c * (wp[1] - y0)
    return math.copysign(1.0, gy), -s * (x1 - x0) + c * (y1 - y0), (a1 - a0 + math.pi) % (2 * math.pi) - math.pi

def ci(v, rng=np.random.default_rng(0)):
    v = np.asarray(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

d = Path(sys.argv[1]); res = {}
ctrl = {r["seed"]: r for r in map(json.loads, open(d / "none.jsonl"))}
for name in sys.argv[2:]:
    lat, yaw = [], []
    for r in map(json.loads, open(d / f"{name}.jsonl")):
        a, b = side_and_eff(r), side_and_eff(ctrl[r["seed"]]) if r["seed"] in ctrl else None
        if a is None or b is None:
            continue
        sg = b[0]                                   # original goal side (from the unedited run at t_edit)
        lat.append(-sg * (a[1] - b[1])); yaw.append(-sg * (a[2] - b[2]))
    res[name] = dict(n=len(lat), toward_mirror_lateral_m=ci(lat), toward_mirror_yaw_rad=ci(yaw))
    print(name, res[name])
(d / "mirror_effects.json").write_text(json.dumps(res, indent=1))
