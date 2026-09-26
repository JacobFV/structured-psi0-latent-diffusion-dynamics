"""Paired edit effects vs the unedited control on the same seeds, in the window [t_edit, end].
Per episode (body frame at t_edit): forward / lateral displacement, yaw change, path length, per-leg contact fraction.
Effect = edited - control (paired by seed); mean and bootstrap 95% CI. Also mean |dz| of the edited packets."""
import json, math, sys
from pathlib import Path
import numpy as np

def feats(r):
    te = r["t_edit"] or 0.0
    tr = [x for x in r["trace"] if x["t"] >= te - 1e-6]
    if len(tr) < 2:
        return None
    x0, y0, a0 = tr[0]["pose"]; x1, y1, a1 = tr[-1]["pose"]
    c, s = math.cos(a0), math.sin(a0)
    dx, dy = x1 - x0, y1 - y0
    path = sum(math.hypot(b["pose"][0] - a["pose"][0], b["pose"][1] - a["pose"][1]) for a, b in zip(tr, tr[1:]))
    con = np.array([x["contact"] for x in tr], float).mean(0)
    dz = [p.get("dz_norm", 0.0) for p in r["packets"] if p["t"] >= te - 1e-6]
    return dict(forward=c * dx + s * dy, lateral=-s * dx + c * dy,
                dyaw=(a1 - a0 + math.pi) % (2 * math.pi) - math.pi, path=path, contact=con,
                dz=float(np.mean(dz)) if dz else 0.0, fell=r["fell"])

def ci(v, rng=np.random.default_rng(0)):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

d = Path(sys.argv[1])
rows = {f.stem: {r["seed"]: r for r in map(json.loads, open(f))} for f in d.glob("*.jsonl")}
ctrl = {s: feats(r) for s, r in rows.get("none", {}).items()}
out = dict(dir=str(d), source=next(iter(rows["none"].values()))["source"] if "none" in rows else None, conditions={})
for name, rs in sorted(rows.items()):
    eff = {k: [] for k in ("forward", "lateral", "dyaw", "path")}
    legs, dzs, fell = [], [], 0
    for s, r in rs.items():
        f, c = feats(r), ctrl.get(s)
        if f is None or c is None:
            continue
        for k in eff:
            eff[k].append(f[k] - c[k])
        legs.append(f["contact"] - c["contact"]); dzs.append(f["dz"]); fell += f["fell"]
    out["conditions"][name] = dict(n=len(eff["path"]), fell=fell, mean_dz=round(float(np.mean(dzs)), 3) if dzs else None,
                                   **{k: ci(v) for k, v in eff.items()},
                                   contact_delta_per_leg=[ci(np.array(legs)[:, m]) for m in range(len(legs[0]))] if legs else None)
(d / "effects.json").write_text(json.dumps(out, indent=1))
for k, v in out["conditions"].items():
    print(k, json.dumps({a: b for a, b in v.items() if a != "contact_delta_per_leg"}), "leg0", (v["contact_delta_per_leg"] or [None])[0])
