"""LEGGED FIXED-SEM REPLICATION: fixed sem vs nosem over training seeds 0 (existing runs), 1, 2; per seed and pooled.
Effects are paired per eval seed vs the same model's unedited run (window t=2-5 s), then pooled over (training seed, eval seed).
usage: python scripts/legged_fixrep_compare.py BODY -> artifacts/runs/legged_fixrep_compare_<BODY>.{json,md}"""
import json, math, sys
from pathlib import Path

import numpy as np

body = sys.argv[1]
L, E = Path(f"artifacts/runs/legged_ladder/{body}"), Path(f"artifacts/runs/legged_edits/{body}")
exec(open("scripts/legged_edit_effects.py").read().split("def ci")[0])   # feats(r): forward/lateral/dyaw/... in window


def src(v, s):
    """(r2 ladder tags, z-edit dirs, ctx dir) for variant v in {fixsem, nosem, sem_orig} and training seed s."""
    if s == 0:
        ed = {"fixsem": "fixsem", "nosem": "nosem", "sem_orig": "sem"}[v]
        lt = {"fixsem": f"fixsem_{body}_lv4", "nosem": f"nosem_{body}_v2", "sem_orig": f"sem_{body}_v2"}[v]
        return ([f"r2_{lt}_snap_s4000", f"r2_{lt}_policy"], [E / f"r2_{ed}_snap_s4000", E / f"r2_{ed}_snap_s4000_bigrand"],
                E / f"r2ctx_{ed}_snap_s4000")
    t = f"{v}_{body}_s{s}"
    return ([f"r2_fixrep_{t}_snap_s4000", f"r2_fixrep_{t}_policy"], [E / f"r2_fixrep_{t}_snap_s4000"], E / f"r2ctx_fixrep_{t}_snap_s4000")


def load(p):
    return {r["seed"]: r for r in map(json.loads, open(p))} if p.exists() else None


def side_lat(r):
    te = r["t_edit"]; tr = [x for x in r["trace"] if x["t"] >= te - 1e-6]
    if len(tr) < 2:
        return None
    x0, y0, a0 = tr[0]["pose"]; x1, y1, a1 = tr[-1]["pose"]
    ev = next((p["ev"] for p in r["packets"] if p["t"] >= te - 1e-6), 0)
    wp = r["waypoints"]["a" if ev == 0 else "b"]; c, s_ = math.cos(a0), math.sin(a0)
    return math.copysign(1.0, -s_ * (wp[0] - x0) + c * (wp[1] - y0)), -s_ * (x1 - x0) + c * (y1 - y0)


def paired(dirs, cond, metric):
    """list of per-eval-seed effects of `cond` vs 'none' in the first dir holding cond."""
    for d in dirs:
        a, n = load(d / f"{cond}.jsonl"), load(d / "none.jsonl")
        if a is None or n is None:
            continue
        out = []
        for sd, r0 in n.items():
            if sd not in a:
                continue
            if metric == "toward":                   # sign-normalized toward-mirror lateral
                b, x = side_lat(r0), side_lat(a[sd])
                if b and x:
                    out.append(-b[0] * (x[1] - b[1]))
            else:
                f0, f1 = feats(r0), feats(a[sd])
                if f0 and f1:
                    out.append(f1[metric] - f0[metric])
        return out
    return None


def ci(v):
    if not v:
        return None
    v = np.asarray(v); rng = np.random.default_rng(0); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


M = [("ctx halt Δforward m", "ctx", "halt", "forward"), ("ctx mirror INACTIVE Δforward m", "ctx", "mirror_inactive", "forward"),
     ("ctx mirror ACTIVE toward m", "ctx", "mirror_active", "toward"), ("ctx mirror INACTIVE toward m", "ctx", "mirror_inactive", "toward"),
     ("ctx ACTIVE−INACTIVE toward m", "ctx", ("mirror_active", "mirror_inactive"), "toward"),
     ("z halt Δforward m", "z", "probe_halt", "forward"), ("z turn +0.6 Δyaw", "z", "probe_yaw_0.6", "dyaw"),
     ("z turn −0.6 Δyaw", "z", "probe_yaw_-0.6", "dyaw"), ("z goal-mirror toward m", "z", "probe_goal_mirror", "toward"),
     ("z random 8 toward m", "z", "rand_norm_8", "toward"), ("z random 16 Δforward m", "z", "rand_norm_16", "forward"),
     ("z random 25 Δforward m", "z", "rand_norm_25", "forward"), ("z random 25 Δyaw", "z", "rand_norm_25", "dyaw")]
res = {"body": body, "variants": {}}
for v in ("fixsem", "nosem", "sem_orig"):
    for s in ((0, 1, 2) if v != "sem_orig" else (0,)):
        lad, zd, cd = src(v, s)
        rr = {}
        for k, tag in zip(("r2_snap_s4000", "r2_final"), lad):
            p = L / f"{tag}.jsonl"
            rows = [json.loads(l) for l in open(p)] if p.exists() else None
            rr[k] = None if rows is None else dict(success=sum(r["success"] for r in rows), n=len(rows),
                                                   fell=sum(bool(r.get("fell")) for r in rows), file=str(p))
        n0 = load(zd[0] / "none.jsonl") or {}
        unedited_fwd = [feats(r)["forward"] for r in n0.values() if feats(r)]
        eff = {}
        for lab, kind, cond, met in M:
            dirs = [cd] if kind == "ctx" else zd
            if isinstance(cond, tuple):
                a, b = paired(dirs, cond[0], met), paired(dirs, cond[1], met)
                eff[lab] = None if not a or not b or len(a) != len(b) else [x - y for x, y in zip(a, b)]
            else:
                eff[lab] = paired(dirs, cond, met)
        res["variants"].setdefault(v, {})[s] = dict(r2=rr, unedited_forward_mean=round(float(np.mean(unedited_fwd)), 3) if unedited_fwd else None,
                                                   effects={k: ci(x) for k, x in eff.items()}, _raw=eff,
                                                   dirs=[str(d) for d in zd] + [str(cd)])
    seeds = res["variants"][v]
    pooled = {}
    for lab, *_ in M:
        vals = [x for s in seeds.values() if s["_raw"].get(lab) for x in s["_raw"][lab]]
        pooled[lab] = ci(vals)
    succ = [s["r2"][k] for s in seeds.values() for k in ("r2_snap_s4000", "r2_final") if s["r2"][k]]
    res["variants"][v]["pooled"] = dict(effects=pooled, training_seeds=sorted(k for k in seeds if k != "pooled"),
                                        r2_snap_s4000=[s["r2"]["r2_snap_s4000"] and f"{s['r2']['r2_snap_s4000']['success']}/{s['r2']['r2_snap_s4000']['n']}" for k, s in seeds.items() if k != "pooled"],
                                        r2_final=[s["r2"]["r2_final"] and f"{s['r2']['r2_final']['success']}/{s['r2']['r2_final']['n']}" for k, s in seeds.items() if k != "pooled"])
for v in res["variants"].values():
    for k, s in v.items():
        if k != "pooled":
            s.pop("_raw", None)


def f(x):
    return "–" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


V = res["variants"]
cols = [("fixsem", 0), ("fixsem", 1), ("fixsem", 2), ("fixsem", "pooled"), ("nosem", 0), ("nosem", 1), ("nosem", 2), ("nosem", "pooled"), ("sem_orig", 0)]
md = [f"| {body} | " + " | ".join(f"{v} s{s}" if s != "pooled" else f"{v} POOLED" for v, s in cols) + " |", "|" + "---|" * (len(cols) + 1)]
def r2cell(v, s, k):
    if s == "pooled":
        return ", ".join(x or "–" for x in V[v]["pooled"][k])
    d = V[v][s]["r2"]["r2_snap_s4000" if k == "r2_snap_s4000" else "r2_final"]
    return "–" if d is None else f"{d['success']}/{d['n']} ({d['fell']} fell)"
md.append("| R2 success snap_s4000 | " + " | ".join(r2cell(v, s, "r2_snap_s4000") for v, s in cols) + " |")
md.append("| R2 success final flow | " + " | ".join(r2cell(v, s, "r2_final") for v, s in cols) + " |")
md.append("| unedited forward in window (m) | " + " | ".join("" if s == "pooled" else str(V[v][s]["unedited_forward_mean"]) for v, s in cols) + " |")
for lab, *_ in M:
    md.append(f"| {lab} | " + " | ".join(f(V[v][s]["effects"][lab]) for v, s in cols) + " |")
Path(f"artifacts/runs/legged_fixrep_compare_{body}.json").write_text(json.dumps(res, indent=1, default=str))
Path(f"artifacts/runs/legged_fixrep_compare_{body}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
