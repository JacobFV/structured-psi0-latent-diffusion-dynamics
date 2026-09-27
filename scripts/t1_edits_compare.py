"""T1 CONTEXT-HALT SUITE: fixed sem vs nosem on the t1 humanoid over training seeds 0/1/3 (existing D-087 models), per seed and
pooled, in the legged_fixrep_compare format, plus FALLS under every edit (t1 can fall; a fall also shortens forward progress).
Effects are paired per eval seed vs the same model's unedited run (window t=2-5 s), pooled over (training seed, eval seed),
bootstrap 95% CI over episodes. Extra rows: the context-halt effect restricted to pairs where neither episode fell.
usage: python scripts/t1_edits_compare.py CKPT (snap_s4000 | policy) -> artifacts/runs/t1_edits_compare_<CKPT>.{json,md}"""
import json, sys
from pathlib import Path

import numpy as np

CK = sys.argv[1]; body = "t1"
L, E = Path(f"artifacts/runs/legged_ladder/{body}"), Path(f"artifacts/runs/legged_edits/{body}")
src = open("scripts/legged_fixrep_compare.py").read()
exec(src[src.index("exec(open"):src.index("def src(")])      # feats()
exec(src[src.index("def load(p)"):src.index("M = [")])         # load, side_lat, paired, ci
SEEDS = (0, 1, 3)
M = [("ctx halt Δforward m", "ctx", "halt", "forward"), ("ctx mirror INACTIVE Δforward m", "ctx", "mirror_inactive", "forward"),
     ("ctx mirror ACTIVE toward m", "ctx", "mirror_active", "toward"), ("ctx mirror INACTIVE toward m", "ctx", "mirror_inactive", "toward"),
     ("ctx ACTIVE−INACTIVE toward m", "ctx", ("mirror_active", "mirror_inactive"), "toward"),
     ("z halt Δforward m", "z", "probe_halt", "forward"), ("z turn +0.6 Δyaw", "z", "probe_yaw_0.6", "dyaw"),
     ("z turn −0.6 Δyaw", "z", "probe_yaw_-0.6", "dyaw"), ("z goal-mirror toward m", "z", "probe_goal_mirror", "toward"),
     ("z random 8 toward m", "z", "rand_norm_8", "toward"), ("z random 16 Δforward m", "z", "rand_norm_16", "forward"),
     ("z random 25 Δforward m", "z", "rand_norm_25", "forward"), ("z random 25 Δyaw", "z", "rand_norm_25", "dyaw")]
FALLS = [("ctx", c) for c in ("none", "mirror_goal", "mirror_active", "mirror_inactive", "halt")] + \
        [("z", c) for c in ("none", "probe_halt", "probe_yaw_0.6", "probe_yaw_-0.6", "probe_goal_mirror", "contact_0_1", "contact_0_0",
                            "rand_norm_8", "rand_norm_16", "rand_norm_25", "zero")]


def nofall_halt(d, cond="halt"):
    a, n = load(d / f"{cond}.jsonl"), load(d / "none.jsonl")
    if a is None or n is None:
        return None
    out = []
    for sd, r0 in n.items():
        if sd in a and not r0.get("fell") and not a[sd].get("fell"):
            f0, f1 = feats(r0), feats(a[sd])
            if f0 and f1:
                out.append(f1["forward"] - f0["forward"])
    return out


res = {"body": body, "ckpt": CK, "variants": {}}
for v in ("fixsem", "nosem"):
    for s in SEEDS:
        T = f"t1edits_{v}_s{s}_{CK}"
        zd, cd = [E / f"r2_{T}"], E / f"r2ctx_{T}"
        p = L / f"r2_{T}.jsonl"
        rows = [json.loads(l) for l in open(p)] if p.exists() else None
        r2 = None if rows is None else dict(success=sum(r["success"] for r in rows), n=len(rows),
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
        eff["ctx halt Δforward m, pairs without a fall"] = nofall_halt(cd)
        eff["ctx mirror INACTIVE Δforward m, pairs without a fall"] = nofall_halt(cd, "mirror_inactive")
        falls = {}
        for kind, c in FALLS:
            rr = load((cd if kind == "ctx" else zd[0]) / f"{c}.jsonl")
            falls[f"{kind}:{c}"] = None if rr is None else f"{sum(bool(r.get('fell')) for r in rr.values())}/{len(rr)}"
        res["variants"].setdefault(v, {})[s] = dict(r2=r2, unedited_forward_mean=round(float(np.mean(unedited_fwd)), 3) if unedited_fwd else None,
                                                   effects={k: ci(x) for k, x in eff.items()}, falls_t2_5s=falls, _raw=eff,
                                                   dirs=[str(d) for d in zd] + [str(cd)])
    seeds = res["variants"][v]
    labs = list(next(iter(seeds.values()))["_raw"].keys())
    pooled = {lab: ci([x for s in seeds.values() if s["_raw"].get(lab) for x in s["_raw"][lab]]) for lab in labs}
    pf = {}
    for k in next(iter(seeds.values()))["falls_t2_5s"]:
        xs = [s["falls_t2_5s"][k] for s in seeds.values() if s["falls_t2_5s"][k]]
        pf[k] = f"{sum(int(x.split('/')[0]) for x in xs)}/{sum(int(x.split('/')[1]) for x in xs)}" if xs else None
    res["variants"][v]["pooled"] = dict(effects=pooled, falls_t2_5s=pf, training_seeds=list(SEEDS),
                                        r2=[s["r2"] and f"{s['r2']['success']}/{s['r2']['n']}" for s in seeds.values()])
for v in res["variants"].values():
    for k, s in v.items():
        if k != "pooled":
            s.pop("_raw", None)


def f(x):
    return "–" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


V = res["variants"]
cols = [("fixsem", s) for s in SEEDS] + [("fixsem", "pooled")] + [("nosem", s) for s in SEEDS] + [("nosem", "pooled")]
md = [f"| t1 ({CK}) | " + " | ".join(f"{v} s{s}" if s != "pooled" else f"{v} POOLED" for v, s in cols) + " |", "|" + "---|" * (len(cols) + 1)]


def r2cell(v, s):
    if s == "pooled":
        return ", ".join(x or "–" for x in V[v]["pooled"]["r2"])
    d = V[v][s]["r2"]
    return "–" if d is None else f"{d['success']}/{d['n']} ({d['fell']} fell)"


md.append(f"| R2 success {CK} (dev 10000–10029, full episode) | " + " | ".join(r2cell(v, s) for v, s in cols) + " |")
md.append("| unedited forward in window (m) | " + " | ".join("" if s == "pooled" else str(V[v][s]["unedited_forward_mean"]) for v, s in cols) + " |")
for lab in V["fixsem"]["pooled"]["effects"]:
    md.append(f"| {lab} | " + " | ".join(f(V[v][s]["effects"][lab]) for v, s in cols) + " |")
md.append(f"| **falls by t=5 s under each condition (20 dev seeds)** |" + " |" * len(cols))
for k in V["fixsem"]["pooled"]["falls_t2_5s"]:
    md.append(f"| falls {k} | " + " | ".join((V[v][s]["falls_t2_5s"][k] or "–") for v, s in cols) + " |")
Path(f"artifacts/runs/t1_edits_compare_{CK}.json").write_text(json.dumps(res, indent=1, default=str))
Path(f"artifacts/runs/t1_edits_compare_{CK}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
