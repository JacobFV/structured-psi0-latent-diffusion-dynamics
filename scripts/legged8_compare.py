"""W8 (contact v2) legged semantic-supervision table from `rrp run-dag dags/legged_v2_anymal.yaml` outputs.
Same measures and statistics as scripts/legged_fixrep_compare.py (D-090): effects are paired per eval seed against the
same model's unedited run over t=2-5 s; per training seed and POOLED over (training seed x eval seed) with a bootstrap
95% CI over episodes. Added: the scripted_teacher and bc:<ckpt> references on the same R2 dev seeds, and per measure
an exact permutation test over the training-seed means (fixsem vs nosem; one-sided in the direction of the observed
difference, and two-sided), plus the pooled fixsem-nosem difference with a bootstrap CI.
usage: python scripts/legged8_compare.py BODY [LINEAGE_PREFIX=legged8] [TRACK_DIR=artifacts/runs/legged8]
   -> <TRACK_DIR>/legged8_compare_<BODY>.{json,md}"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

body = sys.argv[1]
pre = sys.argv[2] if len(sys.argv) > 2 else "legged8"
T = Path(sys.argv[3] if len(sys.argv) > 3 else "artifacts/runs/legged8")
exec(open("scripts/legged_edit_effects.py").read().split("def ci")[0])   # feats(r): forward/lateral/dyaw/... in window
SEEDS = (0, 1, 2)
VARIANTS = ("semfix", "nosem", "sem")


def lin(v):
    return T / f"{pre}-{body}-{v}"


def load(p):
    return {r["seed"]: r for r in map(json.loads, open(p))} if p and p.exists() else None


def first(glob_dir, pattern):
    fs = sorted(glob_dir.glob(pattern)) if glob_dir.exists() else []
    return fs[0] if fs else None


def src(v, s):
    L = lin(v)
    r2 = [first(L / f"eval_r2-snap_s{s}" / body, "r2_*.jsonl"), first(L / f"eval_r2-policy_s{s}" / body, "r2_*.jsonl")]
    zd = first(L / f"edits-z_s{s}", "r2_*")
    cd = first(L / f"edits-ctx_s{s}", "r2ctx_*")
    return r2, [zd] if zd else [], cd


def side_lat(r):
    te = r["t_edit"]; tr = [x for x in r["trace"] if x["t"] >= te - 1e-6]
    if len(tr) < 2:
        return None
    x0, y0, a0 = tr[0]["pose"]; x1, y1, a1 = tr[-1]["pose"]
    ev = next((p["ev"] for p in r["packets"] if p["t"] >= te - 1e-6), 0)
    wp = r["waypoints"]["a" if ev == 0 else "b"]; c, s_ = math.cos(a0), math.sin(a0)
    return math.copysign(1.0, -s_ * (wp[0] - x0) + c * (wp[1] - y0)), -s_ * (x1 - x0) + c * (y1 - y0)


def paired(dirs, cond, metric):
    for d in dirs:
        if d is None:
            continue
        a, n = load(d / f"{cond}.jsonl"), load(d / "none.jsonl")
        if a is None or n is None:
            continue
        out = []
        for sd, r0 in sorted(n.items()):
            if sd not in a:
                continue
            if metric == "toward":
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


def diff_ci(a, b):
    if not a or not b:
        return None
    a, b = np.asarray(a), np.asarray(b); rng = np.random.default_rng(1)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
    return [round(float(a.mean() - b.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def perm(a, b):
    """Exact permutation test over group labels of the per-seed means a (fixsem) vs b (nosem)."""
    if len(a) < 2 or len(b) < 2:
        return None
    allv = list(a) + list(b); n = len(a); obs = np.mean(a) - np.mean(b)
    ds = [np.mean([allv[i] for i in c]) - np.mean([allv[i] for i in range(len(allv)) if i not in c])
          for c in itertools.combinations(range(len(allv)), n)]
    lo = sum(d <= obs + 1e-12 for d in ds) / len(ds); hi = sum(d >= obs - 1e-12 for d in ds) / len(ds)
    two = sum(abs(d) >= abs(obs) - 1e-12 for d in ds) / len(ds)
    return dict(diff=round(float(obs), 3), p_one_sided=round(min(lo, hi), 3), direction="fixsem<nosem" if obs < 0 else "fixsem>nosem",
                p_two_sided=round(two, 3), n_perm=len(ds), seeds_fixsem=[round(float(x), 3) for x in a],
                seeds_nosem=[round(float(x), 3) for x in b],
                every_seed_ordered=bool(max(a) < min(b) or min(a) > max(b)))


M = [("ctx halt Δforward m", "ctx", "halt", "forward"), ("ctx mirror INACTIVE Δforward m", "ctx", "mirror_inactive", "forward"),
     ("ctx mirror ACTIVE toward m", "ctx", "mirror_active", "toward"), ("ctx mirror INACTIVE toward m", "ctx", "mirror_inactive", "toward"),
     ("ctx ACTIVE−INACTIVE toward m", "ctx", ("mirror_active", "mirror_inactive"), "toward"),
     ("z halt Δforward m", "z", "probe_halt", "forward"), ("z turn +0.6 Δyaw", "z", "probe_yaw_0.6", "dyaw"),
     ("z turn −0.6 Δyaw", "z", "probe_yaw_-0.6", "dyaw"), ("z goal-mirror toward m", "z", "probe_goal_mirror", "toward"),
     ("z random 8 toward m", "z", "rand_norm_8", "toward"), ("z random 16 Δforward m", "z", "rand_norm_16", "forward"),
     ("z random 25 Δforward m", "z", "rand_norm_25", "forward"), ("z random 25 Δyaw", "z", "rand_norm_25", "dyaw")]


def r2stat(p):
    rows = [json.loads(l) for l in open(p)] if p and p.exists() else None
    return None if rows is None else dict(success=sum(r["success"] for r in rows), n=len(rows),
                                          fell=sum(bool(r.get("fell")) for r in rows), file=str(p),
                                          source=rows[0]["source"] if rows else None,
                                          contact_versions=sorted({str(r.get("contact_version")) for r in rows}))


res = {"body": body, "contact_version": "contact_v2", "variants": {}, "references": {}}
D = T / f"{pre}-{body}-v2data"
for k, pat in (("teacher", "eval_r2-teacher_s0"), ("bc", "eval_r2-bc_s0")):
    res["references"][k] = r2stat(first(D / pat / body, "*.jsonl"))
raw = {}
for v in VARIANTS:
    for s in SEEDS:
        lad, zd, cd = src(v, s)
        if not any(lad) and not zd and cd is None:
            continue
        rr = {k: r2stat(p) for k, p in zip(("r2_snap_s4000", "r2_final"), lad)}
        n0 = (load(zd[0] / "none.jsonl") if zd else None) or {}
        unedited_fwd = [feats(r)["forward"] for r in n0.values() if feats(r)]
        eff = {}
        for lab, kind, cond, met in M:
            dirs = [cd] if kind == "ctx" else zd
            if isinstance(cond, tuple):
                a, b = paired(dirs, cond[0], met), paired(dirs, cond[1], met)
                eff[lab] = None if not a or not b or len(a) != len(b) else [x - y for x, y in zip(a, b)]
            else:
                eff[lab] = paired(dirs, cond, met)
        raw.setdefault(v, {})[s] = eff
        res["variants"].setdefault(v, {})[s] = dict(r2=rr, unedited_forward_mean=round(float(np.mean(unedited_fwd)), 3) if unedited_fwd else None,
                                                   effects={k: ci(x) for k, x in eff.items()},
                                                   n_pairs={k: (len(x) if x else 0) for k, x in eff.items()},
                                                   dirs=[str(d) for d in zd] + [str(cd)])
    if v not in res["variants"]:
        continue
    seeds = res["variants"][v]
    pooled = {lab: ci([x for s in raw[v].values() if s.get(lab) for x in s[lab]]) for lab, *_ in M}
    seeds["pooled"] = dict(effects=pooled, training_seeds=sorted(k for k in seeds if k != "pooled"),
                           r2_snap_s4000=[s["r2"]["r2_snap_s4000"] and f"{s['r2']['r2_snap_s4000']['success']}/{s['r2']['r2_snap_s4000']['n']}" for k, s in seeds.items() if k != "pooled"],
                           r2_final=[s["r2"]["r2_final"] and f"{s['r2']['r2_final']['success']}/{s['r2']['r2_final']['n']}" for k, s in seeds.items() if k != "pooled"])
tests = {}
if "semfix" in raw and "nosem" in raw:
    for lab, *_ in M:
        a = [float(np.mean(raw["semfix"][s][lab])) for s in sorted(raw["semfix"]) if raw["semfix"][s].get(lab)]
        b = [float(np.mean(raw["nosem"][s][lab])) for s in sorted(raw["nosem"]) if raw["nosem"][s].get(lab)]
        pa = [x for s in raw["semfix"].values() if s.get(lab) for x in s[lab]]
        pb = [x for s in raw["nosem"].values() if s.get(lab) for x in s[lab]]
        tests[lab] = dict(permutation=perm(a, b), pooled_diff_fixsem_minus_nosem=diff_ci(pa, pb))
res["tests_fixsem_vs_nosem"] = tests


def f(x):
    return "–" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


V = res["variants"]
cols = [(v, s) for v in VARIANTS if v in V for s in list(SEEDS) + ["pooled"] if s in V[v]]
md = [f"# {body}, contact_v2 (W8). Sources: learned:<flow ckpt> (R2 deployable route); references scripted_teacher and bc:<ckpt>",
      "", f"| {body} (contact_v2) | " + " | ".join(f"{v} s{s}" if s != "pooled" else f"{v} POOLED" for v, s in cols) + " |",
      "|" + "---|" * (len(cols) + 1)]


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
md += ["", "References on the same R2 dev seeds:", ""]
for k, d in res["references"].items():
    md.append(f"- {k}: " + ("–" if d is None else f"{d['success']}/{d['n']} ({d['fell']} fell), source `{d['source']}`, physics {d['contact_versions']}"))
if tests:
    md += ["", "fixsem vs nosem (per-seed means; exact permutation over the 3 vs 3 training seeds; pooled difference with bootstrap CI):", "",
           "| measure | fixsem seeds | nosem seeds | every seed ordered | p one-sided | p two-sided | pooled fixsem−nosem |", "|---|---|---|---|---|---|---|"]
    for lab, t in tests.items():
        p = t["permutation"]
        if p is None:
            md.append(f"| {lab} | – | – | – | – | – | {f(t['pooled_diff_fixsem_minus_nosem'])} |")
        else:
            md.append(f"| {lab} | {p['seeds_fixsem']} | {p['seeds_nosem']} | {p['every_seed_ordered']} | {p['p_one_sided']} ({p['direction']}) | "
                      f"{p['p_two_sided']} | {f(t['pooled_diff_fixsem_minus_nosem'])} |")
(T / f"legged8_compare_{body}.json").write_text(json.dumps(res, indent=1, default=str))
(T / f"legged8_compare_{body}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
