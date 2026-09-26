"""LEGGED FIXED-SEM RERUN: like-for-like table (original sem / FIXED sem / nosem) from the raw rows and effect files.
usage: python scripts/legged_fixsem_compare.py BODY  -> artifacts/runs/legged_fixsem_compare_<BODY>.{json,md}"""
import json, sys
from pathlib import Path

body = sys.argv[1]
L, E = Path(f"artifacts/runs/legged_ladder/{body}"), Path(f"artifacts/runs/legged_edits/{body}")
V = {"sem_orig": ("sem", f"sem_{body}_v2"), "sem_fixed": ("fixsem", f"fixsem_{body}_lv4"), "nosem": ("nosem", f"nosem_{body}_v2")}


def rows(p):
    return [json.loads(l) for l in open(p)] if p.exists() else None


def J(p):
    return json.load(open(p)) if p.exists() else {}


out = {"body": body, "r2": {}, "z_edits": {}, "ctx_edits": {}}
for v, (ed, tag) in V.items():
    for ck in ("snap_s4000", "policy"):
        r = rows(L / f"r2_{tag}_{ck}.jsonl")
        out["r2"].setdefault(v, {})[ck] = None if r is None else dict(
            success=sum(x["success"] for x in r), n=len(r), fell=sum(bool(x.get("fell")) for x in r),
            source=r[0].get("source"), file=str(L / f"r2_{tag}_{ck}.jsonl"))
    ef = J(E / f"r2_{ed}_snap_s4000/effects.json").get("conditions", {})
    me = J(E / f"r2_{ed}_snap_s4000/mirror_effects.json")
    c = lambda k, m: ef.get(k, {}).get(m)
    out["z_edits"][v] = dict(
        dir=str(E / f"r2_{ed}_snap_s4000"),
        n_none=c("none", "n"), fell_none=c("none", "fell"),
        turn_pos_dyaw=c("probe_yaw_0.6", "dyaw"), turn_pos_dz=c("probe_yaw_0.6", "mean_dz"),
        turn_neg_dyaw=c("probe_yaw_-0.6", "dyaw"), turn_neg_dz=c("probe_yaw_-0.6", "mean_dz"),
        halt_forward=c("probe_halt", "forward"), halt_dz=c("probe_halt", "mean_dz"), halt_fell=c("probe_halt", "fell"),
        goal_mirror_toward=me.get("probe_goal_mirror", {}).get("toward_mirror_lateral_m"), goal_mirror_dz=c("probe_goal_mirror", "mean_dz"),
        rand8_toward=me.get("rand_norm_8", {}).get("toward_mirror_lateral_m"),
        leg0_stance_dcontact=(c("contact_0_1", "contact_delta_per_leg") or [None])[0],
        leg0_swing_dcontact=(c("contact_0_0", "contact_delta_per_leg") or [None])[0],
        contact_dz=[c("contact_0_1", "mean_dz"), c("contact_0_0", "mean_dz")],
        random={k: dict(forward=c(f"rand_norm_{k}", "forward"), dyaw=c(f"rand_norm_{k}", "dyaw")) for k in (1, 2, 4, 8, 12)},
        zero_forward=c("zero", "forward"))
    bg = J(E / f"r2_{ed}_snap_s4000_bigrand/effects.json").get("conditions", {})
    for k in (16, 25):
        out["z_edits"][v]["random"][k] = dict(forward=bg.get(f"rand_norm_{k}", {}).get("forward"), dyaw=bg.get(f"rand_norm_{k}", {}).get("dyaw"))
    nz = rows(E / f"r2_{ed}_snap_s4000/none.jsonl") or []
    zn = [pk["z_norm"] for r in nz for pk in r.get("packets", []) if "z_norm" in pk]
    out["z_edits"][v]["packet_z_norm_mean"] = round(sum(zn) / len(zn), 3) if zn else None
    cf = J(E / f"r2ctx_{ed}_snap_s4000/effects.json").get("conditions", {})
    cm = J(E / f"r2ctx_{ed}_snap_s4000/mirror_effects.json")
    out["ctx_edits"][v] = dict(dir=str(E / f"r2ctx_{ed}_snap_s4000"),
                               **{k: cm.get(k, {}).get("toward_mirror_lateral_m") for k in ("mirror_active", "mirror_goal", "mirror_inactive")},
                               mirror_active_yaw=cm.get("mirror_active", {}).get("toward_mirror_yaw_rad"),
                               halt_forward=cf.get("halt", {}).get("forward"))


def paired_active_minus_inactive(d):
    """per-seed toward-mirror lateral (vs the unedited run), mirror_active minus mirror_inactive; bootstrap 95% CI."""
    import math, numpy as np
    def eff(r):
        te = r["t_edit"]; tr = [x for x in r["trace"] if x["t"] >= te - 1e-6]
        if len(tr) < 2:
            return None
        x0, y0, a0 = tr[0]["pose"]; x1, y1, a1 = tr[-1]["pose"]
        ev = next((p["ev"] for p in r["packets"] if p["t"] >= te - 1e-6), 0)
        wp = r["waypoints"]["a" if ev == 0 else "b"]; c, s_ = math.cos(a0), math.sin(a0)
        return math.copysign(1.0, -s_ * (wp[0] - x0) + c * (wp[1] - y0)), -s_ * (x1 - x0) + c * (y1 - y0)
    R = {k: {r["seed"]: r for r in (rows(d / f"{k}.jsonl") or [])} for k in ("none", "mirror_active", "mirror_inactive")}
    diffs = []
    for sd, r0 in R["none"].items():
        b = eff(r0)
        if b is None or sd not in R["mirror_active"] or sd not in R["mirror_inactive"]:
            continue
        a, i = eff(R["mirror_active"][sd]), eff(R["mirror_inactive"][sd])
        if a is None or i is None:
            continue
        diffs.append(-b[0] * (a[1] - b[1]) - (-b[0] * (i[1] - b[1])))
    if not diffs:
        return None
    v = np.asarray(diffs); rng = np.random.default_rng(0); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


for v, (ed, tag) in V.items():
    out["ctx_edits"][v]["active_minus_inactive_paired"] = paired_active_minus_inactive(E / f"r2ctx_{ed}_snap_s4000")


def f(x):
    if x is None:
        return "–"
    if isinstance(x, list):
        return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def s(d):
    return "–" if d is None else f"{d['success']}/{d['n']} ({d['fell']} fell)"


md = [f"| {body} | original sem | FIXED sem (lv −4) | nosem |", "|---|---|---|---|"]
for ck in ("snap_s4000", "policy"):
    md.append(f"| R2 success, flow {ck} | " + " | ".join(s(out["r2"][v][ck]) for v in V) + " |")
Z = out["z_edits"]
for lab, k in [("z: turn +0.6 Δyaw rad", "turn_pos_dyaw"), ("z: turn −0.6 Δyaw rad", "turn_neg_dyaw"), ("z: halt Δforward m", "halt_forward"),
               ("z: goal-mirror toward-mirror lateral m", "goal_mirror_toward"), ("z: random |dz| 8 toward-mirror lateral m", "rand8_toward"),
               ("z: leg-0 stance Δcontact", "leg0_stance_dcontact"), ("z: leg-0 swing Δcontact", "leg0_swing_dcontact"), ("z = 0 Δforward m", "zero_forward")]:
    md.append(f"| {lab} | " + " | ".join(f(Z[v][k]) for v in V) + " |")
md.append("| z: edit norms |dz| turn+/turn−/halt/goal | " + " | ".join(
    "/".join(f(Z[v][k]) for k in ("turn_pos_dz", "turn_neg_dz", "halt_dz", "goal_mirror_dz")) for v in V) + " |")
md.append("| unedited packet mean |z| | " + " | ".join(f(Z[v]["packet_z_norm_mean"]) for v in V) + " |")
for kk in (4, 8, 12, 16, 25):
    md.append(f"| z: random |dz| {kk} Δforward / Δyaw | " + " | ".join(
        f"{f(Z[v]['random'][kk]['forward'])} / {f(Z[v]['random'][kk]['dyaw'])}" for v in V) + " |")
C = out["ctx_edits"]
for lab, k in [("ctx: mirror ACTIVE waypoint, toward-mirror lateral m", "mirror_active"), ("ctx: mirror both waypoints", "mirror_goal"),
               ("ctx: mirror INACTIVE waypoint (control)", "mirror_inactive"),
               ("ctx: ACTIVE − INACTIVE, paired per seed", "active_minus_inactive_paired"), ("ctx: task view halt, Δforward m", "halt_forward")]:
    md.append(f"| {lab} | " + " | ".join(f(C[v][k]) for v in V) + " |")
Path(f"artifacts/runs/legged_fixsem_compare_{body}.json").write_text(json.dumps(out, indent=1))
Path(f"artifacts/runs/legged_fixsem_compare_{body}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
