"""Legged ladder / edit summaries, moved from scripts (D-126). The scripts are thin wrappers with the same CLI, stdout
and output files:
  rrp suite legged-summary ROWS.jsonl ...        -> ladder_summary_main: <rows>.summary.json per file
  rrp suite legged-edit-effects DIR             -> edit_effects_main: DIR/effects.json
  rrp suite legged-mirror-effect DIR EDIT ...   -> mirror_effect_main: DIR/mirror_effects.json
(scripts/legged8_summary.py stays a script: it is the one-off W8 report with a hard-coded headline and protocol text.)

Bootstrap CIs use one np.random.default_rng(0) stream per main() call, consumed in the scripts' call order; this
reproduces the scripts (whose module-level default rng was created once per process) exactly for a single invocation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ ladder summary
def wilson(k, n, z=1.96):
    """Ladder-summary format of rrp.harness.eval.statistics.wilson: z=1.96, rounded to 3 decimals, (0, 0) for n == 0.
    (The former local copy returned -0.0 as the lower bound at k == 0; the shared formula clamps it to 0.0.)"""
    from rrp.harness.eval.statistics import wilson as _w
    if n == 0:
        return (0.0, 0.0)
    lo, hi = _w(k, n, z)
    return (round(lo, 3), round(hi, 3))


def ladder_summary(f: str, rows: list[dict]) -> dict:
    """Success / failure stages / Wilson 95% of one legged ladder rows file."""
    st = {}
    for r in rows:
        st[r["failure_stage"]] = st.get(r["failure_stage"], 0) + 1
    k = sum(r["success"] for r in rows)
    return dict(file=f, source=rows[0]["source"], edit=rows[0]["edit"], n=len(rows), success=k, wilson95=wilson(k, len(rows)),
                stages=st, seeds=sorted(r["seed"] for r in rows)[:1] + sorted(r["seed"] for r in rows)[-1:],
                events_a=sum(r["events"].get("walk_to_a") == "succeeded" for r in rows),
                events_b=sum(r["events"].get("walk_to_b") == "succeeded" for r in rows))


def ladder_summary_main(argv=None) -> None:
    """Summarize legged ladder rows: success / stages / Wilson 95% per file; writes <file>.summary.json next to each."""
    for f in (sys.argv[1:] if argv is None else argv):
        if f.endswith("summary.json"):
            continue
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        s = ladder_summary(f, rows)
        Path(f).with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s))


# ------------------------------------------------------------------ edit effects
def edit_feats(r):
    """Per-episode features in the window [t_edit, end] (body frame at t_edit); None if the window is too short."""
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


def _ci4(v, rng):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def edit_effects(d: Path, rng=None) -> dict:
    """Paired edit effects vs the unedited control ('none.jsonl') on the same seeds; mean and bootstrap 95% CI."""
    rng = np.random.default_rng(0) if rng is None else rng
    rows = {f.stem: {r["seed"]: r for r in map(json.loads, open(f))} for f in d.glob("*.jsonl")}
    ctrl = {s: edit_feats(r) for s, r in rows.get("none", {}).items()}
    out = dict(dir=str(d), source=next(iter(rows["none"].values()))["source"] if "none" in rows else None, conditions={})
    for name, rs in sorted(rows.items()):
        eff = {k: [] for k in ("forward", "lateral", "dyaw", "path")}
        legs, dzs, fell = [], [], 0
        for s, r in rs.items():
            f, c = edit_feats(r), ctrl.get(s)
            if f is None or c is None:
                continue
            for k in eff:
                eff[k].append(f[k] - c[k])
            legs.append(f["contact"] - c["contact"]); dzs.append(f["dz"]); fell += f["fell"]
        out["conditions"][name] = dict(n=len(eff["path"]), fell=fell, mean_dz=round(float(np.mean(dzs)), 3) if dzs else None,
                                       **{k: _ci4(v, rng) for k, v in eff.items()},
                                       contact_delta_per_leg=[_ci4(np.array(legs)[:, m], rng) for m in range(len(legs[0]))] if legs else None)
    return out


def edit_effects_main(argv=None) -> None:
    """Paired edit effects vs the unedited control on the same seeds, in the window [t_edit, end].
    Per episode (body frame at t_edit): forward / lateral displacement, yaw change, path length, per-leg contact fraction.
    Effect = edited - control (paired by seed); mean and bootstrap 95% CI. Also mean |dz| of the edited packets."""
    argv = sys.argv[1:] if argv is None else argv
    d = Path(argv[0])
    out = edit_effects(d)
    (d / "effects.json").write_text(json.dumps(out, indent=1))
    for k, v in out["conditions"].items():
        print(k, json.dumps({a: b for a, b in v.items() if a != "contact_delta_per_leg"}), "leg0", (v["contact_delta_per_leg"] or [None])[0])


# ------------------------------------------------------------------ mirror effect
def mirror_side_and_eff(r):
    """(original goal side sign, lateral displacement, yaw change) in the body frame at t_edit; None if too short."""
    te = r["t_edit"]; tr = [x for x in r["trace"] if x["t"] >= te - 1e-6]
    if len(tr) < 2:
        return None
    x0, y0, a0 = tr[0]["pose"]; x1, y1, a1 = tr[-1]["pose"]
    ev = next((p["ev"] for p in r["packets"] if p["t"] >= te - 1e-6), 0)
    wp = r["waypoints"]["a" if ev == 0 else "b"]
    c, s = math.cos(a0), math.sin(a0)
    gy = -s * (wp[0] - x0) + c * (wp[1] - y0)
    return math.copysign(1.0, gy), -s * (x1 - x0) + c * (y1 - y0), (a1 - a0 + math.pi) % (2 * math.pi) - math.pi


def _ci3(v, rng):
    v = np.asarray(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def mirror_effect_main(argv=None) -> None:
    """Mirror edits: lateral displacement and yaw change TOWARD the mirrored goal side (sign-normalized by the side of
    the active waypoint at t_edit in the true body frame; privileged, scoring only), paired vs the unedited run."""
    argv = sys.argv[1:] if argv is None else argv
    rng = np.random.default_rng(0)
    d = Path(argv[0]); res = {}
    ctrl = {r["seed"]: r for r in map(json.loads, open(d / "none.jsonl"))}
    for name in argv[1:]:
        lat, yaw = [], []
        for r in map(json.loads, open(d / f"{name}.jsonl")):
            a, b = mirror_side_and_eff(r), mirror_side_and_eff(ctrl[r["seed"]]) if r["seed"] in ctrl else None
            if a is None or b is None:
                continue
            sg = b[0]                                   # original goal side (from the unedited run at t_edit)
            lat.append(-sg * (a[1] - b[1])); yaw.append(-sg * (a[2] - b[2]))
        res[name] = dict(n=len(lat), toward_mirror_lateral_m=_ci3(lat, rng), toward_mirror_yaw_rad=_ci3(yaw, rng))
        print(name, res[name])
    (d / "mirror_effects.json").write_text(json.dumps(res, indent=1))
