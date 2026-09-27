"""Robustness sweeps (W6): where does each policy break as the physics departs from training?

One factor at a time over a grid (plus one joint "all moderate" condition and the nominal), the SAME seeds at every
level (paired), for any legged (waypoint_contact, rrp.evaluation.legged_latent_eval.run_episode) or arm (pick_place,
rrp.evaluation.ladder.run_ladder) route. The perturbations are rrp.envs.perturb.PhysicsPerturbation (model-level
changes keep the robot spec / morphology features the policy sees NOMINAL; nothing enters an observation).

Routes (`--route NAME=SPEC`; NAME is a free label, the SOURCE label of every row comes from the eval code itself):
  legged  teacher                 scripted_teacher (privileged waypoint teacher -> frozen learned tracker; reference)
          bc:<policy.pt>          bc:<ckpt> (plain legged BC, positive control)
          flow:<policy.pt>        learned:<ckpt> (system-i flow -> the same packet -> system 0, deployable R2 route)
  arm     teacher                 scripted_teacher(privileged) (ladder R0)
          learned:<ckpt.pt>       learned:<ckpt> plain BC FlowPolicy (ladder route `learned`)
          generated:<flow.pt>@<representation.pt>   ladder R2 (system-i flow -> system 0), deployable

Grid (FACTORS): legged friction / mass / com_x / com_y / kp / latency (actuator mode v1lat) / push (dv m/s x total
mass = impulse N s, root, t = 4 s) / terrain (bump amplitude m); arm friction / mass / com_x (last arm link) / kp /
latency (ctrl_delay_v0) / push (impulse N s, last arm link, t = 3 s) / object_mass / object_friction; both + all_moderate.

Outputs (resumable; a shard with a complete summary is skipped): <out>/<route>/<robot>/<factor>=<level>.jsonl (one row per
episode: the unchanged eval row + `condition` + `perturbation` + `motion`) and .summary.json. `report` writes
<out>/robustness_report.{json,md}: per route x robot x factor x level success k/n with Wilson 95% CI, falls, task metrics,
motion-quality medians, paired changes vs nominal (lost / gained seeds), and the BREAK-POINT = the first level, moving away
from nominal on each side, whose success rate is more than 20 points below nominal.

usage:
  python -m rrp.evaluation.robustness run --family legged --robots anymal_c --seeds 10000-10019 \\
      --route semfix=flow:artifacts/.../policy.pt --route teacher=teacher --out artifacts/runs/robust/legged [--factors a,b]
      [--shard i/n] [--workers k]
  python -m rrp.evaluation.robustness report --out artifacts/runs/robust/legged
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rrp.envs.perturb import PhysicsPerturbation
from rrp.evaluation.statistics import wilson

BREAK_DROP = 0.20
NOMINAL = "nominal"

# factor -> (levels, level -> PhysicsPerturbation kwargs, nominal level value, one-sided?)
LEGGED_FACTORS = {
    "friction": ([0.4, 0.6, 0.8, 1.25, 1.5], lambda v: dict(friction_scale=v), 1.0),
    "mass": ([0.8, 0.9, 1.1, 1.2], lambda v: dict(mass_scale=v), 1.0),
    "com_x": ([-0.02, -0.01, 0.01, 0.02], lambda v: dict(com_offset_m=(v, 0.0, 0.0)), 0.0),
    "com_y": ([0.01, 0.02], lambda v: dict(com_offset_m=(0.0, v, 0.0)), 0.0),
    "kp": ([0.7, 0.85, 1.15, 1.3], lambda v: dict(kp_scale=v), 1.0),
    "latency_ms": ([0, 10, 20, 30], lambda v: dict(latency_ms=float(v)), None),     # None = ideal actuators
    "push_dv": ([0.25, 0.5, 0.75, 1.0, 1.5], None, 0.0),                           # filled per body (mass)
    "terrain_m": ([0.01, 0.02, 0.03, 0.05, 0.08], lambda v: dict(terrain_amp_m=v), 0.0),
}
LEGGED_MODERATE = dict(friction_scale=0.7, mass_scale=1.1, com_offset_m=(0.01, 0.0, 0.0), kp_scale=0.85, latency_ms=20.0,
                       terrain_amp_m=0.02)
LEGGED_MODERATE_DV = 0.5
LEGGED_PUSH_T = 4.0

ARM_FACTORS = {
    "friction": ([0.4, 0.6, 0.8, 1.25, 1.5], lambda v: dict(friction_scale=v), 1.0),
    "mass": ([0.8, 0.9, 1.1, 1.2], lambda v: dict(mass_scale=v), 1.0),
    "com_x": ([-0.02, -0.01, 0.01, 0.02], lambda v: dict(com_offset_m=(v, 0.0, 0.0)), 0.0),
    "kp": ([0.7, 0.85, 1.15, 1.3], lambda v: dict(kp_scale=v), 1.0),
    "latency_ms": ([10, 20, 30], lambda v: dict(latency_ms=float(v)), 0),
    "push_Ns": ([2.0, 5.0, 10.0, 20.0, 40.0], lambda v: dict(push_impulse_Ns=v, push_time_s=ARM_PUSH_T), 0.0),
    "object_mass": ([0.5, 2.0, 5.0, 10.0, 20.0], lambda v: dict(object_mass_scale=v), 1.0),
    "object_friction": ([0.05, 0.1, 0.2, 0.4, 0.7], lambda v: dict(object_friction_scale=v), 1.0),
}
ARM_PUSH_T = 3.0
ARM_MODERATE = dict(friction_scale=0.8, mass_scale=1.1, com_offset_m=(0.01, 0.0, 0.0), kp_scale=0.85, latency_ms=20.0,
                    push_impulse_Ns=2.0, push_time_s=ARM_PUSH_T, object_mass_scale=2.0, object_friction_scale=0.6)


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def legged_mass(body: str) -> float:
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.legged_core import LeggedBinding
    m, _, meta = standalone_model(legged_body(body))
    return float(m.body_subtreemass[LeggedBinding(m, meta).root_bid])


def conditions(family: str, robot: str, factors=None) -> list[dict]:
    """[{factor, level, pert: PhysicsPerturbation, key}] including the nominal and all_moderate conditions."""
    out = [dict(factor=NOMINAL, level=None, pert=PhysicsPerturbation(), key=NOMINAL)]
    if family == "legged":
        mass = legged_mass(robot)
        F = dict(LEGGED_FACTORS)
        F["push_dv"] = (F["push_dv"][0], lambda v: dict(push_impulse_Ns=round(v * mass, 3), push_time_s=LEGGED_PUSH_T), 0.0)
        mod = dict(LEGGED_MODERATE, push_impulse_Ns=round(LEGGED_MODERATE_DV * mass, 3), push_time_s=LEGGED_PUSH_T)
    else:
        F, mod = ARM_FACTORS, ARM_MODERATE
    for f, (levels, mk, _nom) in F.items():
        if factors and f not in factors:
            continue
        for v in levels:
            out.append(dict(factor=f, level=v, pert=PhysicsPerturbation(**mk(v)), key=f"{f}={_fmt(v)}"))
    if not factors or "all_moderate" in factors:
        out.append(dict(factor="all_moderate", level=1, pert=PhysicsPerturbation(**mod), key="all_moderate"))
    return out


def factor_table(family: str) -> dict:
    F = LEGGED_FACTORS if family == "legged" else ARM_FACTORS
    return {f: dict(levels=list(v[0]), nominal=v[2]) for f, v in F.items()}


def feasible_arm_seeds(robot_key: str, start: int, n: int, task: str = "pick_place") -> list[int]:
    """The ladder's seed-set definition (same as rrp.training.latent_grpo.feasible_seeds, which evaluation may not
    import): the first n seeds >= start whose scene the privileged teacher's feasibility check accepts."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.native import Session
    from rrp.envs.scenario import BUILDERS
    from rrp.teachers.arm import PickPlaceTeacher
    robot = workbench_robots()[robot_key]()
    out, sd = [], start
    while len(out) < n:
        s = Session(BUILDERS[task](robot, sd, n_distractors=sd % 3), seed=sd)
        if PickPlaceTeacher(s).feasibility()["feasible"]:
            out.append(sd)
        sd += 1
    return out


def parse_route(spec: str, family: str) -> dict:
    name, _, rs = spec.partition("=")
    if not rs:
        raise ValueError(f"route {spec!r}: expected NAME=SPEC")
    kind, _, arg = rs.partition(":")
    if family == "legged" and kind not in ("teacher", "bc", "flow"):
        raise ValueError(f"legged route kind {kind!r}")
    if family == "arm" and kind not in ("teacher", "learned", "generated"):
        raise ValueError(f"arm route kind {kind!r}")
    r = dict(name=name, kind=kind)
    if kind == "generated":
        r["flow"], _, r["rep"] = arg.partition("@")
    elif kind != "teacher":
        r["ckpt"] = arg
    return r


def _sha(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ one shard = route x robot x condition
def run_legged_shard(route: dict, body: str, cond: dict, seeds: list[int], out: Path, max_s: float = 60.0,
                     video_dir: Path | None = None, video_seeds=()) -> list[dict]:
    import torch
    from rrp.evaluation.legged_latent_eval import BCController, LatentLeggedController, run_episode, save_video
    dev = torch.device("cpu")
    done = {json.loads(l)["seed"] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    cached = None
    with open(out, "a") as f:
        for sd in seeds:
            if sd in done:
                continue
            # one controller per shard (loading + fingerprinting the checkpoints costs ~15 s); per seed its only
            # seed-dependent state is reset exactly as a fresh construction would set it (generator, packet/trace logs)
            if route["kind"] in ("bc", "flow"):
                if cached is None:
                    cached = (BCController(Path(route["ckpt"]), dev, nfe=8, replan=5, seed=sd) if route["kind"] == "bc" else
                              LatentLeggedController(Path(route["ckpt"]), dev, nfe=8, edit="none", t_edit=1.0, seed=sd))
                ctl = cached
                ctl.gen = torch.Generator(device=dev).manual_seed(sd)
                ctl.packets, ctl.trace = [], []
                if hasattr(ctl, "oracle"):
                    ctl.oracle = None
            else:
                ctl = None
            want = video_dir is not None and sd in video_seeds
            row, frames = run_episode(ctl, body, sd, max_s, video=True if want else None, perturb=cond["pert"])
            row.update(condition=dict(factor=cond["factor"], level=cond["level"], key=cond["key"]), route_name=route["name"])
            for k in ("packets", "packet_log", "trace"):
                row.pop(k, None)                    # large per-tick logs are not needed for the sweep
            if want:
                lab = dict(teacher="SCRIPTED TEACHER (privileged)", bc=f"LEARNED BC (positive control) {row['source']}",
                           flow=f"LEARNED latent sys-i+sys-0 {row['source']}")[route["kind"]]
                row["video"] = save_video(frames, row, video_dir, f"{lab} | ROBUSTNESS {cond['key']}")
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            del frames, ctl
            gc.collect()
    return rows


def run_arm_shard(route: dict, robot: str, cond: dict, seeds: list[int], out: Path, batch: int = 16) -> list[dict]:
    from rrp.evaluation.ladder import LadderConfig, load_models, run_ladder
    if out.exists():
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        if {r["seed"] for r in rows} >= set(seeds):
            return rows
        out.unlink()                                # batches share the flow RNG: a partial shard is redone whole
    kind = route["kind"]
    cfg = LadderConfig(route=kind, robot=robot, seeds=list(seeds), representation=route.get("rep"), flow=route.get("flow"),
                       policy=route.get("ckpt"), policy_label=(Path(route["ckpt"]).stem if route.get("ckpt") else None),
                       replan_ticks=8, max_steps=300, nfe=8, compare_oracle=False, device="cpu", prev_action="zero",
                       perturb=None if cond["factor"] == NOMINAL else cond["pert"])
    models, ids = load_models(cfg) if kind != "teacher" else (dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None), {})
    rows = []
    tmp = out.with_suffix(".partial")
    if tmp.exists():
        tmp.unlink()
    for i in range(0, len(seeds), batch):
        c = replace(cfg, seeds=list(seeds[i:i + batch]))
        rows += run_ladder(c, None, models, ids)
    for r in rows:
        r.update(condition=dict(factor=cond["factor"], level=cond["level"], key=cond["key"]), route_name=route["name"])
        r.pop("by_phase", None); r.pop("lab_err_by_j", None); r.pop("oracle_cmp_by_phase", None)
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    tmp.rename(out)
    return rows


# ------------------------------------------------------------------ per-shard summary
MOTION_KEYS = {"legged": ["slip_ratio", "cot", "joint_jerk_rms", "joint_jerk_peak", "peak_contact_force_bw",
                          "joint_limit_margin_min"],
               "arm": ["joint_jerk_rms", "joint_jerk_peak", "chunk_vel_step_max", "vel_step_any_max", "penetration_max_m",
                       "joint_limit_margin_min"]}


def _success(row, family):
    return bool(row["success"]) if family == "legged" else bool(row["privileged_success"])


def _fell(row, family):
    return bool(row["fell"]) if family == "legged" else row.get("outcome") == "dropped_off_table"


def summarize_shard(rows: list[dict], family: str) -> dict:
    n = len(rows)
    k = sum(_success(r, family) for r in rows)
    lo, hi = wilson(k, n)
    fl = sum(_fell(r, family) for r in rows)
    stages = {}
    for r in rows:
        st = r.get("failure_stage") if family == "legged" else (r.get("failed_stage") or "success")
        stages[str(st)] = stages.get(str(st), 0) + 1
    mot = {}
    for key in MOTION_KEYS[family]:
        xs = [r["motion"].get(key) for r in rows if r.get("motion") and r["motion"].get(key) is not None]
        mot[key] = float(np.median(xs)) if xs else None
    task = {}
    if family == "legged":
        task = dict(public_success=sum(bool(r["public_success"]) for r in rows),
                    mean_sim_time=float(np.mean([r["sim_time"] for r in rows])) if rows else None)
    else:
        task = dict(public_success=sum(bool(r["public_success"]) for r in rows),
                    median_min_tcp_cube_m=float(np.median([r["min_tcp_cube_m"] for r in rows])) if rows else None,
                    mean_steps=float(np.mean([r["steps"] for r in rows])) if rows else None)
    src = sorted({(r["source"] if isinstance(r["source"], str) else json.dumps(r["source"])) for r in rows})
    return dict(n=n, success=k, rate=(k / n if n else None), wilson95=[lo, hi], falls=fl, stages=stages, task=task,
                motion_median=mot, sources=src, seeds=sorted(r["seed"] for r in rows),
                success_by_seed={str(r["seed"]): _success(r, family) for r in rows})


# ------------------------------------------------------------------ run
def _job(args):
    family, route, robot, cond, seeds, out_dir, video = args
    import torch
    torch.set_num_threads(1)
    d = Path(out_dir) / route["name"] / robot
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{cond['key']}.jsonl"
    summ = out.with_suffix(".summary.json")
    if summ.exists() and json.loads(summ.read_text()).get("n") == len(seeds):
        return cond["key"], "skipped"
    t0 = time.time()
    if family == "legged":
        vd, vs = (Path(video[0]), video[1]) if video else (None, ())
        rows = run_legged_shard(route, robot, cond, seeds, out, video_dir=vd, video_seeds=vs)
    else:
        rows = run_arm_shard(route, robot, cond, seeds, out)
    rows = [r for r in rows if r["seed"] in set(seeds)]
    s = summarize_shard(rows, family)
    s.update(route=route, robot=robot, condition=dict(factor=cond["factor"], level=cond["level"], key=cond["key"]),
             perturbation=cond["pert"].to_dict(), wall_s=time.time() - t0, family=family)
    summ.write_text(json.dumps(s, indent=1, default=str))
    return cond["key"], f"{s['success']}/{s['n']}"


def cmd_run(a):
    family = a.family
    robots = a.robots.split(",")
    routes = [parse_route(r, family) for r in a.route]
    factors = set(a.factors.split(",")) if a.factors else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    manifest = dict(family=family, robots=robots, routes=routes, factors=factor_table(family), break_drop=BREAK_DROP,
                    seeds={}, checkpoints={}, contact_model=(os.environ.get("RRP_CONTACT_MODEL") if family == "legged" else "arm pick_place world (no contact version)"),
                    created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for r in routes:
        for k in ("ckpt", "flow", "rep"):
            if r.get(k):
                manifest["checkpoints"][r[k]] = _sha(r[k])
    for robot in robots:
        if family == "legged":
            from rrp.contracts.runs import parse_seed_spec
            seeds = list(parse_seed_spec(a.seeds))
        else:
            start, _, n = a.seeds.partition("+")
            seeds = feasible_arm_seeds(robot, int(start), int(n))
        manifest["seeds"][robot] = seeds
        for cond in conditions(family, robot, factors):
            for r in routes:
                jobs.append((family, r, robot, cond, seeds, str(out), None))
    mpath = out / "manifest.json"
    if not mpath.exists():
        mpath.write_text(json.dumps(manifest, indent=1))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        jobs = jobs[i::n]
    print(f"{len(jobs)} shard jobs, workers={a.workers}", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        res = map(_job, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4)
        res = pool.imap_unordered(_job, jobs)
    nd = 0
    for key, status in res:
        nd += 1
        print(f"[{nd}/{len(jobs)} {time.time() - t0:.0f}s] {key}: {status}", flush=True)


# ------------------------------------------------------------------ report
def break_points(levels: list[dict], nominal_rate: float, nominal_level) -> dict:
    """levels: [{level, rate}] of ONE factor. Returns {side: first level moving away from nominal whose rate is more
    than BREAK_DROP below nominal, or None}. Sides: 'low' (levels < nominal) and 'high' (levels > nominal); a factor
    whose nominal is not a numeric level (latency None) has only the 'high' side."""
    out = {}
    num = [l for l in levels if l["rate"] is not None]
    if isinstance(nominal_level, (int, float)):
        low = sorted([l for l in num if l["level"] < nominal_level], key=lambda l: -l["level"])
        high = sorted([l for l in num if l["level"] > nominal_level], key=lambda l: l["level"])
        sides = dict(low=low, high=high) if low else dict(high=high)
    else:
        sides = dict(high=sorted(num, key=lambda l: l["level"]))
    for side, ls in sides.items():
        out[side] = next((l["level"] for l in ls if l["rate"] < nominal_rate - BREAK_DROP - 1e-9), None)
    return out


def _paired(nom: dict, lev: dict) -> dict:
    lost = sum(1 for s, v in nom.items() if v and s in lev and not lev[s])
    gained = sum(1 for s, v in nom.items() if not v and s in lev and lev[s])
    return dict(lost=lost, gained=gained)


def build_report(out: Path) -> dict:
    man = json.loads((out / "manifest.json").read_text())
    family = man["family"]
    ft = man["factors"]
    rep = dict(family=family, manifest=man, routes={})
    for summ in sorted(out.glob("*/*/*.summary.json")):
        s = json.loads(summ.read_text())
        rn, robot = s["route"]["name"], s["robot"]
        rep["routes"].setdefault(rn, {}).setdefault(robot, {})[s["condition"]["key"]] = s
    table = {}
    for rn, per in rep["routes"].items():
        for robot, conds in per.items():
            nom = conds.get(NOMINAL)
            if nom is None:
                continue
            t = dict(nominal=dict(success=nom["success"], n=nom["n"], rate=nom["rate"], wilson95=nom["wilson95"],
                                  public_success=nom["task"]["public_success"],
                                  falls=nom["falls"], motion=nom["motion_median"], sources=nom["sources"]), factors={})
            for f, spec in list(ft.items()) + [("all_moderate", dict(levels=[1], nominal=0))]:
                lv = []
                for v in spec["levels"]:
                    key = "all_moderate" if f == "all_moderate" else f"{f}={_fmt(v)}"
                    c = conds.get(key)
                    if c is None:
                        continue
                    lv.append(dict(level=v, key=key, success=c["success"], n=c["n"], rate=c["rate"], wilson95=c["wilson95"],
                                   falls=c["falls"], drop=(nom["rate"] - c["rate"]), stages=c["stages"], task=c["task"],
                                   motion=c["motion_median"], **_paired(nom["success_by_seed"], c["success_by_seed"])))
                if lv:
                    pn = nom["task"]["public_success"] / nom["n"]
                    lvp = [dict(level=l["level"], rate=l["task"]["public_success"] / l["n"]) for l in lv]
                    t["factors"][f] = dict(levels=lv, nominal_level=spec["nominal"],
                                           break_point=break_points(lv, nom["rate"], spec["nominal"]),
                                           break_point_public=break_points(lvp, pn, spec["nominal"]))
            allk = [l for f, x in t["factors"].items() if f != "all_moderate" for l in x["levels"]]
            ks, ns = sum(l["success"] for l in allk), sum(l["n"] for l in allk)
            t["pooled_perturbed"] = dict(success=ks, n=ns, rate=(ks / ns if ns else None), wilson95=list(wilson(ks, ns)),
                                         lost=sum(l["lost"] for l in allk), gained=sum(l["gained"] for l in allk))
            table.setdefault(rn, {})[robot] = t
    rep["table"] = table
    return rep


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}"


def render_md(rep: dict) -> str:
    fam = rep["family"]
    L = [f"# robustness sweep ({fam})", "",
         "Cells: privileged success k/n (public task-runtime success in parentheses; fN = N falls"
         + (", i.e. dropped cube)" if fam == "arm" else ")") + ". ",
         f"Break-point = first level (moving away from nominal) whose success rate is > {int(BREAK_DROP * 100)} points below "
         "nominal; `lost/gained` = paired seeds that succeed at nominal and fail at the level / the reverse. Wilson 95% CIs. "
         "Motion columns are medians over episodes. Checkpoint sha256: see manifest.", ""]
    mk = MOTION_KEYS[fam]
    L += cross_route_md(rep)
    for rn, per in rep["table"].items():
        for robot, t in per.items():
            nm = t["nominal"]
            L.append(f"## {rn} on {robot}  (sources: {', '.join(nm['sources'])})")
            L.append("")
            L.append(f"nominal {nm['success']}/{nm['n']} [{_pct(nm['wilson95'][0])}, {_pct(nm['wilson95'][1])}] falls "
                     f"{nm['falls']}; pooled over perturbed levels {t['pooled_perturbed']['success']}/"
                     f"{t['pooled_perturbed']['n']} (lost {t['pooled_perturbed']['lost']}, gained {t['pooled_perturbed']['gained']})")
            L.append("")
            L.append("| factor | level | success | 95% CI | falls | lost/gained | " + " | ".join(mk) + " | break-point |")
            L.append("|---|---|---|---|---|---|" + "---|" * len(mk) + "---|")
            nmv = " | ".join("-" if nm["motion"].get(x) is None else f"{nm['motion'][x]:.3g}" for x in mk)
            L.append(f"| nominal | - | {nm['success']}/{nm['n']} | {_pct(nm['wilson95'][0])}-{_pct(nm['wilson95'][1])} | "
                     f"{nm['falls']} | - | {nmv} | |")
            for f, x in t["factors"].items():
                bp = ", ".join(f"{k}: {_fmt(v) if v is not None else 'none'}" for k, v in x["break_point"].items())
                for i, l in enumerate(x["levels"]):
                    mv = " | ".join("-" if l["motion"].get(k) is None else f"{l['motion'][k]:.3g}" for k in mk)
                    flag = " **" if l["rate"] < nm["rate"] - BREAK_DROP - 1e-9 else ""
                    L.append(f"| {f} | {_fmt(l['level'])} | {l['success']}/{l['n']}{flag} | {_pct(l['wilson95'][0])}-"
                             f"{_pct(l['wilson95'][1])} | {l['falls']} | {l['lost']}/{l['gained']} | {mv} | "
                             f"{bp if i == 0 else ''} |")
            L.append("")
    return "\n".join(L)


def cross_route_md(rep: dict) -> list[str]:
    """One table per robot: success k/n of every route at every level side by side, then break-points per route."""
    L = []
    robots = sorted({r for per in rep["table"].values() for r in per})
    for robot in robots:
        routes = [rn for rn in rep["table"] if robot in rep["table"][rn]]
        T = {rn: rep["table"][rn][robot] for rn in routes}
        L += [f"## cross-route summary: {robot}", "",
              "| factor | level | " + " | ".join(routes) + " |", "|---|---|" + "---|" * len(routes)]
        L.append("| nominal | - | " + " | ".join(f"{T[r]['nominal']['success']}/{T[r]['nominal']['n']} "
                                                 f"({T[r]['nominal']['public_success']})" for r in routes) + " |")
        facs = []
        for r in routes:
            for f in T[r]["factors"]:
                if f not in facs:
                    facs.append(f)
        for f in facs:
            levels = []
            for r in routes:
                for l in T[r]["factors"].get(f, {}).get("levels", []):
                    if l["level"] not in levels:
                        levels.append(l["level"])
            for v in levels:
                cells = []
                for r in routes:
                    l = next((x for x in T[r]["factors"].get(f, {}).get("levels", []) if x["level"] == v), None)
                    if l is None:
                        cells.append("-")
                    else:
                        broke = l["rate"] < T[r]["nominal"]["rate"] - BREAK_DROP - 1e-9
                        cells.append(f"{'**' if broke else ''}{l['success']}/{l['n']}{'**' if broke else ''}"
                                     + f" ({l['task']['public_success']})" + (f" f{l['falls']}" if l["falls"] else ""))
                L.append(f"| {f} | {_fmt(v)} | " + " | ".join(cells) + " |")
        L.append("| pooled perturbed | - | " + " | ".join(
            f"{T[r]['pooled_perturbed']['success']}/{T[r]['pooled_perturbed']['n']} (lost {T[r]['pooled_perturbed']['lost']})"
            for r in routes) + " |")
        L += ["", f"Break-points ({robot}; first level > {int(BREAK_DROP * 100)} points below the route's own nominal; "
              "privileged success, **bold** cells above; [public-success break-point in brackets]):", "",
              "| factor | " + " | ".join(routes) + " |", "|---|" + "---|" * len(routes)]
        fb = lambda bp: "-" if bp is None else ", ".join(f"{k} {_fmt(v) if v is not None else 'none'}" for k, v in bp.items())
        for f in facs:
            cells = []
            for r in routes:
                x = T[r]["factors"].get(f, {})
                cells.append(f"{fb(x.get('break_point'))} [{fb(x.get('break_point_public'))}]" if x else "-")
            L.append(f"| {f} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def cmd_report(a):
    out = Path(a.out)
    rep = build_report(out)
    (out / "robustness_report.json").write_text(json.dumps(rep["table"], indent=1, default=str))
    (out / "robustness_report.md").write_text(render_md(rep))
    print(render_md(rep))


# ------------------------------------------------------------------ labelled failure videos
def _cond_by_key(family, robot, key):
    return next(c for c in conditions(family, robot) if c["key"] == key)


def render_video(family: str, route: dict, robot: str, key: str, seed: int, out_dir: Path, sweep_out: Path | None = None,
                 fps: int = 25) -> dict:
    """Re-run ONE sweep episode exactly (legged: per-seed controller RNG; arm: the same seed batches in order, since the
    flow RNG is shared by a batch) and save a captioned mp4 + an artifacts/video/INDEX.md line. The caption names the
    controller source and the physics condition."""
    import datetime as dt
    import re
    import imageio
    os.environ.setdefault("MUJOCO_GL", "egl")
    cond = _cond_by_key(family, robot, key)
    out_dir.mkdir(parents=True, exist_ok=True)
    pert_s = json.dumps({k: v for k, v in cond["pert"].to_dict().items()
                         if k != "version" and v not in (None, 0.0, 1.0, [0.0, 0.0, 0.0])})
    if family == "legged":
        import torch
        from rrp.evaluation.legged_latent_eval import BCController, LatentLeggedController, run_episode, _caption
        dev = torch.device("cpu")
        ctl = (BCController(Path(route["ckpt"]), dev, nfe=8, replan=5, seed=seed) if route["kind"] == "bc" else
               LatentLeggedController(Path(route["ckpt"]), dev, nfe=8, edit="none", t_edit=1.0, seed=seed)
               if route["kind"] == "flow" else None)
        row, frames = run_episode(ctl, robot, seed, 60.0, video=True, perturb=cond["pert"])
        src = row["source"]
        ok, tag = row["success"], ("success" if row["success"] else ("fell" if row["fell"] else f"failure-{row['failure_stage']}"))
        lab = dict(teacher="SCRIPTED TEACHER (privileged)", bc="LEARNED BC (positive control)",
                   flow="LEARNED latent sys-i -> packet -> sys-0")[route["kind"]]
        imgs = [_caption(f, [f"{lab} | {src}"[:90], f"ROBUSTNESS {key} {pert_s}"[:90], f"{robot} waypoint_contact seed {seed} | {st}"[:90]])
                for f, st in frames]
    else:
        import mujoco
        from rrp.evaluation.ladder import LadderConfig, load_models, run_ladder
        from rrp.evaluation.legged_latent_eval import _caption
        man = json.loads((sweep_out / "manifest.json").read_text()) if sweep_out else None
        seeds = man["seeds"][robot] if man else feasible_arm_seeds(robot, 3_000_000, 20)
        kind = route["kind"]
        cfg = LadderConfig(route=kind, robot=robot, seeds=list(seeds), representation=route.get("rep"), flow=route.get("flow"),
                           policy=route.get("ckpt"), policy_label=(Path(route["ckpt"]).stem if route.get("ckpt") else None),
                           replan_ticks=8, max_steps=300, nfe=8, compare_oracle=False, device="cpu", prev_action="zero",
                           perturb=None if cond["factor"] == NOMINAL else cond["pert"])
        models, ids = load_models(cfg) if kind != "teacher" else (dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None), {})
        lab = dict(teacher="SCRIPTED TEACHER (privileged)", learned="LEARNED plain BC",
                   generated="LEARNED sys-i flow -> packet -> sys-0 (frozen arm route)")[kind]
        imgs, rend, row = [], {}, None
        for i in range(0, len(seeds), 16):
            b = list(seeds[i:i + 16])
            tgt = b.index(seed) if seed in b else None

            def cb(k, s, step, phase, tgt=tgt):
                if k != tgt:
                    return
                if "r" not in rend:
                    rend["r"] = mujoco.Renderer(s.model, 360, 480)
                rend["r"].update_scene(s.data, camera="front")
                st = " ".join(f"{e}:{v.status}" for e, v in s.runtime.instances.items())
                imgs.append(_caption(rend["r"].render().copy(), [f"{lab}"[:90], f"ROBUSTNESS {key} {pert_s}"[:90],
                                                                 f"{robot} pick_place seed {seed} t={s.data.time:.1f}s {st}"[:90]]))
            rows = run_ladder(replace(cfg, seeds=b), None, models, ids, frame_cb=cb if tgt is not None else None)
            if tgt is not None:
                row = rows[tgt]
                break
        src = row["source"]
        ok = row["privileged_success"]
        tag = "success" if ok else f"failure-{row['failed_stage']}"
    s_ = re.sub(r"[^A-Za-z0-9_.+-]+", "-", f"{route['name']}")[:40]
    name = f"{dt.date.today()}_robust_{s_}_{robot}_{key.replace('=', '')}_s{seed}_{tag}.mp4"
    imageio.mimsave(out_dir / name, imgs, fps=fps, quality=6)
    with open(out_dir / "INDEX.md", "a") as fh:
        fh.write(f"- `{name}` — W6 robustness sweep, source={src} (route {route['name']}) robot={robot} "
                 f"task={'waypoint_contact' if family == 'legged' else 'pick_place'} seed={seed} condition={key} "
                 f"{pert_s} outcome={tag} (privileged evaluator)\n")
    return dict(video=name, success=bool(ok), outcome=tag, source=src)


def cmd_video(a):
    route = parse_route(a.route, a.family)
    r = render_video(a.family, route, a.robot, a.condition, a.seed, Path(a.video_dir), Path(a.out) if a.out else None)
    print(json.dumps(r))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=["legged", "arm"], required=True)
    r.add_argument("--robots", required=True)
    r.add_argument("--route", action="append", required=True, help="NAME=SPEC (see module doc)")
    r.add_argument("--seeds", required=True, help="legged: seed spec (10000-10019); arm: START+N feasible seeds")
    r.add_argument("--factors", default=None, help="comma list (default: all factors + all_moderate)")
    r.add_argument("--out", required=True)
    r.add_argument("--shard", default=None, help="i/n: run every n-th shard job starting at i")
    r.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    v = sub.add_parser("video")
    v.add_argument("--family", choices=["legged", "arm"], required=True)
    v.add_argument("--route", required=True)
    v.add_argument("--robot", required=True)
    v.add_argument("--condition", required=True, help="condition key, e.g. push_dv=1.5")
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--video-dir", default="artifacts/video")
    v.add_argument("--out", default=None, help="sweep output dir (arm: seed list from its manifest)")
    a = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report, "video": cmd_video}[a.cmd](a)


if __name__ == "__main__":
    main()
