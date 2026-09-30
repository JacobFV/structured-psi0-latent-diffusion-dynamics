"""Independent tracker validation + eligibility freeze (BEFORE any learned high-level policy).

Protocol per body (fixed seeds, fixed command scripts, no pushes unless stated):
  stand      : zero command, 6 s
  forward    : vx = 0.6 * vx_max, 10 s
  turn       : wz = 0.6 * wz_max, 8 s
  arc        : vx = 0.5 * vx_max, wz = 0.4 * wz_max, 10 s
  push_fwd   : forward + a 0.3 m/s lateral base kick at t=4 s (0.15 for bipeds)
Metrics: falls (height < min / tilt > limit / non-foot ground contact), forward-speed tracking
ratio (achieved/commanded mean body-frame vx over the last 60%), yaw-rate ratio, distance,
slip (mean contact-foot horizontal speed), cost of transport (sum|tau*qdot| dt / (m g d)).
Gate (frozen in eligibility.json): no fall in >= 90% of episodes AND forward ratio in [0.5, 1.5]
AND turn ratio in [0.4, 1.6] AND not falling while standing.

Contact/gait metrics (protocol v2, 2026-09-26, contact track; measured the same way for every tracker):
  slip_cp_mps    normal-force-weighted horizontal speed of each foot AT ITS FLOOR CONTACT POINTS, averaged over
                 (tick, loaded foot) with foot load > 2% of body weight (true stance slip; `slip_mps` is the legacy
                 foot-body-origin speed of any touching foot)
  slip_ratio     slip_cp_mps / mean horizontal body speed (same ticks, last 60%)
  duty_factor    per foot, fraction of ticks in contact (last 60%); `duty_min/max` over feet
  air_time_s     mean duration of completed swings; `swing_apex_m` mean apex foot-site clearance per swing
  step_hz        touchdowns per second per foot
contact gate: the gate above AND forward slip_ratio < 0.15 AND every foot steps in the forward trial
(0.3 <= duty <= 0.9) AND mean forward swing apex >= 0.3 * the body's swing_height target.

usage: python -m rrp.cli suite tracker-validation --body go2 --kind learned [--actor path] --seeds 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from rrp.envs.mujoco.legged_core import LeggedBinding, quat_rotate_inv, yaw_of
from rrp.envs.mujoco.legged_tracker import CPGTracker, LearnedTracker, TRACKER_DIR
from rrp.bodies.legged import legged_body, standalone_model


def scripts(b: LeggedBinding):
    r = b.cmd_ranges
    vx, wz = r["vx"][1], r["wz"][1]
    kick = 0.15 if b.biped else 0.3
    return {
        "stand": dict(T=6.0, cmd=[0, 0, 0]),
        "forward": dict(T=10.0, cmd=[0.6 * vx, 0, 0]),
        "turn": dict(T=8.0, cmd=[0, 0, 0.6 * wz]),
        "turn_fast": dict(T=8.0, cmd=[0, 0, 0.8 * wz]),      # the W8 waypoint teacher's pure-turn rate (reported, not in the gate)
        "arc": dict(T=10.0, cmd=[0.5 * vx, 0, 0.4 * wz]),
        "push_fwd": dict(T=10.0, cmd=[0.6 * vx, 0, 0], push=(4.0, kick)),
    }


def run_episode(model, b: LeggedBinding, tracker, script: dict, seed: int, record=False, act=None):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    b.set_default(d, yaw=rng.uniform(-math.pi, math.pi), noise=0.03, rng=rng)
    mujoco.mj_forward(model, d)
    tracker.reset(phase=0.0)
    if act is not None:
        act.reset(0, d.ctrl[b.pol_act].copy())
    dt = 0.02
    sub = max(1, int(round(dt / model.opt.timestep)))
    steps = int(script["T"] / dt)
    cmd = np.array(script["cmd"], float)
    mass = float(model.body_subtreemass[b.root_bid])
    vxs, wzs, slips, energy = [], [], [], 0.0
    cp_slip, speeds, contact_hist = [], [], []
    air = np.zeros(b.nf)
    apex = np.zeros(b.nf)
    swings, apexes, touchdowns = [], [], 0
    w_load = 0.02 * mass * 9.81
    p0 = d.qpos[b.qa:b.qa + 2].copy()
    fell, fell_t = False, None
    traj = []
    peak_fn, peak_raw, margin = 0.0, 0.0, float("inf")
    fwin = max(1, int(round(FORCE_WINDOW_S / model.opt.timestep)))      # 20 ms moving average (D-112 revision)
    fbuf = []
    rng_j = np.maximum(b.jhi - b.jlo, 1e-9)
    lim_j = b.jhi > b.jlo
    for k in range(steps):
        t = k * dt
        if "push" in script and abs(t - script["push"][0]) < dt / 2:
            yaw = yaw_of(d.qpos[b.qa + 3:b.qa + 7])
            s = script["push"][1]
            d.qvel[b.da:b.da + 2] += [-math.sin(yaw) * s, math.cos(yaw) * s]
        tgt = tracker.act(d, cmd)
        if act is not None:
            act.command(0, tgt)
        else:
            d.ctrl[b.pol_act] = tgt
        for _ in range(sub):
            if act is not None:
                d.ctrl[b.pol_act] = act.substep_ctrl(0, d)
            mujoco.mj_step(model, d)
            energy += float(np.sum(np.abs(d.actuator_force[b.pol_act] * d.qvel[b.pol_dadr]))) * model.opt.timestep
            fn_s = b.stance(d)[1]                                            # W6 gate: per-foot normal force (N)
            peak_raw = max(peak_raw, float(np.max(fn_s)))
            fbuf.append(fn_s)
            if len(fbuf) > fwin:
                fbuf.pop(0)
            if len(fbuf) == fwin:
                peak_fn = max(peak_fn, float(np.max(np.mean(fbuf, axis=0))))
        v = b.base_lin_vel_body(d)
        w = d.qvel[b.da + 3:b.da + 6]
        fc, fn, slip_cp, bad = b.stance(d)
        clr = b.foot_clearance(d)
        qj = d.qpos[b.pol_qadr]
        if lim_j.any():
            margin = min(margin, float(np.min((np.minimum(qj - b.jlo, b.jhi - qj) / rng_j)[lim_j])))
        if k > 0.4 * steps:
            vxs.append(v[0])
            wzs.append(w[2])
            speeds.append(float(np.hypot(*d.qvel[b.da:b.da + 2])))
            loaded = fn > w_load
            cp_slip += [float(x) for x in slip_cp[loaded]]
            contact_hist.append(fc.copy())
            first = fc & (air > 0)
            for i in np.flatnonzero(first):
                swings.append(air[i])
                apexes.append(apex[i])
                touchdowns += 1
        apex = np.where(fc, 0.0, np.maximum(apex, clr))
        air = np.where(fc, 0.0, air + dt)
        for i, fb in enumerate(b.foot_bids):
            if fc[i]:
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, fb, vel, 0)
                slips.append(float(np.linalg.norm(vel[3:5])))
        if record and k % 5 == 0:
            traj.append([round(t, 2), *map(float, d.qpos[b.qa:b.qa + 3])])
        if bad or d.qpos[b.qa + 2] < b.min_h or b.tilt(d) > b.tilt_limit or not np.isfinite(d.qpos).all():
            fell, fell_t = True, t
            break
    dist = float(np.linalg.norm(d.qpos[b.qa:b.qa + 2] - p0))
    ch = np.array(contact_hist) if contact_hist else np.zeros((0, b.nf))
    duty = ch.mean(0) if len(ch) else np.full(b.nf, np.nan)
    slip_cp_m = float(np.mean(cp_slip)) if cp_slip else None
    spd = float(np.mean(speeds)) if speeds else None
    gait = dict(slip_cp_mps=slip_cp_m, body_speed_mps=spd,
                slip_ratio=(slip_cp_m / max(spd, 0.02)) if (slip_cp_m is not None and spd is not None) else None,
                duty_factor=[float(x) for x in duty], duty_min=float(np.min(duty)) if len(ch) else None,
                duty_max=float(np.max(duty)) if len(ch) else None,
                air_time_s=float(np.mean(swings)) if swings else 0.0,
                swing_apex_m=float(np.mean(apexes)) if apexes else 0.0,
                step_hz=touchdowns / max(len(ch) * dt, 1e-9) / b.nf if len(ch) else 0.0)
    out = dict(fell=fell, fell_t=fell_t, dist_m=dist, mean_vx=float(np.mean(vxs)) if vxs else None,
               mean_wz=float(np.mean(wzs)) if wzs else None, slip_mps=float(np.mean(slips)) if slips else None,
               cot=float(energy / (mass * 9.81 * max(dist, 1e-3))) if dist > 0.2 else None,
               peak_force_bw=peak_fn / (mass * 9.81), peak_force_raw_bw=peak_raw / (mass * 9.81), joint_limit_margin_min=margin if math.isfinite(margin) else None,
               cmd=cmd.tolist(), **gait)
    if record:
        out["traj"] = traj
    return out


def validate(body: str, kind: str, actor: str | None, seeds: int, contact: str | None = "v1", actuator: str = "v1",
             latency_ms: float = 0.0, robust: bool = False) -> dict:
    from rrp.envs.mujoco.legged_tracker import tracker_path
    mod = legged_body(body)
    model, _, meta = standalone_model(mod, contact=contact)
    b = LeggedBinding(model, meta)
    if kind == "learned":
        path = Path(actor) if actor else tracker_path(body, contact)
        tracker = LearnedTracker(path, b, body)
        tsha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    else:
        tracker = CPGTracker(b, meta)
        tsha = "scripted"
    act = None
    if actuator in ("v2", "v1lat"):
        from rrp.bodies.actuator import ActuatorModel
        act = ActuatorModel(model, b, 1, None, name=meta["name"], randomize=False, latency_ms=latency_ms, mode=actuator)
    res = {}
    t0 = time.time()
    for name, sc in scripts(b).items():
        res[name] = [run_episode(model, b, tracker, sc, 1000 + s, act=act) for s in range(seeds)]
    eps = [e for v in res.values() for e in v]
    no_fall = float(np.mean([not e["fell"] for e in eps]))
    fwd = [e["mean_vx"] / e["cmd"][0] for e in res["forward"] if e["mean_vx"] is not None and not e["fell"]]
    trn = [e["mean_wz"] / e["cmd"][2] for e in res["turn"] if e["mean_wz"] is not None and not e["fell"]]
    arc = [e["mean_wz"] / e["cmd"][2] for e in res["arc"] if e["mean_wz"] is not None and not e["fell"]]
    tf = [e["mean_wz"] / e["cmd"][2] for e in res.get("turn_fast", []) if e["mean_wz"] is not None and not e["fell"]]
    stand_ok = not any(e["fell"] for e in res["stand"])
    fr = float(np.mean(fwd)) if fwd else 0.0
    tr = float(np.mean(trn)) if trn else 0.0
    gate = dict(no_fall_rate=no_fall, forward_ratio=fr, turn_ratio=tr, stand_ok=stand_ok,
                arc_yaw_ratio=float(np.mean(arc)) if arc else 0.0, turn_fast_ratio=float(np.mean(tf)) if tf else None,
                passed=bool(no_fall >= 0.9 and 0.5 <= fr <= 1.5 and 0.4 <= tr <= 1.6 and stand_ok))
    summary = {k: dict(fall_rate=float(np.mean([e["fell"] for e in v])),
                       dist_m=float(np.mean([e["dist_m"] for e in v])),
                       mean_vx=_nanmean([e["mean_vx"] for e in v]), mean_wz=_nanmean([e["mean_wz"] for e in v]),
                       slip_mps=_nanmean([e["slip_mps"] for e in v]), cot=_nanmean([e["cot"] for e in v]),
                       **{g: _nanmean([e[g] for e in v]) for g in GAIT_KEYS},
                       peak_force_bw=_nanmax([e.get("peak_force_bw") for e in v]),
                       peak_force_raw_bw=_nanmax([e.get("peak_force_raw_bw") for e in v]),
                       joint_limit_margin_min=_nanmin([e.get("joint_limit_margin_min") for e in v]),
                       cmd=v[0]["cmd"]) for k, v in res.items()}
    fw = summary["forward"]
    fw_duty = [x for e in res["forward"] if not e["fell"] for x in e["duty_factor"]]
    sr = fw["slip_ratio"]
    cg = dict(slip_ratio=sr, slip_ok=bool(sr is not None and sr < 0.15),
              duty_min=min(fw_duty) if fw_duty else None, duty_max=max(fw_duty) if fw_duty else None,
              stepping_ok=bool(fw_duty and min(fw_duty) >= 0.3 and max(fw_duty) <= 0.9),
              swing_apex_m=fw["swing_apex_m"], swing_target_m=b.swing_height,
              clearance_ok=bool((fw["swing_apex_m"] or 0) >= 0.3 * b.swing_height))
    cg["passed"] = bool(gate["passed"] and cg["slip_ok"] and cg["stepping_ok"] and cg["clearance_ok"])
    gate["contact_gate"] = cg
    out = dict(body=body, tracker_kind=kind, tracker_source=tracker.source, tracker_version=tracker.version,
                tracker_sha=tsha, family=meta["family"], synthetic=meta.get("synthetic", False),
                seeds=seeds, gate=gate, summary=summary, episodes=res, wall_s=time.time() - t0,
                contact_model=meta["contact_model"], actuator=actuator, actuator_limits=meta.get("actuator_limits"),
                tracker_actuator_limits=getattr(tracker, "actuator_limits", None),
                limits_mismatch_override=getattr(tracker, "limits_override", False), latency_ms=latency_ms if actuator != "v1" else None,
                actuator_params=act.params if act is not None else None, tracker_contact_model=getattr(tracker, "contact_model", None),
                **({"actuator_speed_estimated": act.speed_estimated} if act is not None else {}),
                protocol="rrp.control.tracker_validation/v2", mujoco=mujoco.__version__)
    if robust:
        out["robustness"] = robust_check(body, actor, kind, seeds, contact, fr)
    from rrp.harness.eval.gates import check_tracker
    out["w6_gate"] = check_tracker(out)            # D-112 gates (pass | fail | incomplete, per criterion)
    return out


def training_range(tracker, contact: str | None) -> dict:
    """The randomization a learned tracker was TRAINED under (contact_v2 ContactRandomizer + the trainer's pushes +
    the optional actuator mode), used by the W6 robustness gate. Levels sit inside the range, near its edges."""
    from rrp.bodies.contact import CONTACT_MODELS, resolve
    c = resolve(contact)
    R = CONTACT_MODELS[c].get("randomization") or {}
    meta = getattr(tracker, "meta", {}) or {}
    act = (meta.get("args") or {}).get("actuator") or meta.get("actuator") or "v1"
    lat_hi = 30.0 if act in ("v1lat", "v2") else 2.0 * (R.get("latency_substeps") or [0, 0])[1]   # ms (dt 2 ms)
    mu = R.get("mu") or [1.0, 1.0]
    ms = R.get("root_mass_scale") or [1.0, 1.0]
    return dict(contact=c, mu=[round(mu[0] + 0.05, 3), round(mu[1] - 0.05, 3)], root_mass_scale=list(ms),
                latency_ms=lat_hi, train_actuator=act, push_mps=None)


def robust_check(body: str, tracker_path_: str | None, kind: str, seeds: int, contact: str | None,
                 nominal_forward_ratio: float) -> dict:
    """W6 gate: the forward trial under each condition INSIDE the tracker's training randomization (floor mu near both
    ends, root mass x0.9 / x1.1, v1lat latency at the trained maximum, a lateral kick at the training magnitude
    0.4 m/s (biped 0.2) at t = 4 s). Reports no-fall rate and forward ratio per condition."""
    from rrp.bodies.contact import floor_friction
    from rrp.envs.mujoco.legged_tracker import tracker_path
    out, rng_info = {}, None
    base = scripts
    for cond in ("mu_lo", "mu_hi", "mass_lo", "mass_hi", "latency", "push"):
        mod = legged_body(body)
        model, _, meta = standalone_model(mod, contact=contact)
        b = LeggedBinding(model, meta)
        tracker = LearnedTracker(Path(tracker_path_) if tracker_path_ else tracker_path(body, contact), b, body) \
            if kind == "learned" else CPGTracker(b, meta)
        tr = training_range(tracker, contact)
        rng_info = tr
        tr["push_mps"] = 0.2 if b.biped else 0.4
        act = None
        if cond in ("mu_lo", "mu_hi"):
            model.geom_friction[b.floor] = floor_friction(tr["mu"][0 if cond == "mu_lo" else 1], contact or "v2")
        elif cond in ("mass_lo", "mass_hi"):
            s = tr["root_mass_scale"][0 if cond == "mass_lo" else 1]
            model.body_mass[b.root_bid] *= s
            model.body_inertia[b.root_bid] *= s
            mujoco.mj_setConst(model, mujoco.MjData(model))
        elif cond == "latency":
            from rrp.bodies.actuator import ActuatorModel
            act = ActuatorModel(model, b, 1, None, name=meta["name"], randomize=False, latency_ms=tr["latency_ms"],
                                mode="v1lat")
        sc = dict(base(b)["forward"])
        if cond == "push":
            sc["push"] = (4.0, tr["push_mps"])
        eps = [run_episode(model, b, tracker, sc, 1000 + s, act=act) for s in range(seeds)]
        fwd = [e["mean_vx"] / e["cmd"][0] for e in eps if e["mean_vx"] is not None and not e["fell"]]
        out[cond] = dict(no_fall_rate=float(np.mean([not e["fell"] for e in eps])),
                         forward_ratio=float(np.mean(fwd)) if fwd else 0.0,
                         slip_ratio=_nanmean([e["slip_ratio"] for e in eps]), n=len(eps))
    return dict(conditions=out, nominal_forward_ratio=nominal_forward_ratio, training_range=rng_info,
                protocol="rrp.evaluation.tracker_validation.robust_check/v1 (W6, D-112)")


FORCE_WINDOW_S = 0.02    # peak foot force = peak of the 20 ms moving average (impact transients filtered; lead, D-112 revision)
GATE_EXIT = 86          # exit code of a validation whose W6 gate verdict is "fail" (the pipeline marks the node failed)

GAIT_KEYS = ("slip_cp_mps", "body_speed_mps", "slip_ratio", "duty_min", "duty_max", "air_time_s", "swing_apex_m",
             "step_hz")


def _nanmax(xs):
    xs = [x for x in xs if x is not None]
    return float(max(xs)) if xs else None


def _nanmin(xs):
    xs = [x for x in xs if x is not None]
    return float(min(xs)) if xs else None


def _nanmean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--kind", default="learned", choices=["learned", "cpg"])
    ap.add_argument("--actor")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--contact", default="v1", help="physics contact model to validate in (v1 | v2)")
    ap.add_argument("--actuator", default=None, help="v1 (= ideal) PD | v1lat | v2 rrp.bodies.actuator (nominal params, fixed "
                    "latency); default $RRP_ACTUATOR_MODE, else rrp.bodies.actuator.ACTUATOR_MODE_DEFAULT (ideal)")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="actuation latency for --actuator v2")
    ap.add_argument("--freeze", action="store_true", help="write eligibility.json next to the frozen tracker")
    ap.add_argument("--gate-dir", default=None, help="also write the W6 gate report (gate_report.json) here")
    ap.add_argument("--gate-exit", action="store_true", help=f"exit {GATE_EXIT} when the W6 gate verdict is fail")
    ap.add_argument("--robust", action="store_true", help="W6 gate: forward trial under the tracker's training randomization")
    a = ap.parse_args(argv)
    from rrp.bodies.actuator import legacy_mode_name
    a.actuator = legacy_mode_name(a.actuator)      # D-126 #14: canonical name; the default stays "v1" (records unchanged)
    r = validate(a.body, a.kind, a.actor, a.seeds, a.contact, a.actuator, a.latency_ms, robust=a.robust)
    print(json.dumps(dict(body=r["body"], kind=r["tracker_kind"], gate=r["gate"], summary=r["summary"],
                          w6_gate={k: r["w6_gate"][k] for k in ("verdict", "failed", "not_evaluated")}), indent=1))
    from rrp.bodies.contact import resolve
    sub = "" if resolve(a.contact) == "v1" else f"contact_{resolve(a.contact)}/"
    out = Path(a.out) if a.out else TRACKER_DIR / a.body / f"{sub}validation_{a.kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    if a.freeze:
        el = dict(body=a.body, tracker_kind=a.kind, tracker_sha=r["tracker_sha"], tracker_version=r["tracker_version"],
                  eligible=r["gate"]["passed"], gate=r["gate"], w6_gate_verdict=r["w6_gate"]["verdict"],
                  w6_gate_failed=r["w6_gate"]["failed"], frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  validation_file=str(out.name), note="frozen on independent tracker validation, before any "
                  "learned high-level policy result")
        (out.parent / f"eligibility_{a.kind}.json").write_text(json.dumps(el, indent=1))
    if a.gate_dir:
        from rrp.harness.eval.gates import write_report
        write_report(r["w6_gate"], Path(a.gate_dir))
    if a.gate_exit and r["w6_gate"]["verdict"] == "fail":
        sys.exit(GATE_EXIT)
