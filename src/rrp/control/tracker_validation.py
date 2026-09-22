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

usage: python -m rrp.control.tracker_validation --body go2 --kind learned [--actor path] --seeds 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from rrp.control.legged_core import LeggedBinding, quat_rotate_inv, yaw_of
from rrp.control.legged_tracker import CPGTracker, LearnedTracker, TRACKER_DIR
from rrp.morphology.legged import legged_body, standalone_model


def scripts(b: LeggedBinding):
    r = b.cmd_ranges
    vx, wz = r["vx"][1], r["wz"][1]
    kick = 0.15 if b.biped else 0.3
    return {
        "stand": dict(T=6.0, cmd=[0, 0, 0]),
        "forward": dict(T=10.0, cmd=[0.6 * vx, 0, 0]),
        "turn": dict(T=8.0, cmd=[0, 0, 0.6 * wz]),
        "arc": dict(T=10.0, cmd=[0.5 * vx, 0, 0.4 * wz]),
        "push_fwd": dict(T=10.0, cmd=[0.6 * vx, 0, 0], push=(4.0, kick)),
    }


def run_episode(model, b: LeggedBinding, tracker, script: dict, seed: int, record=False):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    b.set_default(d, yaw=rng.uniform(-math.pi, math.pi), noise=0.03, rng=rng)
    mujoco.mj_forward(model, d)
    tracker.reset(phase=0.0)
    dt = 0.02
    sub = max(1, int(round(dt / model.opt.timestep)))
    steps = int(script["T"] / dt)
    cmd = np.array(script["cmd"], float)
    mass = float(model.body_subtreemass[b.root_bid])
    vxs, wzs, slips, energy = [], [], [], 0.0
    p0 = d.qpos[b.qa:b.qa + 2].copy()
    fell, fell_t = False, None
    traj = []
    for k in range(steps):
        t = k * dt
        if "push" in script and abs(t - script["push"][0]) < dt / 2:
            yaw = yaw_of(d.qpos[b.qa + 3:b.qa + 7])
            s = script["push"][1]
            d.qvel[b.da:b.da + 2] += [-math.sin(yaw) * s, math.cos(yaw) * s]
        d.ctrl[b.pol_act] = tracker.act(d, cmd)
        for _ in range(sub):
            mujoco.mj_step(model, d)
            energy += float(np.sum(np.abs(d.actuator_force[b.pol_act] * d.qvel[b.pol_dadr]))) * model.opt.timestep
        v = b.base_lin_vel_body(d)
        w = d.qvel[b.da + 3:b.da + 6]
        fc, bad = b.contacts(d)
        if k > 0.4 * steps:
            vxs.append(v[0])
            wzs.append(w[2])
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
    out = dict(fell=fell, fell_t=fell_t, dist_m=dist, mean_vx=float(np.mean(vxs)) if vxs else None,
               mean_wz=float(np.mean(wzs)) if wzs else None, slip_mps=float(np.mean(slips)) if slips else None,
               cot=float(energy / (mass * 9.81 * max(dist, 1e-3))) if dist > 0.2 else None,
               cmd=cmd.tolist())
    if record:
        out["traj"] = traj
    return out


def validate(body: str, kind: str, actor: str | None, seeds: int) -> dict:
    mod = legged_body(body)
    model, _, meta = standalone_model(mod)
    b = LeggedBinding(model, meta)
    if kind == "learned":
        path = Path(actor) if actor else TRACKER_DIR / body / "actor.pt"
        tracker = LearnedTracker(path, b, body)
        tsha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    else:
        tracker = CPGTracker(b, meta)
        tsha = "scripted"
    res = {}
    t0 = time.time()
    for name, sc in scripts(b).items():
        res[name] = [run_episode(model, b, tracker, sc, 1000 + s) for s in range(seeds)]
    eps = [e for v in res.values() for e in v]
    no_fall = float(np.mean([not e["fell"] for e in eps]))
    fwd = [e["mean_vx"] / e["cmd"][0] for e in res["forward"] if e["mean_vx"] is not None and not e["fell"]]
    trn = [e["mean_wz"] / e["cmd"][2] for e in res["turn"] if e["mean_wz"] is not None and not e["fell"]]
    stand_ok = not any(e["fell"] for e in res["stand"])
    fr = float(np.mean(fwd)) if fwd else 0.0
    tr = float(np.mean(trn)) if trn else 0.0
    gate = dict(no_fall_rate=no_fall, forward_ratio=fr, turn_ratio=tr, stand_ok=stand_ok,
                passed=bool(no_fall >= 0.9 and 0.5 <= fr <= 1.5 and 0.4 <= tr <= 1.6 and stand_ok))
    summary = {k: dict(fall_rate=float(np.mean([e["fell"] for e in v])),
                       dist_m=float(np.mean([e["dist_m"] for e in v])),
                       mean_vx=_nanmean([e["mean_vx"] for e in v]), mean_wz=_nanmean([e["mean_wz"] for e in v]),
                       slip_mps=_nanmean([e["slip_mps"] for e in v]), cot=_nanmean([e["cot"] for e in v]),
                       cmd=v[0]["cmd"]) for k, v in res.items()}
    return dict(body=body, tracker_kind=kind, tracker_source=tracker.source, tracker_version=tracker.version,
                tracker_sha=tsha, family=meta["family"], synthetic=meta.get("synthetic", False),
                seeds=seeds, gate=gate, summary=summary, episodes=res, wall_s=time.time() - t0,
                protocol="rrp.control.tracker_validation/v1", mujoco=mujoco.__version__)


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
    ap.add_argument("--freeze", action="store_true", help="write eligibility.json next to the frozen tracker")
    a = ap.parse_args(argv)
    r = validate(a.body, a.kind, a.actor, a.seeds)
    print(json.dumps(dict(body=r["body"], kind=r["tracker_kind"], gate=r["gate"], summary=r["summary"]), indent=1))
    out = Path(a.out) if a.out else TRACKER_DIR / a.body / f"validation_{a.kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    if a.freeze:
        el = dict(body=a.body, tracker_kind=a.kind, tracker_sha=r["tracker_sha"], tracker_version=r["tracker_version"],
                  eligible=r["gate"]["passed"], gate=r["gate"], frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  validation_file=str(out.name), note="frozen on independent tracker validation, before any "
                  "learned high-level policy result")
        (out.parent / f"eligibility_{a.kind}.json").write_text(json.dumps(el, indent=1))


if __name__ == "__main__":
    main()
