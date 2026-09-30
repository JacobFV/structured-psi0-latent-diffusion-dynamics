"""Humanoid tracker / expert evaluations in C MuJoCo (W13, D-138; full self-collision, contact_v2): the recipe stage
`legged/eval_tracker` runs them through `rrp suite {humanoid-steps, humanoid-gap, humanoid-gap-smoke, contact-waypoint}`
(they were scripts/humanoid_{steps,gap}_eval.py, humanoid_gap_smoke.py and contact_waypoint_eval.py).

* steps: an h_steps expert (privileged_teacher:rl_expert, height scan) or a blind tracker driven by the scripted heading command.
* gap: an h_gap_sidestep expert or blind tracker driven by the scripted GapTeacher. Privileged evaluator: success = 0.5 L past
  the wall with |psi_f - yaw| < 0.3 held 0.5 s; failures: fell | wall_collision (> 0.2 s of robot-wall contact) | timeout (30 s).
* gap-smoke (peer GPU): WarpGapEnv at level 0 and 1 with a tracker actor: wall contacts, successes, finite observations.
* waypoint: WaypointTeacher (scripted_teacher, privileged) driving a given tracker on waypoint_contact.
* lab_gate: the W13 lab gate over a tracker_validation JSON (no-fall 1.0, forward >= 0.8, turn >= 0.5, slip < 0.15).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

SRC_EXPERT = "privileged_teacher:rl_expert + scripted_teacher command"
SRC_BLIND = "learned_tracker (blind) + scripted_teacher command"


def _patch_tracker(actor: str, attach) -> dict:
    """Make LeggedSession load THIS actor; `attach(tracker, binding, cur)` sets extra_fn for experts (extra_obs_dim > 0)."""
    import rrp.envs.mujoco.legged as _L
    from rrp.envs.mujoco.legged_tracker import LearnedTracker
    cur: dict = {}

    def _load(key, binding, meta, kind="auto"):
        tr = LearnedTracker(actor, binding, key)
        cur["expert"] = bool(int(tr.meta.get("extra_obs_dim") or 0))
        if cur["expert"]:
            attach(tr, binding, cur)
        return tr
    _L.load_tracker = _load
    return cur


def _render_setup(s):
    import mujoco
    s.model.vis.global_.offwidth, s.model.vis.global_.offheight = 560, 420
    ren = mujoco.Renderer(s.model, 420, 560)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = s.binding.root_bid
    cam.distance = max(1.5, 3.5 * float(s.scenario.meta["L"]))
    cam.azimuth, cam.elevation = 90.0, -10.0
    return ren, cam


def _caption(frame, lines):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def steps_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite humanoid-steps", description="C-MuJoCo h_steps evaluation of an expert or blind tracker.")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--seed0", type=int, default=7200)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h-frac", type=float, default=None, help="step height / leg length (default: the scene's own draw)")
    ap.add_argument("--render", type=int, default=0, help="render the first N episodes to artifacts/video (MUJOCO_GL=egl)")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps, steps_scan_np
    from rrp.envs.mujoco.legged import LeggedSession
    from rrp.policies.teachers.humanoid import StepsHeadingTeacher

    def attach(tr, binding, cur):        # the PRIVILEGED scan is attached before the session's first (settling) tick
        tr.extra_fn = lambda d: steps_scan_np(d.qpos[binding.qa:binding.qa + 7], cur["L"], cur["h"])
    CUR = _patch_tracker(a.actor, attach)
    rows = []
    for seed in range(a.seed0, a.seed0 + a.n):
        sc = build_h_steps(a.body, seed, h_frac=a.h_frac)
        CUR.update(L=float(sc.meta["L"]), h=float(sc.meta["staircase"]["h"]))
        s = LeggedSession(sc, tracker_kind="learned", seed=seed)      # plain trackers (no scan input) run blind
        s.reset(seed)
        te = StepsHeadingTeacher(s)
        t0, status, frames = time.time(), "timeout", []
        rend = _render_setup(s) if len(rows) < a.render else None
        src = s.tracker.version + (" +privileged scan" if s.tracker.extra_fn else " (blind)")
        for _ in range(400):                                           # 10 Hz commands, 40 s
            s.step(te.act())
            if rend is not None:
                rend[0].update_scene(s.data, camera=rend[1])
                frames.append(_caption(rend[0].render().copy(), [
                    (f"PRIVILEGED TEACHER (rl_expert) + scripted heading command | {a.label}" if s.tracker.extra_fn else
                     f"learned tracker (blind) + scripted heading command | {a.label}"), f"{src}",
                    f"{a.body} h_steps  step h={sc.meta['h_frac']:.2f} L  seed {seed}  t={s.data.time:4.1f}s  {'FELL' if s.fell else ''}"]))
            if s.fell:
                status = "fell"
                break
            if s.runtime.succeeded():
                status = "success"
                break
        if frames:
            import datetime as _dt
            import imageio
            vd = Path("artifacts/video")
            name = f"{_dt.date.today()}_humanoid_steps_{a.body}_h{sc.meta['h_frac']:.2f}_s{seed}_{a.label or 'eval'}_{status}.mp4"
            frames += [frames[-1]] * 10
            imageio.mimsave(str(vd / name), frames, fps=10, quality=5)
            with open(vd / "INDEX.md", "a") as f:
                f.write(f"- `{name}` — W13 h_steps: {src}; body {a.body}, step height {sc.meta['h_frac']:.2f} L, seed {seed}, outcome "
                        f"{status} (privileged evaluator), 1x speed. source label: "
                        f"{'privileged_teacher:rl_expert (height scan) + scripted_teacher heading command' if s.tracker.extra_fn else 'learned_tracker (blind) + scripted_teacher heading command'}.\n")
        rows.append(dict(seed=seed, status=status, h_frac=sc.meta["h_frac"], x=float(s.data.qpos[s.binding.qa]), x_end=sc.meta["x_end"],
                         sim_s=float(s.data.time), wall_s=time.time() - t0))
        print(rows[-1], flush=True)
    summ = dict(body=a.body, actor=a.actor, n=a.n, source=SRC_EXPERT if CUR.get("expert") else SRC_BLIND,
                success=sum(r["status"] == "success" for r in rows), fell=sum(r["status"] == "fell" for r in rows), rows=rows)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    return 0


def gap_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite humanoid-gap", description="C-MuJoCo h_gap_sidestep evaluation of an expert or blind tracker.")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--seed0", type=int, default=7200)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=1.0)
    a = ap.parse_args(argv)
    import mujoco
    from rrp.envs.mujoco.humanoid_scenes import build_h_gap, gap_obs_np
    from rrp.envs.mujoco.legged import LeggedSession
    from rrp.policies.teachers.humanoid import GapTeacher

    def attach(tr, binding, cur):
        tr.extra_fn = lambda d: gap_obs_np(d.qpos[binding.qa:binding.qa + 7], cur["meta"],
                                           cur["teacher"].phase2 if cur.get("teacher") else 0.0)
    CUR = _patch_tracker(a.actor, attach)
    rows = []
    for seed in range(a.seed0, a.seed0 + a.n):
        sc = build_h_gap(a.body, seed, level=a.level)
        CUR.update(meta=sc.meta, teacher=None)
        s = LeggedSession(sc, tracker_kind="learned", seed=seed)
        s.reset(seed)
        te = GapTeacher(s)
        CUR["teacher"] = te
        m = s.model
        walls = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, w) for w in sc.meta["walls"]}
        wall_t, hold_t, status, t0 = 0.0, 0.0, "timeout", time.time()
        for _ in range(300):
            s.step(te.act())
            d = s.data
            hit = any((d.contact[i].geom1 in walls) ^ (d.contact[i].geom2 in walls) for i in range(d.ncon))
            wall_t = wall_t + 0.1 if hit else 0.0
            _, _, yaw = s.base_pose_truth()
            ok = te.phase2 and abs(math.atan2(math.sin(sc.meta["psi_f"] - yaw), math.cos(sc.meta["psi_f"] - yaw))) < 0.3
            hold_t = hold_t + 0.1 if ok else 0.0
            if s.fell:
                status = "fell"
                break
            if wall_t > 0.2:
                status = "wall_collision"
                break
            if hold_t >= 0.5:
                status = "success"
                break
        rows.append(dict(seed=seed, status=status, gap_ratio=sc.meta["gap_ratio"], y_c=sc.meta["y_c"], psi_f=sc.meta["psi_f"],
                         x=float(s.data.qpos[s.binding.qa]), sim_s=float(s.data.time), wall_s=time.time() - t0))
        print(rows[-1], flush=True)
    summ = dict(body=a.body, actor=a.actor, n=a.n, level=a.level, source=SRC_EXPERT if CUR.get("expert") else SRC_BLIND,
                success=sum(r["status"] == "success" for r in rows), status=dict(Counter(r["status"] for r in rows)), rows=rows)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    return 0


def gap_smoke_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite humanoid-gap-smoke", description="WarpGapEnv smoke with a tracker actor (peer GPU).")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    from rrp.envs.mujoco.tracker_nets import mlp
    from rrp.envs.warp.task_env import WarpGapEnv
    st = torch.load(a.actor, map_location="cuda", weights_only=False)
    m = st["meta"]
    net = mlp(m["obs_dim"], tuple(m["hidden"]), m["act_dim"]).cuda()
    net.load_state_dict(st["actor"])
    mean, std = st["obs_mean"].cuda(), (st["obs_var"].cuda() + 1e-8).sqrt()
    out = {}
    for lv in (0.0, 1.0):
        e = WarpGapEnv(a.body, 256, seed=5, level=lv, clock_gate=bool(m.get("clock_gate")), push=False)
        for _ in range(1000):
            o = e.observe()[:, :m["obs_dim"]]
            e.step(net(((o - mean) / std).clamp(-5, 5)))
        s = e.pop_stats()
        out[f"level{lv}"] = dict(episodes=s["episodes"], falls=s["falls"], successes=s["successes"], obs_dim=e.obs_dim,
                                 gap_w_over_bw=float((e.gap_w / e.bw).mean()), finite=bool(torch.isfinite(e.observe()).all()))
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def waypoint_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite contact-waypoint",
                                 description="WaypointTeacher (scripted_teacher, privileged) driving a given tracker on waypoint_contact "
                                             "(env RRP_CONTACT_MODEL=v2 [RRP_ACTUATOR_LIMITS=...]).")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import rrp.envs.mujoco.legged as _L
    from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
    from rrp.envs.mujoco.legged_tracker import LearnedTracker
    from rrp.policies.teachers.legged import WaypointTeacher
    _L.load_tracker = lambda key, binding, meta, kind="auto": LearnedTracker(a.actor, binding, key)      # evaluate THIS actor
    rows = []
    for seed in range(a.seed0, a.seed0 + a.n):
        sc = build_waypoint_contact(a.body, seed)
        s = LeggedSession(sc, tracker_kind="learned", seed=seed)
        te = WaypointTeacher(s)
        steps, cmds = 0, []
        for _ in range(1300):
            cmd = te.act()
            cmds.append(cmd.groups["base_velocity"])
            s.step(cmd)
            steps += 1
            if te.done or s.fell:
                break
        c = np.array(cmds)
        status = "success" if s.privileged_success() and not s.fell else ("fell" if s.fell else "failure")
        rows.append(dict(seed=seed, status=status, steps=steps, sim_s=steps * s.dt,
                         pure_turn_frac=float(np.mean((c[:, 0] < 0.05) & (np.abs(c[:, 2]) > 0.05))),
                         contact_model=sc.meta.get("contact_model"), actuator_limits=s.binding.meta.get("actuator_limits")))
        print(rows[-1], flush=True)
    summ = dict(body=a.body, actor=a.actor, n=a.n, success=sum(r["status"] == "success" for r in rows),
                fell=sum(r["status"] == "fell" for r in rows),
                mean_sim_s_success=float(np.mean([r["sim_s"] for r in rows if r["status"] == "success"] or [0])), rows=rows)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    return 0


def lab_gate(validation: dict, gate_report: dict | None = None, waypoint: dict | None = None) -> dict:
    """The W13 lab gate over a tracker_validation result: no-fall 1.0, forward >= 0.8, turn >= 0.5, slip < 0.15 (plus the D-112
    verdict and the waypoint_contact success when given). Verdicts are data; the caller decides what fails."""
    g = validation["gate"]
    lab = dict(no_fall=g.get("no_fall_rate"), forward=g.get("forward_ratio"), turn=g.get("turn_ratio"),
               slip=(g.get("contact_gate") or {}).get("slip_ratio"))
    lab["passed"] = bool(lab["no_fall"] == 1.0 and (lab["forward"] or 0) >= 0.8 and (lab["turn"] or 0) >= 0.5
                         and (lab["slip"] if lab["slip"] is not None else 1) < 0.15)
    if gate_report is not None:
        lab["d112_verdict"] = gate_report.get("verdict")
        lab["d112_criteria"] = {c["name"]: [c.get("value"), c.get("status")] for c in gate_report.get("criteria", [])}
    if waypoint is not None:
        lab.update(waypoint_success=waypoint["success"], waypoint_fell=waypoint["fell"], waypoint_n=waypoint["n"])
    return lab
