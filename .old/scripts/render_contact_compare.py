"""Side-by-side contact_v1 vs contact_v2 learned-tracker video for one body and one command (contact track).

Left: the v1 tracker in contact_v1 physics. Right: the v2 tracker in contact_v2 physics. Same body, command,
seed and duration. The camera tracks the root from the side. Each panel's caption shows the controller source
(learned_tracker:<body>:iter<N>), the contact model and the running forward stance-slip ratio (contact-point
slip / body speed). The final frame repeats the episode metrics. Writes artifacts/video/<date>_contact_<body>_<tag>.mp4
and appends a line to artifacts/video/INDEX.md.

usage: MUJOCO_GL=egl python scripts/render_contact_compare.py --body t1 [--left v1 --right v2] [--cmd forward] [--T 8]
       (left/right: v1 | v2 | an actor path, followed by :v1 or :v2 for its physics, e.g. runs/x/actor_alpha0.pt:v2)
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rrp.envs.mujoco.legged_core import LeggedBinding  # noqa: E402
from rrp.envs.mujoco.legged_tracker import LearnedTracker, tracker_path  # noqa: E402
from rrp.harness.eval.tracker_validation import scripts  # noqa: E402
from rrp.bodies.legged import legged_body, standalone_model  # noqa: E402

W, H = 480, 360


def caption(frame, lines, color=(255, 255, 255)):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=color)
    return np.asarray(im)


def add_grid(scn, x0, y0, step=0.25, n=24):
    """Visual-only floor stripes (every 25 cm) so foot sliding is visible; not part of the physics model."""
    xs = np.floor(x0 / step) * step + step * np.arange(-n // 2, n // 2)
    for x in xs:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.006, 3.0, 0.0008]),
                            np.array([x, y0, 0.0008]), np.eye(3).ravel(), np.array([0.15, 0.15, 0.18, 1.0], np.float32))
        scn.ngeom += 1


def parse(spec, body):
    if spec in ("v1", "v2"):
        return tracker_path(body, spec), spec
    path, phys = spec.rsplit(":", 1)
    return Path(path), phys


def episode(body, actor, phys, cmd, T, seed, label_extra="", actuator="v1", latency_ms=0.0, limits=None):
    mod = legged_body(body, limits=limits)
    model, _, meta = standalone_model(mod, contact=phys)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    b = LeggedBinding(model, meta)
    tr = LearnedTracker(actor, b, body)
    act = None
    if actuator != "v1":
        from rrp.bodies.actuator import ActuatorModel
        act = ActuatorModel(model, b, 1, None, name=meta["name"], randomize=False, latency_ms=latency_ms, mode=actuator)
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    b.set_default(d, yaw=0.0, noise=0.03, rng=rng)
    mujoco.mj_forward(model, d)
    tr.reset(0.0)
    if act is not None:
        act.reset(0, d.ctrl[b.pol_act].copy())
    ren = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = b.root_bid
    cam.distance = max(1.2, 3.2 * b.nominal_height())
    cam.azimuth, cam.elevation = 90.0, -12.0
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    sub = max(1, int(round(0.02 / model.opt.timestep)))
    mass = float(model.body_subtreemass[b.root_bid])
    frames, slips, speeds, fell = [], [], [], False
    src = tr.version + (" (GPU-trained)" if tr.meta.get("sim_engine") else "")     # W13: shared/transfer trackers name themselves
    for k in range(int(T / 0.02)):
        tgt = tr.act(d, cmd)
        if act is not None:
            act.command(0, tgt)
        else:
            d.ctrl[b.pol_act] = tgt
        for _ in range(sub):
            if act is not None:
                d.ctrl[b.pol_act] = act.substep_ctrl(0, d)
            mujoco.mj_step(model, d)
        fc, fn, sl, bad = b.stance(d)
        if k > 50:
            ld = fn > 0.02 * mass * 9.81
            slips += list(sl[ld])
            speeds.append(float(np.hypot(*d.qvel[b.da:b.da + 2])))
        if bad or d.qpos[b.qa + 2] < b.min_h or b.tilt(d) > b.tilt_limit:
            fell = True
        if k % 2 == 0:       # 25 fps
            ren.update_scene(d, camera=cam, scene_option=opt)
            add_grid(ren.scene, d.qpos[b.qa], d.qpos[b.qa + 1])
            sr = (np.mean(slips) / max(np.mean(speeds), 0.02)) if slips else float("nan")
            frames.append(caption(ren.render().copy(), [
                f"{src}  {label_extra}", f"trained {tr.contact_model} | PHYSICS {meta['contact_model']} actuator {actuator}"
                f" limits {meta.get('actuator_limits')}"
                f"{'' if actuator == 'v1' else ' %dms' % latency_ms} | "
                f"cmd vx={cmd[0]:.2f} wz={cmd[2]:.2f}",
                f"t={k * 0.02:4.1f}s  stance slip ratio={sr:.2f}  {'FELL' if fell else ''}"]))
        if fell and k > 0 and len(frames) > 25:
            break
    ren.close()
    sr = float(np.mean(slips) / max(np.mean(speeds), 0.02)) if slips else float("nan")
    return frames, dict(src=src, contact=meta["contact_model"], trained=tr.contact_model, slip_ratio=sr, speed=float(np.mean(speeds)) if speeds else 0,
                        fell=fell, alpha=tr.meta.get("alpha"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--left", default="v1")
    ap.add_argument("--right", default="v2")
    ap.add_argument("--left-label", default="")
    ap.add_argument("--right-label", default="")
    ap.add_argument("--cmd", default="forward")
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--tag", default="v1-vs-v2")
    ap.add_argument("--actuator", default="v1", help="v1 | v2 | v1lat (rrp.physics.actuator, nominal params) for both panels")
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--left-limits", default=None, help="actuator limits for the left panel body (legacy_gains_v0 | sourced_v1)")
    ap.add_argument("--right-limits", default=None)
    ap.add_argument("--out", default="artifacts/video")
    a = ap.parse_args()
    mod = legged_body(a.body)
    m0, _, meta0 = standalone_model(mod)
    sc = scripts(LeggedBinding(m0, meta0))[a.cmd]
    cmd = np.array(sc["cmd"], float)
    panels = []
    for spec, lab, lim in ((a.left, a.left_label, a.left_limits), (a.right, a.right_label, a.right_limits)):
        path, phys = parse(spec, a.body)
        panels.append(episode(a.body, path, phys, cmd, a.T, a.seed, lab, a.actuator, a.latency_ms, lim))
    n = max(len(p[0]) for p in panels)
    out_frames = []
    for i in range(n):
        row = [p[0][min(i, len(p[0]) - 1)] for p in panels]
        out_frames.append(np.concatenate(row, axis=1))
    outcome = "_".join(("fell" if p[1]["fell"] else "ok") for p in panels)
    date = dt.date.today().isoformat()
    out = Path(a.out) / f"{date}_contact_{a.body}_{a.cmd}_{a.tag}_{outcome}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, out_frames, fps=25, quality=6, macro_block_size=8)
    m = [p[1] for p in panels]
    desc = " vs ".join(f"{x['src']} (trained {x['trained']}, physics {x['contact']}{', alpha=%s' % x['alpha'] if x['alpha'] is not None else ''}{', ' + lab if lab else ''}): "
                       f"stance slip ratio {x['slip_ratio']:.2f} at {x['speed']:.2f} m/s{', FELL' if x['fell'] else ''}"
                       for x, lab in zip(m, (a.left_label, a.right_label)))
    line = (f"- `{out.name}` — CONTACT TRACK ({a.tag}). {a.body}, validation `{a.cmd}` command {cmd.tolist()}, seed {a.seed}, "
            f"actuator {a.actuator}{'' if a.actuator == 'v1' else ' latency %d ms' % a.latency_ms}, "
            f"{a.T:.0f} s; left | right: {desc}. source=learned_tracker (PPO actor, privileged critic in training only).\n")
    with open(Path(a.out) / "INDEX.md", "a") as f:
        f.write(line)
    print(out, m)


if __name__ == "__main__":
    main()
