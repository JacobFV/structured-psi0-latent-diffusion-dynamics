"""Render labelled clips of arm pick-place TEACHER versions (W7 before/after), plus optional side-by-side clips.

  render_teacher_versions.py --robot ur5e_tf3 --seeds 10 --versions v1,v2 --sbs --out artifacts/video

Every clip is a scripted_teacher (privileged) episode, captioned with the teacher version, body, seed, time, phase and
the privileged-evaluator outcome; one INDEX.md line per clip. Metrics come from rrp.evaluation.teacher_quality (the same
code path as the reported tables), so the clip and the table row are the same episode.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

NAMES = {"v1": "v1 waypoint (old default)", "v2": "v2 min-jerk"}


def _caption(frame, lines, color=(255, 255, 255)):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=color)
    return np.asarray(im)


def render(robot_key, seed, version, args):
    from rrp.bodies.catalog import workbench_robots
    from rrp.evaluation.teacher_quality import run_quality_episode
    from rrp.envs.native import Session
    from rrp.envs.scenario import BUILDERS
    robot = workbench_robots()[robot_key]()
    # a throwaway session of the same scene to size the renderer (the episode builds its own identical session)
    s0 = Session(BUILDERS["pick_place"](robot, seed, n_distractors=seed % 3), seed=seed)
    rend = mujoco.Renderer(s0.model, args.height, args.width)
    frames = []

    gc = os.environ.get("RRP_GRASP_CONTACT", "v1")

    def cap(img, s, teacher, k):
        pt = (f"  obj friction x{args.obj_friction:g}" if args.obj_friction != 1.0 else "") + \
            (f"  cube mass x{args.obj_mass:g}" if args.obj_mass != 1.0 else "")
        return _caption(img, [f"SCRIPTED TEACHER {NAMES[version]} (privileged) | {robot_key} | pick_place | seed {seed}",
                              f"t={s.data.time:4.1f}s  phase={teacher.phase}  grasp contact grasp_{gc}{pt}"])

    cam = args.camera
    if cam == "closeup":        # free camera following the task cube (shows finger/cube overlap)
        vc = mujoco.MjvCamera()
        vc.type = mujoco.mjtCamera.mjCAMERA_FREE
        vc.distance, vc.elevation, vc.azimuth = 0.28, -12.0, 150.0
        cb = mujoco.mj_name2id(s0.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

        def cam(m, d):
            vc.lookat[:] = d.xpos[cb]
            return vc
    row = run_quality_episode(robot_key, seed, version, max_steps=args.max_steps, robot=robot,
                              frames=dict(renderer=rend, camera=cam, every=1, out=frames, caption=cap),
                              obj_friction=args.obj_friction, obj_mass=args.obj_mass)
    rend.close()
    return row, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--versions", default="v1,v2")
    ap.add_argument("--sbs", action="store_true", help="also write a side-by-side clip of the first two versions")
    ap.add_argument("--out", default="artifacts/video")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--note", default="")
    ap.add_argument("--obj-friction", type=float, default=1.0, help="x friction of the cube and finger pads")
    ap.add_argument("--obj-mass", type=float, default=1.0, help="x cube mass")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    vers = a.versions.split(",")
    for sd in [int(x) for x in a.seeds.split(",")]:
        clips = {}
        for v in vers:
            row, frames = render(a.robot, sd, v, a)
            if not row.get("feasible"):
                print(a.robot, sd, "infeasible: skipped", flush=True)
                break
            tag = "success" if row["success"] else "failure"
            stats = (f"time {row['time_s']:.1f}s, joint cmd jerk peak {row['joint_cmd_jerk_peak']:.0f} rad/s^3, "
                     f"max joint vel step {row['vel_jump_any_max']:.2f} rad/s/tick, grasp penetration "
                     f"{1000 * row['pen_hand_max_grasp_m']:.0f} mm" + (f", failure {row['failure_stage']}" if tag == "failure" else ""))
            last = _caption(frames[-1].copy(), [f"SCRIPTED TEACHER {NAMES[v]} (privileged) | {a.robot} | seed {sd}",
                                                f"END: {tag.upper()} (privileged evaluator) | {stats[:90]}"])
            frames = frames + [last] * a.fps          # hold the outcome card 1 s
            gct = os.environ.get("RRP_GRASP_CONTACT")
            pt = (f"_objfric{a.obj_friction:g}" if a.obj_friction != 1.0 else "") + (f"_objmass{a.obj_mass:g}" if a.obj_mass != 1.0 else "")
            name = f"{today}_scripted_teacher_{v}{('_grasp' + gct) if gct else ''}{pt}_{a.robot}_pick_place_s{sd}_{tag}.mp4"
            imageio.mimsave(out / name, frames, fps=a.fps, quality=6)
            clips[v] = (frames, tag, row)
            with open(out / "INDEX.md", "a") as f:
                f.write(f"- `{name}` — source=scripted_teacher:{row['source'].split(':', 1)[1]} (privileged; W7 teacher "
                        f"{'before' if v == 'v1' else 'after'}) robot={a.robot} task=pick_place seed={sd} outcome={tag} "
                        f"(privileged evaluator); grasp contact grasp_{os.environ.get('RRP_GRASP_CONTACT', 'v1')}; {stats}"
                        f"{('; ' + a.note) if a.note else ''}\n")
            print(name, tag, stats, flush=True)
        if a.sbs and len(clips) >= 2:
            (fa, ta, _), (fb, tb, _) = clips[vers[0]], clips[vers[1]]
            n = max(len(fa), len(fb))
            fa = fa + [fa[-1]] * (n - len(fa))
            fb = fb + [fb[-1]] * (n - len(fb))
            sbs = [np.concatenate([x, np.full((x.shape[0], 6, 3), 255, np.uint8), y], axis=1) for x, y in zip(fa, fb)]
            name = f"{today}_scripted_teacher_{vers[0]}_vs_{vers[1]}_{a.robot}_pick_place_s{sd}_{ta}_vs_{tb}.mp4"
            imageio.mimsave(out / name, sbs, fps=a.fps, quality=6)
            with open(out / "INDEX.md", "a") as f:
                f.write(f"- `{name}` — side by side, left scripted_teacher {vers[0]} ({ta}), right scripted_teacher "
                        f"{vers[1]} ({tb}); both privileged; robot={a.robot} task=pick_place seed={sd}; same scene\n")
            print(name, flush=True)


if __name__ == "__main__":
    main()
