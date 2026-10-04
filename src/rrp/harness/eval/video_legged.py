"""`rrp video legged`: render short labelled clips of the SCRIPTED TEACHER (privileged) driving the frozen body tracker on legged /
humanoid bodies (waypoint_contact task), or (`--task h_steps|h_gap --actor <spec|actor.pt>`) a terrain tracker under the task's
scripted command layer (`rl_expert`, the same policy, env and judge as `rrp suite humanoid-steps / humanoid-gap`). Same scene builder, teacher and tracker as the teacher route of
rrp.harness.eval.legged_latent_eval (run_episode with ctl=None). No learned high-level policy is involved.

usage (GPU lease for EGL on the peer):
  MUJOCO_GL=egl rrp video legged --bodies go2,t1 --seeds 10000,10001 --out artifacts/video \
      [--arc-only g1] [--rows artifacts/runs/bodies_teacher_ref/render_rows.jsonl]
Each clip is <= --max-clip-s seconds (long episodes are played back faster; the speed-up is in the caption), and
its caption carries source label, tracker, body, task, seed and privileged-evaluator outcome. A line is appended to
<out>/INDEX.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rrp.core.runs import parse_seed_spec  # noqa: E402
from rrp.harness.eval.legged_latent_eval import run_episode  # noqa: E402

BAR = (0, 0, 0)


def caption(img, lines, color=(255, 255, 255), outcome_color=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=BAR)
    for i, t in enumerate(lines):
        c = outcome_color if (outcome_color and i == len(lines) - 1) else color
        d.text((6, 3 + 14 * i), t, fill=c)
    return np.asarray(im)


def tracker_label(ver: str | None) -> str:
    v = ver or ""
    if "learned_tracker" in v:
        i = v.index("learned_tracker")
        return "frozen learned tracker " + ":".join(v[i:].split(":")[1:3])
    if "cpg" in v.lower():
        return "frozen CPG gait tracker (scripted)"
    return f"frozen tracker {v.split(':')[2] if v.count(':') >= 2 else v}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--arc-only", default="none", help="none | all | comma list of bodies (declared teacher variant)")
    ap.add_argument("--out", default="artifacts/video")
    ap.add_argument("--rows", default=None, help="append the episode rows (JSONL) here")
    ap.add_argument("--max-s", type=float, default=60.0, help="episode sim-time limit (same as the eval default)")
    ap.add_argument("--max-clip-s", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quality", type=float, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--cam-scale", default="*=0.8,t1=0.6,g1=0.6,h1=0.55",
                    help="camera distance multipliers, body=x comma list ('*' = default)")
    ap.add_argument("--frame-every", type=int, default=1, help="record one frame every N native steps (memory)")
    ap.add_argument("--task", default="waypoint_contact", help="waypoint_contact (scripted teacher) | h_steps | h_gap (rl_expert)")
    ap.add_argument("--actor", default=None, help="h_steps / h_gap: tracker spec <body>:<version> or an actor.pt file")
    ap.add_argument("--teacher", action="store_true", help="h_* task: the task's SCRIPTED TEACHER (privileged) over --actor as the "
                    "tracker (what the sealed-body checks run); default: the tracker alone (rl_expert)")
    ap.add_argument("--label", default="", help="extra caption line (e.g. the gate verdict of this checkpoint)")
    ap.add_argument("--scene", default=None, help="h_steps / h_gap: scene JSON, e.g. '{\"h_frac\": 0.2}' or '{\"level\": 1.0}'")
    a = ap.parse_args(argv)
    a.cam_scale = {k: float(v) for k, v in (kv.split("=") for kv in a.cam_scale.split(","))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.task != "waypoint_contact":
        return task_clips(a, out)
    for body in a.bodies.split(","):
        arc = a.arc_only == "all" or body in a.arc_only.split(",")
        for sd in parse_seed_spec(a.seeds):
            row, frames = run_episode(None, body, sd, a.max_s, video=True, arc_only=arc, frame_every=a.frame_every,
                                     cam_scale=a.cam_scale.get(body, a.cam_scale.get('*', 1.0)), size=(a.height, a.width))
            # frames are recorded every frame_every native steps; subsample so the clip fits max_clip_s at fps
            sim_t = row["sim_time"]
            n_max = int(a.max_clip_s * a.fps) - a.fps          # leave 1 s to hold the final frame
            stride = max(1, int(np.ceil(len(frames) / n_max)))
            fr = frames[::stride]
            if frames and fr[-1] is not frames[-1]:
                fr.append(frames[-1])
            speed = (sim_t / max(1e-6, len(fr) / a.fps))
            tag = "success" if row["success"] else ("fell" if row["fell"] else "failure")
            oc = {"success": (120, 255, 120), "fell": (255, 110, 110), "failure": (255, 200, 80)}[tag]
            src = "SCRIPTED TEACHER (privileged) + " + tracker_label(row.get("tracker"))
            l2 = f"{body} | waypoint_contact | seed {sd}" + (" | teacher variant arc_only" if arc else "") + \
                " | NOT a learned policy"
            l_out = f"outcome: {tag.upper()} (privileged evaluator), episode {sim_t:.1f}s sim"
            imgs = []
            for f, st in fr:
                imgs.append(caption(f, [src, l2, f"{st} | playback {speed:.1f}x", l_out], outcome_color=oc))
            imgs += [imgs[-1]] * a.fps
            name = (f"{dt.date.today()}_scripted_teacher{('_' + a.tag) if a.tag else ''}_legged_{body}"
                    f"{'_arconly' if arc else ''}_waypoint_contact_s{sd}_{tag}.mp4")
            imageio.mimsave(out / name, imgs, fps=a.fps, quality=a.quality, macro_block_size=8)
            mb = (out / name).stat().st_size / 1e6
            with open(out / "INDEX.md", "a") as f:
                f.write(f"- `{name}` — source=scripted_teacher{':arc_only' if arc else ''} (privileged) + "
                        f"{tracker_label(row.get('tracker'))} robot={body} task=waypoint_contact seed={sd} "
                        f"outcome={tag} (privileged evaluator); playback {speed:.1f}x; no learned policy\n")
            row["video"] = name
            if a.rows:
                Path(a.rows).parent.mkdir(parents=True, exist_ok=True)
                with open(a.rows, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(json.dumps(dict(body=body, seed=sd, outcome=tag, sim_time=round(sim_t, 1), frames=len(imgs),
                                  speed=round(speed, 2), mb=round(mb, 2), video=name)), flush=True)



def task_clips(a, out: Path) -> int:
    """h_steps / h_gap clips of a terrain tracker (`rl_expert`) under the task's scripted command layer; outcome = the task judge
    (privileged evaluator). One clip per (body, seed)."""
    import mujoco
    from rrp.harness.eval.evaluate import evaluate, task_hooks
    from rrp.harness.eval.humanoid_eval import _expert_source, _status, resolve_actor
    from rrp.harness.hooks import FrameCallback
    from rrp.policies.teachers.humanoid import make_rl_expert
    if not a.actor:
        raise SystemExit("--task h_steps / h_gap needs --actor")
    from rrp.core.sealed import SealedSplit
    scene = json.loads(a.scene) if a.scene else None
    SealedSplit.load().assert_train_allowed(a.bodies.split(","), list(parse_seed_spec(a.seeds)), what="video")   # sealed: adaptation seeds only
    for body in a.bodies.split(","):
        spec = resolve_actor(body, a.actor)
        if a.teacher:
            from rrp.policies.base import make_policy
            pol = make_policy(f"teacher:{a.task}")
        else:
            pol = make_rl_expert(arg=spec)
        for sd in parse_seed_spec(a.seeds):
            frames, rend = [], {}

            def cb(i, env, k, phase):
                if k % a.frame_every:
                    return
                if "r" not in rend:
                    rend["r"] = mujoco.Renderer(env.model, a.height, a.width)
                    cam = rend["cam"] = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    cam.trackbodyid = env.binding.root_bid
                    cs = a.cam_scale.get(body, a.cam_scale.get("*", 1.0))
                    cam.distance, cam.elevation, cam.azimuth = cs * 3.0 * max(0.5, env.binding.nominal_height() / 0.35) ** 0.5, -20, 120
                rend["r"].update_scene(env.data, camera=rend["cam"])
                frames.append((rend["r"].render().copy(), f"t={env.data.time:.1f}s"))

            ep = evaluate(pol, "mujoco/legged", a.task, body, [sd], scene=scene, batch=1, max_seconds=a.max_s,
                          hooks=[*task_hooks(a.task, "mujoco/legged"), FrameCallback(cb)], env_kw=dict(tracker=spec))[0]
            st = _status(ep)
            tag = "success" if st == "success" else ("fell" if st == "fell" else "failure")
            oc = {"success": (120, 255, 120), "fell": (255, 110, 110), "failure": (255, 200, 80)}[tag]
            n_max = int(a.max_clip_s * a.fps) - a.fps
            stride = max(1, int(np.ceil(len(frames) / max(1, n_max))))
            fr = frames[::stride] + ([frames[-1]] if frames and len(frames) % stride != 1 and stride > 1 else [])
            speed = ep.time / max(1e-6, len(fr) / a.fps)
            src0 = "SCRIPTED TEACHER (privileged)" if a.teacher else _expert_source(pol)
            src = f"{src0} | tracker {spec}"
            l2 = f"{body} | {a.task} {json.dumps(scene) if scene else ''} | seed {sd}"
            l_out = f"outcome: {st.upper()} (task judge, privileged evaluator), {ep.time:.1f}s sim"
            lines = [src, l2] + ([a.label] if a.label else []) + [f"{t} | playback {speed:.1f}x", l_out]
            imgs = [caption(f, lines, outcome_color=oc) for f, t in fr]
            imgs += [imgs[-1]] * a.fps
            label = "scripted_teacher_over_learned_tracker" if a.teacher else ("privileged_rl_expert" if src0.startswith("privileged") else
                     "learned_tracker_blind" if "(blind)" in src0 else "learned_rl_expert")
            name = f"{dt.date.today()}_{label}{('_' + a.tag) if a.tag else ''}_{body}_{a.task}_s{sd}_{tag}.mp4"
            imageio.mimsave(out / name, imgs, fps=a.fps, quality=a.quality, macro_block_size=8)
            with open(out / "INDEX.md", "a") as f:
                f.write(f"- `{name}` — source={src0} tracker={spec} label={a.label!r} actor={a.actor} robot={body} task={a.task} "
                        f"scene={json.dumps(scene) if scene else 'default'} seed={sd} outcome={st} (task judge); playback {speed:.1f}x\n")
            print(json.dumps(dict(body=body, seed=sd, outcome=st, sim_time=round(ep.time, 1), frames=len(imgs), video=name)), flush=True)
    return 0
