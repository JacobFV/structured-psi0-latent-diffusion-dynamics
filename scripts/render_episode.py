"""Render short labelled demo videos of teacher or learned-policy episodes (GPU EGL on the peer/host
GPU lease). Usage:
  render_episode.py --robot panda_pg2 --seeds 3000001,3000002 --source learned --checkpoint ckpt.pt --out artifacts/video
  render_episode.py --robot panda_pg2 --seeds 3000001 --source scripted_teacher --out artifacts/video
  render_episode.py --robot panda_pg2 --seeds 3000001 --source learned_latent --checkpoint flow/policy.pt
Each video's caption states the controller source, robot, task, seed and privileged-evaluator outcome, and a
line is appended to artifacts/video/INDEX.md.
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


def caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def run(args):
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.teachers import PickPlaceTeacher
    robot = workbench_robots()[args.robot]()
    pol = None
    if args.source == "learned":
        import torch
        from rrp.policy.runner import LearnedPolicy
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            from rrp.ops.gpu import apply_cap
            apply_cap()
        pol = LearnedPolicy.from_checkpoint(args.checkpoint, device=dev, execute_prefix=args.prefix)
    elif args.source == "learned_latent":             # corrected path: system i packet -> system 0 every tick
        import torch
        from rrp.policy.latent_runner import LatentPolicy
        from rrp.learning.checkpoint import load_checkpoint
        from rrp.learning.latent_train import load_representation
        from rrp.control.latent_realizer import LatentSystem0
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            from rrp.ops.gpu import apply_cap
            apply_cap()
        pol = LatentPolicy.from_checkpoint(args.checkpoint, device=dev, nfe=8)
        _, _, realizer, _, _ = load_representation(Path(load_checkpoint(args.checkpoint)["config"]["representation"]), dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "INDEX.md"
    for sd in [int(x) for x in args.seeds.split(",")]:
        s = Session(BUILDERS[args.task](robot, sd, n_distractors=sd % 3), seed=sd)
        r = mujoco.Renderer(s.model, args.height, args.width)
        teacher = PickPlaceTeacher(s) if args.source == "scripted_teacher" else None
        frames = []
        label = "SCRIPTED TEACHER (privileged)" if teacher else f"LEARNED {Path(args.checkpoint).parent.name}"
        if args.source == "learned_latent":
            label = f"LEARNED latent (sys-i flow + sys-0) {Path(args.checkpoint).parent.name}"
            s0 = LatentSystem0(realizer, pol.featurizer(s), latent_space_version=pol.lsv,
                               realizer_compat_version=pol.rcv, device=dev)
        for k in range(args.max_steps):
            if teacher:
                s.step(teacher.act())
                done = teacher.done
            elif args.source == "learned_latent" and k < args.teacher_prefix_steps:   # labelled curriculum prefix
                if k == 0:
                    pteacher = PickPlaceTeacher(s)
                s.step(pteacher.act())
                done = s.runtime.succeeded()
            elif args.source == "learned_latent":
                if k % args.replan == 0 or s0.packet is None:
                    try:
                        s0.receive(pol.packets([s])[0], now=float(s.data.time), graph_version=s.runtime.graph_version)
                    except Exception as e:          # rejected/stale packet: system 0 keeps its fallback hold
                        print("packet rejected:", e, flush=True)
                s.step(s0.tick(s, s.controller_version()))
                done = s.runtime.succeeded()
            else:
                if not s.executor.queue:
                    s.submit_chunk(pol.chunks([s])[0], execute_prefix=pol.execute_prefix)
                s.step(None)
                done = s.runtime.succeeded()
            if k % args.every == 0:
                r.update_scene(s.data, camera=args.camera)
                st = " ".join(f"{e}:{v.status}" for e, v in s.runtime.instances.items())
                lab = label if not (args.source == "learned_latent" and k < args.teacher_prefix_steps) else \
                    f"SCRIPTED TEACHER prefix (privileged) {k + 1}/{args.teacher_prefix_steps}, then {label}"
                frames.append(caption(r.render().copy(), [f"{lab} | {args.robot} | {args.task} | seed {sd}",
                                                          f"t={s.data.time:.1f}s  {st}"]))
            if done:
                break
        ok = s.privileged_success()
        tag = "success" if ok else "failure"
        pf = f"_teacherprefix{args.teacher_prefix_steps}" if args.teacher_prefix_steps else ""
        name = f"{dt.date.today()}_{args.source}{('_' + args.tag) if args.tag else ''}{pf}_{args.robot}_{args.task}_s{sd}_{tag}.mp4"
        imageio.mimsave(out / name, frames, fps=args.fps, quality=6)
        with open(index, "a") as f:
            f.write(f"- `{name}` — source={args.source} ckpt={args.checkpoint or '-'} robot={args.robot} "
                    f"task={args.task} seed={sd} outcome={tag} (privileged evaluator)"
                    + (f" teacher_prefix={args.teacher_prefix_steps} ticks (scripted, privileged) then learned" if pf else "")
                    + "\n")
        print(name, tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--task", default="pick_place")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--source", choices=["scripted_teacher", "learned", "learned_latent"], required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--prefix", type=int, default=8)
    ap.add_argument("--replan", type=int, default=8, help="learned_latent: system-i period in control ticks")
    ap.add_argument("--out", default="artifacts/video")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--teacher-prefix-steps", type=int, default=0, help="learned_latent: scripted teacher for the first N ticks")
    ap.add_argument("--tag", default="", help="extra filename tag (e.g. grpo / reference)")
    run(ap.parse_args())
