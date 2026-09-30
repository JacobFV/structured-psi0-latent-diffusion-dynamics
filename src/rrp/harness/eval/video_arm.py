"""`rrp video arm`: render short labelled demo videos of teacher or learned-policy episodes (GPU EGL on the peer/host
GPU lease). Usage:
  rrp video arm --robot panda_pg2 --seeds 3000001,3000002 --source learned --checkpoint ckpt.pt --out artifacts/video
  rrp video arm --robot panda_pg2 --seeds 3000001 --source scripted_teacher --out artifacts/video
  rrp video arm --robot panda_pg2 --seeds 3000001 --source learned_latent --checkpoint flow/policy.pt
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

from rrp.harness import hooks as H
from rrp.harness.eval.captions import caption
from rrp.harness.rollout import rollout


class _LatentVideo:
    """The `learned_latent` video controller: system i packet every `replan` ticks -> system 0 every tick (a rejected
    packet leaves system 0's fallback hold), after an optional labelled scripted-teacher prefix of `prefix` ticks (the
    replan phase counts the prefix ticks, as the video always did)."""

    def __init__(self, sys_i, realizer, dev, replan, prefix):
        from rrp.policies.latent import LatentStackPolicy
        self.sys_i, self.realizer, self.dev, self.replan, self.prefix = sys_i, realizer, dev, replan, prefix
        self.info = LatentStackPolicy(sys_i, realizer, replan_ticks=replan, device=dev, name="video:latent",
                                      privileged=prefix > 0).info

    def reset(self, spec, task, seeds, *, envs=None):
        from rrp.policies.system0 import LatentSystem0
        self.env = envs[0]
        self.s0 = LatentSystem0(self.realizer, self.sys_i.featurizer(self.env), latent_space_version=self.sys_i.lsv,
                                realizer_compat_version=self.sys_i.rcv, device=self.dev)
        self.k, self.pteacher = 0, None

    def act(self, obs):
        from rrp.policies.base import Act
        from rrp.policies.teachers.arm import PickPlaceTeacher
        s, k = self.env, self.k
        self.k += 1
        if k < self.prefix:                                   # labelled curriculum prefix
            if k == 0:
                self.pteacher = PickPlaceTeacher(s)
            return {0: Act(self.pteacher.act())}
        if k % self.replan == 0 or self.s0.packet is None:
            try:
                self.s0.receive(self.sys_i.packets([s])[0], now=float(s.data.time),
                                graph_version=s.runtime.graph_version)
            except Exception as e:          # rejected/stale packet: system 0 keeps its fallback hold
                print("packet rejected:", e, flush=True)
        return {0: Act(self.s0.tick(s, s.controller_version()))}


def run(args):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.policies.teachers.arm import PickPlaceTeacher
    robot = workbench_robots()[args.robot]()
    pol = None
    if args.source == "learned":
        import torch
        from rrp.policies.bc import LearnedPolicy
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            from rrp.ops.workload import apply_cap
            apply_cap()
        pol = LearnedPolicy.from_checkpoint(args.checkpoint, device=dev, execute_prefix=args.prefix)
    elif args.source == "learned_latent":             # corrected path: system i packet -> system 0 every tick
        import torch
        from rrp.policies.latent import LatentPolicy
        from rrp.policies.nets.checkpoint import load_checkpoint
        from rrp.policies.bundles import load_representation
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            from rrp.ops.workload import apply_cap
            apply_cap()
        pol = LatentPolicy.from_checkpoint(args.checkpoint, device=dev, nfe=8)
        _, _, realizer, _, _ = load_representation(Path(load_checkpoint(args.checkpoint)["config"]["representation"]), dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "INDEX.md"
    for sd in [int(x) for x in args.seeds.split(",")]:
        if args.task == "pick_place_paired":         # binding pairs: identical scene, patient = which cube is bound
            s = Session(BUILDERS[args.task](robot, sd, patient=args.patient), seed=sd)
        else:
            s = Session(BUILDERS[args.task](robot, sd, n_distractors=sd % 3), seed=sd)
        r = mujoco.Renderer(s.model, args.height, args.width)
        teacher = PickPlaceTeacher(s) if args.source == "scripted_teacher" else None
        frames = []
        ptag = f" | patient=cube#{args.patient}" if args.task == "pick_place_paired" else ""
        label = "SCRIPTED TEACHER (privileged)" if teacher else f"LEARNED {Path(args.checkpoint).parent.name}"
        if args.source == "learned_latent":
            label = f"LEARNED latent (sys-i flow + sys-0) {Path(args.checkpoint).parent.name}"
        if teacher:
            from rrp.policies.teachers import TeacherPolicy
            vpol = TeacherPolicy(args.task, lambda e: teacher, "video:pick_place", ("joint_position", "gripper"))
            end = lambda i, e: bool(teacher.done)
        elif args.source == "learned_latent":
            vpol = _LatentVideo(pol, realizer, dev, args.replan, args.teacher_prefix_steps)
            end = lambda i, e: bool(e.runtime.succeeded())
        else:
            from rrp.policies.bc import BCPolicy
            vpol = BCPolicy(pol, name="video:learned")
            end = lambda i, e: bool(e.runtime.succeeded())
        tick = [0]

        def frame(i, e, act, step):
            k = tick[0]
            tick[0] += 1
            if k % args.every == 0:
                r.update_scene(e.data, camera=args.camera)
                st = " ".join(f"{n}:{v.status}" for n, v in e.runtime.instances.items())
                lab = label if not (args.source == "learned_latent" and k < args.teacher_prefix_steps) else \
                    f"SCRIPTED TEACHER prefix (privileged) {k + 1}/{args.teacher_prefix_steps}, then {label}"
                frames.append(caption(r.render().copy(), [f"{lab} | {args.robot} | {args.task}{ptag} | seed {sd}",
                                                          f"t={e.data.time:.1f}s  {st}"]))
        ep = rollout(lambda _sd: s, vpol, H.budget_task(args.task, s.spec.env_id), [sd], batch=1,
                     max_steps=args.max_steps, hooks=[H.EndWhen(end), H.Recorder(on_step=frame)])[0]
        if ep.outcome == "crash":
            raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
        ok = s.privileged_success()
        tag = "success" if ok else "failure"
        pf = f"_teacherprefix{args.teacher_prefix_steps}" if args.teacher_prefix_steps else ""
        pp = f"_patient{args.patient}" if args.task == "pick_place_paired" else ""
        name = f"{dt.date.today()}_{args.source}{('_' + args.tag) if args.tag else ''}{pf}_{args.robot}_{args.task}{pp}_s{sd}_{tag}.mp4"
        imageio.mimsave(out / name, frames, fps=args.fps, quality=6)
        with open(index, "a") as f:
            f.write(f"- `{name}` — source={args.source} ckpt={args.checkpoint or '-'} robot={args.robot} "
                    f"task={args.task}{pp} seed={sd} outcome={tag} (privileged evaluator)"
                    + (f" teacher_prefix={args.teacher_prefix_steps} ticks (scripted, privileged) then learned" if pf else "")
                    + "\n")
        print(name, tag, flush=True)


def main(argv=None):
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
    ap.add_argument("--patient", type=int, default=0, help="pick_place_paired: index of the cube bound as patient")
    ap.add_argument("--tag", default="", help="extra filename tag (e.g. grpo / reference)")
    run(ap.parse_args(argv))
