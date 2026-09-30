"""`rrp video dual`: render labelled videos of dual-arm episodes (support_insert / handover) on the corrected latent path.

  rrp video dual --task handover --pair panda_pg2__ur5e_pg2 --seeds 3000001 --source learned_latent \
      --checkpoint artifacts/runs/dualarm_flow_sem_v1/policy.pt [--probe probe.pt]
  rrp video dual --task support_insert --pair parm5_pg2__parm5_pg2 --seeds 3000001 --source scripted_teacher

Caption: controller source (LEARNED latent / SCRIPTED TEACHER), task, pair, seed, runtime event statuses, and for
learned runs the per-slot answers of the packet probe on the RECEIVED packet (diagnostic readout only; slot L = left
role, R = right role). Outcome = privileged evaluator. A line is appended to artifacts/video/INDEX.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco

from rrp.harness.eval import hooks as H
from rrp.harness.eval.captions import caption
from rrp.harness.rollout import rollout


class _DualLatentVideo:
    """The dual `learned_latent` video controller: system i packet every `replan` ticks (a rejected packet leaves system
    0's fallback hold) -> system 0 every tick. `readout` = the probe's per-slot diagnostic of the last emitted packet
    (shown in the caption; never fed back)."""

    def __init__(self, sys_i, realizer, probe, ents, dev, replan):
        from rrp.policies.latent import LatentStackPolicy
        self.sys_i, self.realizer, self.probe, self.ents, self.dev, self.replan = sys_i, realizer, probe, ents, dev, replan
        self.info = LatentStackPolicy(sys_i, realizer, replan_ticks=replan, device=dev, name="video:dual_latent").info
        self.readout = []

    def reset(self, spec, task, seeds, *, envs=None):
        from rrp.policies.system0 import DualLatentSystem0
        self.env = envs[0]
        self.s0 = DualLatentSystem0(self.realizer, self.sys_i.featurizer(self.env), latent_space_version=self.sys_i.lsv,
                                    realizer_compat_version=self.sys_i.rcv, device=self.dev)
        self.k = 0

    def act(self, obs):
        import torch
        from rrp.harness.eval.dual_latent_eval import probe_readout
        from rrp.policies.base import Act
        s, k = self.env, self.k
        self.k += 1
        if k % self.replan == 0 or self.s0.packet is None:
            p = self.sys_i.packets([s])[0]
            try:
                self.s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
            except Exception as e:          # noqa: BLE001 - rejected packet: fallback hold
                print("packet rejected:", e, flush=True)
            with torch.no_grad():
                z = torch.from_numpy(p.z)[None].to(self.dev)
                o = self.probe(z, torch.tensor([p.assembly_mask], device=self.dev), len(self.ents))
            self.readout = probe_readout({kk: v.cpu() for kk, v in o.items()}, len(self.ents), ent_names=self.ents)
        return {0: Act(self.s0.tick(s))}


def run(a):
    import torch
    from rrp.policies.teachers.dual_validate import make_session
    from rrp.policies.teachers.dual import TEACHERS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = R = P = None
    if a.source == "learned_latent":
        from rrp.policies.latent import DualLatentPolicy
        from rrp.policies.nets.checkpoint import load_checkpoint
        from rrp.policies.bundles import load_representation
        if dev == "cuda":
            from rrp.ops.workload import apply_cap
            apply_cap()
        pol = DualLatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=8)
        _, _, R, P, _ = load_representation(Path(load_checkpoint(a.checkpoint)["config"]["representation"]), dev)
        if a.probe:
            from rrp.policies.nets.latent_probes import PacketProbe
            st = torch.load(a.probe, map_location=dev, weights_only=False)
            P = PacketProbe(**st["cfg"]).to(dev).eval()
            P.load_state_dict(st["state"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sd in [int(x) for x in a.seeds.split(",")]:
        s = make_session(a.task, a.pair, sd)
        rd = mujoco.Renderer(s.model, a.height, a.width)
        teacher = TEACHERS[a.task](s) if a.source == "scripted_teacher" else None
        if teacher is not None and not teacher.feasibility()["feasible"]:
            print(sd, "infeasible layout, skipped", flush=True)
            continue
        label = "SCRIPTED TEACHER (privileged)" if teacher else f"LEARNED latent sys-i flow + sys-0 {Path(a.checkpoint).parent.name}"
        ents = [o.sim_body for o in s.detectables]
        if teacher is not None:
            from rrp.policies.teachers import TeacherPolicy
            vpol = TeacherPolicy(a.task, lambda e: teacher, "video:dual", ("joint_position", "gripper"))
            end = lambda i, e: bool(teacher.done or e.runtime.succeeded())
        else:
            vpol = _DualLatentVideo(pol, R, P, ents, dev, a.replan)
            end = lambda i, e: bool(e.runtime.succeeded())
        frames = []
        tick = [0]

        def frame(i, e, act, step):
            k = tick[0]
            tick[0] += 1
            if k % a.every == 0:
                rd.update_scene(e.data, camera=a.camera)
                st = " ".join(f"{n}:{v.status[:6]}" for n, v in e.runtime.instances.items())
                lines = [f"{label} | {a.task} | {a.pair} | seed {sd}", f"t={e.data.time:.1f}s {st}"]
                readout = getattr(vpol, "readout", None)
                if readout:
                    lines += ["probe(received packet), diagnostic:"] + readout
                frames.append(caption(rd.render().copy(), lines))
        ep = rollout(lambda _sd: s, vpol, H.budget_task(a.task, s.spec.env_id), [sd], batch=1, max_steps=a.max_steps,
                     hooks=[H.EndWhen(end), H.Recorder(on_step=frame), H.Settle(5)])[0]
        if ep.outcome == "crash":
            raise RuntimeError(ep.metrics.get("note") or ep.failure_reason)
        ok = bool(ep.success_privileged)
        tag = "success" if ok else "failure"
        src = "scripted_teacher" if teacher else f"learned-{Path(a.checkpoint).parent.name}"
        name = f"{dt.date.today()}_dual_{src}_{a.task}_{a.pair}_s{sd}_{tag}.mp4"
        imageio.mimsave(out / name, frames, fps=a.fps, quality=5)
        with open(out / "INDEX.md", "a") as f:
            f.write(f"- `{name}` — source={src} ckpt={a.checkpoint or '-'} robots={a.pair} task={a.task} seed={sd} "
                    f"outcome={tag} (privileged evaluator); caption shows per-slot probe of the received packet\n")
        print(name, tag, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--source", choices=["scripted_teacher", "learned_latent"], required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--probe")
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--out", default="artifacts/video")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    run(ap.parse_args(argv))
