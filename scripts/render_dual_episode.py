"""Render labelled videos of dual-arm episodes (support_insert / handover) on the corrected latent path.

  render_dual_episode.py --task handover --pair panda_pg2__ur5e_pg2 --seeds 3000001 --source learned_latent \
      --checkpoint artifacts/runs/dualarm_flow_sem_v1/policy.pt [--probe probe.pt]
  render_dual_episode.py --task support_insert --pair parm5_pg2__parm5_pg2 --seeds 3000001 --source scripted_teacher

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
import numpy as np
from PIL import Image, ImageDraw


def caption(frame, lines):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def run(a):
    import torch
    from rrp.control.dual_validate import make_session
    from rrp.control.dual_teachers import TEACHERS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = R = P = None
    if a.source == "learned_latent":
        from rrp.evaluation.dual_latent_eval import DualLatentPolicy, DualLatentSystem0, probe_readout
        from rrp.learning.checkpoint import load_checkpoint
        from rrp.learning.latent_train import load_representation
        if dev == "cuda":
            from rrp.ops.gpu import apply_cap
            apply_cap()
        pol = DualLatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=8)
        _, _, R, P, _ = load_representation(Path(load_checkpoint(a.checkpoint)["config"]["representation"]), dev)
        if a.probe:
            from rrp.model.latent_probes import PacketProbe
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
        readout = []
        if pol is not None:
            s0 = DualLatentSystem0(R, pol.featurizer(s), latent_space_version=pol.lsv, realizer_compat_version=pol.rcv,
                                   device=dev)
        frames = []
        for k in range(a.max_steps):
            if teacher:
                s.step(teacher.act())
                done = teacher.done or s.runtime.succeeded()
            else:
                if k % a.replan == 0 or s0.packet is None:
                    p = pol.packets([s])[0]
                    try:
                        s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
                    except Exception as e:          # noqa: BLE001 - rejected packet: fallback hold
                        print("packet rejected:", e, flush=True)
                    with torch.no_grad():
                        z = torch.from_numpy(p.z)[None].to(dev)
                        o = P(z, torch.tensor([p.assembly_mask], device=dev), len(ents))
                    readout = probe_readout({kk: v.cpu() for kk, v in o.items()}, len(ents), ent_names=ents)
                s.step(s0.tick(s))
                done = s.runtime.succeeded()
            if k % a.every == 0:
                rd.update_scene(s.data, camera=a.camera)
                st = " ".join(f"{e}:{v.status[:6]}" for e, v in s.runtime.instances.items())
                lines = [f"{label} | {a.task} | {a.pair} | seed {sd}", f"t={s.data.time:.1f}s {st}"]
                if readout:
                    lines += ["probe(received packet), diagnostic:"] + readout
                frames.append(caption(rd.render().copy(), lines))
            if done:
                break
        for _ in range(5):
            s.step(None)
        ok = s.privileged_success()
        tag = "success" if ok else "failure"
        src = "scripted_teacher" if teacher else f"learned-{Path(a.checkpoint).parent.name}"
        name = f"{dt.date.today()}_dual_{src}_{a.task}_{a.pair}_s{sd}_{tag}.mp4"
        imageio.mimsave(out / name, frames, fps=a.fps, quality=5)
        with open(out / "INDEX.md", "a") as f:
            f.write(f"- `{name}` — source={src} ckpt={a.checkpoint or '-'} robots={a.pair} task={a.task} seed={sd} "
                    f"outcome={tag} (privileged evaluator); caption shows per-slot probe of the received packet\n")
        print(name, tag, flush=True)


if __name__ == "__main__":
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
    run(ap.parse_args())
