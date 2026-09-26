"""Render short labelled clips of the SCRIPTED TEACHER (privileged) driving the frozen body tracker on legged /
humanoid bodies (waypoint_contact task). Same scene builder, teacher and tracker as the teacher route of
rrp.evaluation.legged_latent_eval (run_episode with ctl=None). No learned high-level policy is involved.

usage (GPU lease for EGL on the peer):
  MUJOCO_GL=egl python scripts/render_legged_episode.py --bodies go2,t1 --seeds 10000,10001 --out artifacts/video \
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rrp.evaluation.legged_latent_eval import run_episode, _seeds  # noqa: E402

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
    a = ap.parse_args(argv)
    a.cam_scale = {k: float(v) for k, v in (kv.split("=") for kv in a.cam_scale.split(","))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for body in a.bodies.split(","):
        arc = a.arc_only == "all" or body in a.arc_only.split(",")
        for sd in _seeds(a.seeds):
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


if __name__ == "__main__":
    main()
