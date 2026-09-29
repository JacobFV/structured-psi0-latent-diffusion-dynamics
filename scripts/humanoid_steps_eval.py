"""W13 P2: C-MuJoCo evaluation of an h_steps expert (privileged_teacher:rl_expert with height scan) driven by the scripted
heading command, full self-collision, contact_v2. usage: python scripts/humanoid_steps_eval.py <body> <actor.pt> <seed0> <n> <out.json> [h_frac]"""
import json
import sys
import time

import numpy as np

import rrp.envs.mujoco.legged as _L
from rrp.envs.mujoco.humanoid_scenes import build_h_steps
from rrp.envs.mujoco.legged import LeggedSession
from rrp.envs.mujoco.legged_tracker import LearnedTracker
from rrp.policies.teachers.humanoid import StepsHeadingTeacher

body, actor, s0, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
hf = float(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "none" else None
RENDER = int(sys.argv[7]) if len(sys.argv) > 7 else 0          # render the first N episodes (MUJOCO_GL=egl)
LABEL = sys.argv[8] if len(sys.argv) > 8 else ""


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
CUR = {}


def _load(key, binding, meta, kind="auto"):
    """The expert with its PRIVILEGED scan attached before the session's first (settling) tick."""
    from rrp.envs.mujoco.humanoid_scenes import steps_scan_np
    tr = LearnedTracker(actor, binding, key)
    CUR["expert"] = bool(int(tr.meta.get("extra_obs_dim") or 0))
    if CUR["expert"]:
        L, h = CUR["L"], CUR["h"]
        tr.extra_fn = lambda d: steps_scan_np(d.qpos[binding.qa:binding.qa + 7], L, h)
    return tr


_L.load_tracker = _load
rows = []
for seed in range(s0, s0 + n):
    sc = build_h_steps(body, seed, h_frac=hf)
    CUR.update(L=float(sc.meta["L"]), h=float(sc.meta["staircase"]["h"]))
    s = LeggedSession(sc, tracker_kind="learned", seed=seed)   # plain trackers (no scan input) run blind
    s.reset(seed)
    te = StepsHeadingTeacher(s)
    t0, status = time.time(), "timeout"
    frames = []
    rend = _render_setup(s) if len(rows) < RENDER else None
    src = s.tracker.version + (" +privileged scan" if s.tracker.extra_fn else " (blind)")
    for k in range(400):                            # 10 Hz commands, 40 s
        s.step(te.act())
        if rend is not None:
            rend[0].update_scene(s.data, camera=rend[1])
            frames.append(_caption(rend[0].render().copy(), [
                f"PRIVILEGED TEACHER (rl_expert) + scripted heading command | {LABEL}" if s.tracker.extra_fn else
                f"learned tracker (blind) + scripted heading command | {LABEL}",
                f"{src}", f"h1 h_steps  step h={sc.meta['h_frac']:.2f} L  seed {seed}  t={s.data.time:4.1f}s  {'FELL' if s.fell else ''}"]))
        if s.fell:
            status = "fell"
            break
        if s.runtime.succeeded():
            status = "success"
            break
    x = float(s.data.qpos[s.binding.qa])
    if frames:
        import datetime as _dt
        import imageio
        from pathlib import Path
        vd = Path("artifacts/video")
        name = f"{_dt.date.today()}_humanoid_steps_{body}_h{sc.meta['h_frac']:.2f}_s{seed}_{LABEL or 'eval'}_{status}.mp4"
        frames += [frames[-1]] * 10
        imageio.mimsave(str(vd / name), frames, fps=10, quality=5)
        with open(vd / "INDEX.md", "a") as f:
            f.write(f"- `{name}` — W13 h_steps: {src}; body {body}, step height {sc.meta['h_frac']:.2f} L, seed {seed}, outcome "
                    f"{status} (privileged evaluator), 1x speed. source label: "
                    f"{'privileged_teacher:rl_expert (height scan) + scripted_teacher heading command' if s.tracker.extra_fn else 'learned_tracker (blind) + scripted_teacher heading command'}.\n")
    rows.append(dict(seed=seed, status=status, h_frac=sc.meta["h_frac"], x=x, x_end=sc.meta["x_end"], sim_s=float(s.data.time),
                     wall_s=time.time() - t0))
    print(rows[-1], flush=True)
summ = dict(body=body, actor=actor, n=n, source=("privileged_teacher:rl_expert + scripted_teacher command" if CUR.get("expert")
                                                  else "learned_tracker (blind) + scripted_teacher command"),
            success=sum(r["status"] == "success" for r in rows), fell=sum(r["status"] == "fell" for r in rows), rows=rows)
json.dump(summ, open(out, "w"), indent=1)
print({k: v for k, v in summ.items() if k != "rows"})
