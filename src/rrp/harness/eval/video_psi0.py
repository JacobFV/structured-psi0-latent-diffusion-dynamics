"""`rrp video psi0`: short labelled videos of Ψ₀ closed-loop episodes on a SIMPLE task (Isaac Sim, peer GPU lease).

The frames are the policy's own head camera (`head_stereo_left`, a deployable observation), captured from the same
`evaluate` path as the `eval_r2` stage (same policy kwargs, body, hooks). Each video is a NEW rollout: Ψ₀ samples its flow
noise, so the outcome can differ from the recorded eval row of the same seed; the caption states this rollout's outcome
(privileged SIMPLE success predicate) and the controller source. Arms run one after another in child processes inside
one lease (one 2.6 B-parameter model in memory at a time).

  rrp video psi0 --task G1WholebodyTabletopGraspMP-v0 --seeds 3,5 \
      --arm direct=artifacts/runs/psi0/psi0-tabletop/train_bc_s0/final.pt \
      --arm structured=artifacts/runs/psi0/psi0-tabletop/train_flow_s0/final.pt \
      --stage-a artifacts/runs/psi0/psi0-tabletop/train_rep_s0/stage_a.pt --out artifacts/video
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ARM_LABEL = {"released": "RELEASED upstream Psi0 ckpt (not ours)", "direct": "LEARNED Psi0 direct head (ours)",
             "structured": "LEARNED Psi0 + structure (ours, generated packet)"}


def _strip(frame: np.ndarray, lines: list[str], scale: int = 2) -> np.ndarray:
    """Frame upscaled `scale`x under a black caption strip (the strip never covers the image)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(frame[..., :3]).astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    h = 14 * len(lines) + 8
    out = Image.new("RGB", (im.width, im.height + h), (0, 0, 0))
    out.paste(im, (0, h))
    d = ImageDraw.Draw(out)
    for i, t in enumerate(lines):
        d.text((6, 4 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(out)


def _truth_row(env) -> dict:
    """Privileged per-step diagnostic (source privileged:sim): palm-target distance per hand, palm-target contact,
    target height, task reward. Read after the step, from the simulator; the policy never sees it."""
    t = env.truth()
    tgt = np.asarray(t["objects"].get("target", np.full(3, np.nan)), np.float32).reshape(-1)[:3]
    return dict(step=int(t["step"]), reward=float(t["reward"]), target_z=float(tgt[2]),
                dist={s: float(np.linalg.norm(np.asarray(p)[:3] - tgt)) for s, p in t["palm"].items()},
                contact={k: bool(v) for k, v in t["contact"].items() if k.endswith(":target")}, success=bool(t["success"]))


def run_arm(a) -> None:
    import imageio
    from rrp.envs.simple import SimpleEnv
    from rrp.harness.eval.evaluate import evaluate, task_hooks
    from rrp.policies.base import make_policy

    frames: dict = {}
    diag: dict = {}         # PRIVILEGED sim truth per step (labels for the diagnosis only; never a policy input)
    cur: dict = {}
    orig_reset, orig_step = SimpleEnv.reset, SimpleEnv.step

    def reset(self, seed=None):
        r = orig_reset(self, seed)
        k = (int(seed or 0), sum(1 for s, _ in frames if s == int(seed or 0)))
        frames[k] = [np.asarray(self._obs["image"]).copy()]
        diag[k] = [_truth_row(self)]
        cur[id(self)] = k
        return r

    def step(self, command=None):
        r = orig_step(self, command)
        k = cur.get(id(self))
        if k is not None:
            frames[k].append(np.asarray(self._obs["image"]).copy())
            diag[k].append(_truth_row(self))
        return r

    SimpleEnv.reset, SimpleEnv.step = reset, step
    task = f"simple/{a.task}"
    kw: dict = dict(task=task)
    if a.arm == "released":
        kind = "psi0_direct"
    else:
        kind = f"psi0_{a.arm}"
        kw["weights"] = a.weights
        if a.arm == "structured":
            kw["stage_a"] = a.stage_a
    pol = make_policy(kind, **kw)
    seeds = [int(s) for s in a.seeds.split(",")]
    rows_out = Path(a.out) / f"psi0_video_{a.arm}.jsonl"
    eps = evaluate(pol, "simple", task, "g1_simple", seeds, batch=len(seeds), hooks=task_hooks(task, "simple"),
                   out=rows_out)
    day = dt.date.today().isoformat()
    index = Path(a.out) / "INDEX.md"
    for e in eps:
        row = e.row()
        fr = frames[(int(row["seed"]), 0)]
        outcome = row.get("outcome")
        src = str(row.get("source"))
        name = f"{day}_psi0_{a.task_key}_{a.arm}_s{row['seed']}_{outcome}.mp4"
        lines = [ARM_LABEL[a.arm], f"source: {src[-60:]}",
                 f"{a.task}  seed {row['seed']}  (new rollout; outcome: {outcome}, {row.get('steps')} steps)",
                 "camera: policy head_stereo_left; success judged by privileged SIMPLE predicate"]
        step = max(1, len(fr) // 400)
        with imageio.get_writer(str(Path(a.out) / name), fps=max(1, 50 // step // 2), codec="libx264", quality=4,   # tracked videos must stay < 4 MB (test_layout)
                                macro_block_size=8) as w:
            for i, f in enumerate(fr[::step]):
                w.append_data(_strip(f, lines + [f"t = {i * step} control steps"]))
        with open(index, "a") as fh:
            fh.write(f"- {name}: {ARM_LABEL[a.arm]}; {a.task} seed {row['seed']}; outcome {outcome} ({row.get('steps')} steps); "
                     f"source {src}; new rollout, not the recorded eval row (D-147 T7)\n")
        (Path(a.out) / name.replace(".mp4", ".diag.json")).write_text(json.dumps(dict(
            source="privileged:sim (diagnostic labels; not a policy input)", arm=a.arm, policy_source=src, seed=row["seed"],
            outcome=outcome, steps=row.get("steps"), per_step=diag[(int(row["seed"]), 0)])))
        print(json.dumps(dict(video=name, seed=row["seed"], outcome=outcome, steps=row.get("steps"))), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp video psi0")
    ap.add_argument("--task", required=True, help="SIMPLE task id without the simple/ prefix")
    ap.add_argument("--task-key", default="tabletop")
    ap.add_argument("--seeds", required=True, help="comma list of distinct seeds (one batch)")
    ap.add_argument("--arm", action="append", required=True, help="released | direct=WEIGHTS | structured=WEIGHTS (repeatable)")
    ap.add_argument("--stage-a", help="structured arm: stage_a.pt")
    ap.add_argument("--out", default="artifacts/video")
    ap.add_argument("--_one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a._one:
        arm, _, w = a.arm[0].partition("=")
        a.arm, a.weights = arm, w or None
        run_arm(a)
        return 0
    rc = 0
    for spec in a.arm:      # one child per arm: one model in memory at a time
        cmd = [sys.executable, "-m", "rrp.cli", "video", "psi0", "--_one", "--task", a.task, "--task-key", a.task_key,
               "--seeds", a.seeds, "--arm", spec, "--out", a.out] + (["--stage-a", a.stage_a] if a.stage_a else [])
        rc = subprocess.call(cmd) or rc
    return rc

