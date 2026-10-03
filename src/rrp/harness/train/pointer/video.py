"""Labelled demo videos of any pointer policy (`rrp train pointer video`; peer only: rendering). Frames come from the
privileged renderer for humans only; the caption bar names the controller source."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FrameHook:
    """Rollout hook: the ComputerWorld screen (env.render, the privileged renderer; frames are for humans only) at reset
    and after every tick, downscaled, with a caption bar naming the controller source."""

    def __init__(self, caption: str, scale: float = 0.5):
        self.caption, self.scale, self.frames = caption, scale, {}

    def _grab(self, i, env, note=""):
        from PIL import Image, ImageDraw
        im = Image.fromarray(env.render()[..., :3])
        im = im.resize((int(im.width * self.scale), int(im.height * self.scale)))
        bar = Image.new("RGB", (im.width, 44), (20, 20, 20))
        d = ImageDraw.Draw(bar)
        d.text((6, 4), self.caption, fill=(255, 255, 255))
        d.text((6, 24), f"{env.task_name} seed {env.seed}  t={env.time:.1f}s  {note}", fill=(255, 220, 120))
        out = Image.new("RGB", (im.width, im.height + 44))
        out.paste(bar, (0, 0))
        out.paste(im, (0, 44))
        self.frames.setdefault(i, []).append(np.asarray(out))

    def on_reset(self, i, env, obs):
        self._grab(i, env)

    def on_step(self, i, env, act, step):
        self._grab(i, env)

    def on_end(self, i, env, ep):
        tag = ep.outcome + (f" ({ep.failure_reason})" if ep.failure_reason else "")
        self._grab(i, env, tag.upper())
        self.frames[i] += [self.frames[i][-1]] * 10         # hold the final frame 1 s
        return {}


def cmd_video(a):
    """Render labelled episodes of one policy (JSON spec as for rrp eval) into one mp4 (peer only: rendering)."""
    import imageio
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.base import make_policy
    name, _, kw = a.policy.partition("=")
    pol = make_policy(name, **(json.loads(kw) if kw else {}))
    from rrp.cli.harness import _env_kw
    env_kw = _env_kw(getattr(a, "env_kw", None)) or None
    frames, rows = [], []
    for task, seed in (x.split("@") for x in a.episodes):
        h = FrameHook(f"{pol.info.source.upper()}  {pol.info.name}  {pol.info.variant or ''}  {a.caption}")
        ep = evaluate(pol, "computerworld", task, "cw_pointer", [int(seed)], batch=1, hooks=[h],
                      env_kw=env_kw)[0]
        frames += h.frames[0]
        rows.append(dict(task=task, seed=int(seed), outcome=ep.outcome, failure_reason=ep.failure_reason,
                         steps=ep.steps))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=10, quality=6)
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(policy=a.policy, source=pol.info.source, env_kw=env_kw,
                                                                episodes=rows), indent=1))
    print(json.dumps(rows))
