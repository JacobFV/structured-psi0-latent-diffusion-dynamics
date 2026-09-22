"""Deployable locomotion trackers: base-velocity command -> joint position targets.

Two families, both consuming ONLY the public tracker observation (rrp.control.legged_core):
  * LearnedTracker  - PPO actor exported by rrp.control.tracker_training (source label
                      `learned_tracker`; trained with a privileged critic, deployed without it).
  * CPGTracker      - scripted open-loop central pattern generator for procedural sprawl/mammal
                      legged bodies (source label `scripted_controller`); the baseline gait.

A tracker is bound to one body (spec hash of the body it was trained/validated on). Applying a
tracker to a different body raises TrackerMismatch unless explicitly re-validated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .legged_core import LeggedBinding


class TrackerMismatch(ValueError):
    code = "tracker_body_mismatch"


class LearnedTracker:
    source = "learned_tracker"

    def __init__(self, path: str | Path, binding: LeggedBinding, body_key: str):
        import torch
        from rrp.control.tracker_nets import mlp
        st = torch.load(str(path), map_location="cpu", weights_only=False)
        meta = st["meta"]
        if meta["body"] != body_key:
            raise TrackerMismatch(f"tracker trained for {meta['body']}, not {body_key}")
        if meta["act_dim"] != binding.n or meta["obs_dim"] != binding.obs_dim:
            raise TrackerMismatch("tracker dims do not match body binding")
        self.meta = meta
        self.net = mlp(meta["obs_dim"], tuple(meta["hidden"]), meta["act_dim"])
        self.net.load_state_dict(st["actor"])
        self.net.eval()
        self.mean = st["obs_mean"].numpy()
        self.std = np.sqrt(st["obs_var"].numpy() + 1e-8)
        self.b = binding
        self.torch = torch
        self.dt = float(meta["control_dt"])
        self.version = f"learned_tracker:{body_key}:iter{meta.get('iter')}"
        self.reset()

    def reset(self, phase: float = 0.0):
        self.last_a = np.zeros(self.b.n)
        self.phase = phase

    def act(self, data, cmd) -> np.ndarray:
        o = self.b.public_obs(data, cmd, self.last_a, self.phase)
        x = np.clip((o - self.mean) / self.std, -5, 5).astype(np.float32)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(x)[None])[0].numpy().astype(np.float64)
        a = np.clip(a, -5, 5)
        self.last_a = a
        self.phase = (self.phase + self.dt / self.b.period) % 1.0
        return self.b.targets(a)

    def state(self):
        return dict(last_a=self.last_a.tolist(), phase=self.phase)

    def load(self, st):
        self.last_a = np.array(st["last_a"])
        self.phase = st["phase"]


class CPGTracker:
    """Scripted gait for procedural legged bodies (3 joints/leg: yaw-or-roll, lift, knee).

    Leg groups alternate (tripod for 6, trot for 4, wave-ish pairs for 8). Stance legs sweep
    their yaw/pitch backwards proportional to the commanded forward speed and turn rate; swing
    legs lift. Open-loop w.r.t. body state (uses only the command and its internal clock)."""
    source = "scripted_controller"

    def __init__(self, binding: LeggedBinding, meta: dict, stride_gain: float = 1.0):
        self.b, self.meta = binding, meta
        self.dt = 0.02
        self.version = f"cpg:{meta['name']}"
        acts = meta["legged"]["policy_actuators"]
        self.legs = []  # (idx_yaw, idx_lift, idx_knee, side_sign, x_pos_rank)
        names = sorted({a.split("_")[1] + "_" + a.split("_")[2] for a in acts})  # leg_l0 ...
        n_side = len(names) // 2
        for k, leg in enumerate(names):
            idx = [acts.index(f"act_{leg}_{j}") for j in ("coxa", "femur", "tibia")]
            side = 1 if leg.startswith("leg_l") else -1
            i = int(leg[5:])
            group = (i + (0 if side > 0 else 1)) % 2
            self.legs.append(dict(idx=idx, side=side, rank=i, group=group, n_side=n_side))
        self.layout = meta["params"].get("layout", "sprawl")
        self.gain = stride_gain
        self.reset()

    def reset(self, phase: float = 0.0):
        self.phase = phase

    def act(self, data, cmd) -> np.ndarray:
        vx, vy, wz = cmd
        q = self.b.q0.copy()
        if abs(vx) < 0.02 and abs(vy) < 0.02 and abs(wz) < 0.03:
            return np.clip(q, self.b.lo, self.b.hi)          # stand still: default stance, no stepping
        self.phase = (self.phase + self.dt / self.b.period) % 1.0
        for leg in self.legs:
            ph = (self.phase + 0.5 * leg["group"]) % 1.0
            swing = ph < 0.5
            s = ph / 0.5 if swing else (ph - 0.5) / 0.5            # 0..1 within sub-phase
            # stride displacement for this leg: forward speed + turn (x offset of leg sets turn lever)
            xl = 1.0 - 2.0 * leg["rank"] / max(leg["n_side"] - 1, 1)  # +1 front .. -1 rear
            stride = self.gain * (vx * 1.6 - leg["side"] * wz * 0.35 + 0 * vy)
            sweep = (s - 0.5) * stride if swing else (0.5 - s) * stride   # swing forward, stance backward
            iy, il, ik = leg["idx"]
            if self.layout == "sprawl":
                q[iy] -= leg["side"] * sweep
                lift = 0.45 * math.sin(math.pi * s) if swing else 0.0
                q[il] += lift
                q[ik] += 0.3 * lift
            else:  # mammal: pitch sweep at the hip, knee flex on swing
                q[il] -= 0.6 * sweep
                lift = 0.5 * math.sin(math.pi * s) if swing else 0.0
                q[il] += 0.3 * lift
                q[ik] -= lift
            _ = xl
        return np.clip(q, self.b.lo, self.b.hi)

    def state(self):
        return dict(phase=self.phase)

    def load(self, st):
        self.phase = st["phase"]


TRACKER_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "trackers"


def load_tracker(body_key: str, binding: LeggedBinding, meta: dict, kind: str = "auto"):
    """kind: learned | cpg | auto (learned if a frozen, validated actor exists, else cpg for procedural)."""
    p = TRACKER_DIR / body_key / "actor.pt"
    if kind in ("learned", "auto") and p.exists():
        return LearnedTracker(p, binding, body_key)
    if kind == "learned":
        raise FileNotFoundError(p)
    if meta.get("synthetic"):
        return CPGTracker(binding, meta)
    raise FileNotFoundError(f"no tracker for {body_key}")


def eligibility(body_key: str) -> dict | None:
    f = TRACKER_DIR / body_key / "eligibility.json"
    return json.loads(f.read_text()) if f.exists() else None
