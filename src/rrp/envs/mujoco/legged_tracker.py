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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rrp.core.provenance import file_digest
from rrp.core.paths import rrp_home
from rrp.envs.mujoco.legged_core import (EXTRA_DIM_RING, RANGE_RING_VERSION, SCAN_DIM, TERRAIN_SCAN_VERSION, LeggedBinding,
                                          range_ring_spec, terrain_scan_spec)


class TrackerMismatch(ValueError):
    code = "tracker_body_mismatch"


# public extra-input kinds (`extra_kind`) -> the public sensors, in input order, that fill the actor's extra block
PUBLIC_EXTRA = {"none": (), "terrain_scan": ("terrain_scan",), "terrain_scan+range_ring": ("terrain_scan", "range_ring")}


def extra_kind(meta: dict) -> str:
    """What the actor's extra input block is: none | terrain_scan (the PUBLIC D-146 scan, layout-versioned) | terrain_scan+range_ring
    (the scan, then the PUBLIC HS1 range ring: the gap actors) | privileged (any other extra_obs_dim: a task-specific privileged
    block the caller must supply through `extra_fn`). Keys of `PUBLIC_EXTRA` are the public kinds."""
    n = int(meta.get("extra_obs_dim") or 0)
    if n == 0:
        return "none"
    scan = (meta.get("terrain_scan") or {}).get("version") == TERRAIN_SCAN_VERSION
    ring = (meta.get("range_ring") or {}).get("version") == RANGE_RING_VERSION
    if n == SCAN_DIM and scan and not ring:
        return "terrain_scan"
    if n == EXTRA_DIM_RING and scan and ring:
        return "terrain_scan+range_ring"
    return "privileged"


class LearnedTracker:
    source = "learned_tracker"

    def __init__(self, path: str | Path, binding: LeggedBinding, body_key: str):
        import torch
        from rrp.envs.mujoco.tracker_nets import mlp
        st = torch.load(str(path), map_location="cpu", weights_only=False)
        meta = st["meta"]
        self.morph = None
        # U1: the actor also reads the upper-body joint state (binding.upper_obs), appended after the extra block
        self.upper_obs = bool(meta.get("upper_obs"))
        if meta.get("obs_format") == "morph_v1":
            if self.upper_obs:
                raise TrackerMismatch("morph_v1 trackers have no upper-body block (upper_obs)")        # W13 shared morphology-conditioned tracker (rrp.envs.morph_obs)
            from rrp.envs.mujoco.morph_obs import OBS_DIM, NS, MorphSpec
            if meta["obs_dim"] != OBS_DIM + int(meta.get("extra_obs_dim") or 0) or meta["act_dim"] != NS:
                raise TrackerMismatch("morph_v1 tracker dims do not match rrp.envs.morph_obs")
            self.morph = MorphSpec(binding.model, binding, binding.meta)
            self.transfer = body_key not in (meta.get("train_bodies") or [])   # evaluated on a body it never trained on
        elif meta["body"] != body_key:
            raise TrackerMismatch(f"tracker trained for {meta['body']}, not {body_key}")
        elif meta["act_dim"] != binding.n or meta["obs_dim"] != (binding.obs_dim + int(meta.get("extra_obs_dim") or 0)
                                                                  + (binding.upper_dim if self.upper_obs else 0)):
            raise TrackerMismatch("tracker dims do not match body binding")
        self.meta = meta
        # W13 task experts: PRIVILEGED extra inputs (e.g. a height scan) appended to the observation; the caller sets
        # extra_fn(data) -> np.ndarray (meta extra_obs_dim). None for every plain tracker.
        self.extra_fn = None
        self.extra_kind = extra_kind(meta)
        self.spec = None                                 # '<body>:<version>' when loaded from the registry
        if "terrain_scan" in PUBLIC_EXTRA.get(self.extra_kind, ()):
            want, got = terrain_scan_spec(), meta["terrain_scan"]
            if any(got.get(k) != want[k] for k in ("shape", "cell_m", "x0_m", "y0_m", "frame", "range_m")):
                raise TrackerMismatch("tracker was trained with a different terrain-scan layout")
        if self.extra_kind == "terrain_scan+range_ring":
            want, got = range_ring_spec(), meta["range_ring"]
            if any(got.get(k) != want[k] for k in ("n", "frame", "angle0", "range_m", "height")):
                raise TrackerMismatch("tracker was trained with a different range-ring layout")
        self.net = mlp(meta["obs_dim"], tuple(meta["hidden"]), meta["act_dim"])
        self.net.load_state_dict(st["actor"])
        self.net.eval()
        self.mean = st["obs_mean"].numpy()
        self.std = np.sqrt(st["obs_var"].numpy() + 1e-8)
        self.b = binding
        self.torch = torch
        self.dt = float(meta["control_dt"])
        self.contact_model = meta.get("contact_model", "contact_v1")   # pre-v2 actors carry no key: v1 physics
        # torque-limit version the actor was trained with (pre-2026-09-27 actors: the legacy gains table)
        self.actuator_limits = meta.get("actuator_limits", "legacy_gains_v0")
        from rrp.bodies.actuator import LIMITS_CHANGED
        scene_limits = binding.meta.get("actuator_limits")
        import os
        self.limits_override = bool(os.environ.get("RRP_ALLOW_LIMITS_MISMATCH"))   # EVALUATION ONLY (transfer checks); recorded
        if self.morph is None and body_key in LIMITS_CHANGED and scene_limits and scene_limits != self.actuator_limits \
                and not self.limits_override:
            raise TrackerMismatch(f"{body_key} tracker trained with actuator limits {self.actuator_limits}, scene uses "
                                  f"{scene_limits} (set RRP_ACTUATOR_LIMITS={self.actuator_limits} to run it)")
        cv = "" if self.contact_model == "contact_v1" else f":{self.contact_model}"
        self.version = f"learned_tracker:{body_key}:iter{meta.get('iter')}{cv}"
        if self.morph is not None:
            self.version = f"learned_tracker:shared_morph_v1{':transfer' if self.transfer else ''}:{body_key}:iter{meta.get('iter')}{cv}"
        # W8: exact actor identity (two v2 actors can share an iter)
        self.path = str(path)
        self.sha256 = file_digest(Path(path), length=None)
        self.run = (meta.get("args") or {}).get("out")
        self.reset()

    def reset(self, phase: float = 0.0):
        self.last_a = np.zeros(self.b.n)
        self.phase = phase
        if getattr(self, "morph", None) is not None:
            from rrp.envs.mujoco.morph_obs import NS
            self.last_slot = np.zeros(NS)

    def _act_morph(self, data, cmd) -> np.ndarray:
        o = self.morph.obs(self.b, data, cmd, self.last_slot, self.phase, bool(self.meta.get("clock_gate")))
        if self.extra_fn is not None:
            o = np.concatenate([o, self.extra_fn(data)]).astype(np.float32)
        x = np.clip((o - self.mean) / self.std, -5, 5).astype(np.float32)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(x)[None])[0].numpy().astype(np.float64)
        a = np.clip(a, -5, 5) * self.morph_pm
        self.last_slot = a
        self.last_a = self.morph.from_slots(a)
        self.phase = (self.phase + self.dt / self.b.period) % 1.0
        return self.b.targets(self.last_a, target_margin=float(self.meta.get("target_margin") or 0.0))

    def act(self, data, cmd) -> np.ndarray:
        if self.morph is not None:
            return self._act_morph(data, cmd)
        o = self.b.public_obs(data, cmd, self.last_a, self.phase, bool(self.meta.get("clock_gate")))
        if self.extra_fn is not None:
            o = np.concatenate([o, self.extra_fn(data)]).astype(np.float32)
        if self.upper_obs:
            o = np.concatenate([o, self.b.upper_obs(data)]).astype(np.float32)
        x = np.clip((o - self.mean) / self.std, -5, 5).astype(np.float32)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(x)[None])[0].numpy().astype(np.float64)
        a = np.clip(a, -5, 5)
        self.last_a = a
        amp = float(self.meta.get("ref_ff") or 0.0)
        ref = self.b.ref_offset(self.phase, cmd, amp, self.meta.get("ref_ff_vmax")) if amp else None
        self.phase = (self.phase + self.dt / self.b.period) % 1.0
        return self.b.targets(a, ref, target_margin=float(self.meta.get("target_margin") or 0.0))

    @property
    def morph_pm(self):
        return self.morph.present.astype(np.float64)

    def state(self):
        if self.morph is not None:
            return dict(last_a=self.last_a.tolist(), last_slot=self.last_slot.tolist(), phase=self.phase)
        return dict(last_a=self.last_a.tolist(), phase=self.phase)

    def load(self, st):
        self.last_a = np.array(st["last_a"])
        self.phase = st["phase"]
        if "last_slot" in st:
            self.last_slot = np.array(st["last_slot"])


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


TRACKER_DIR = rrp_home() / "artifacts" / "trackers"
V1 = "contact_v1"                                   # the v1 actor lives at trackers/<body>/ (no version subdirectory)


@dataclass(frozen=True)
class TrackerEntry:
    """One registered actor: `artifacts/trackers/<body>/<version>/meta.json` (v1: `<body>/meta.json`, version contact_v1)."""
    body: str
    version: str
    sha256: str | None          # meta pin of the actor file (None = unpinned); checked against actor.pt on load
    obs_format: str | None      # meta obs_format (None = per-body; morph_v1 = shared morphology-conditioned)
    extra_obs: str              # none | terrain_scan | terrain_scan+range_ring | privileged   (see extra_kind)
    gate: bool                  # actor was trained with the gait-clock gate
    decision: str               # meta decision, else "rejected" when the version name says so, else "accepted"
    store: Path                 # directory holding actor.pt / meta.json

    @property
    def spec(self) -> str:
        return f"{self.body}:{self.version}"

    @property
    def actor(self) -> Path:
        return self.store / "actor.pt"


def scan_trackers(root: Path | None = None) -> dict[tuple[str, str], TrackerEntry]:
    out: dict[tuple[str, str], TrackerEntry] = {}
    root = Path(TRACKER_DIR if root is None else root)
    for f in sorted(root.glob("*/meta.json")) + sorted(root.glob("*/*/meta.json")):
        meta = json.loads(f.read_text())
        body = f.parent.name if f.parent.parent == root else f.parent.parent.name
        version = V1 if f.parent.parent == root else f.parent.name
        decision = meta.get("decision") or ("rejected" if "rejected" in version else "accepted")
        out[(body, version)] = TrackerEntry(
            body=body, version=version, sha256=meta.get("sha256"), obs_format=meta.get("obs_format"),
            extra_obs=extra_kind(meta), gate=bool(meta.get("clock_gate")), decision=decision, store=f.parent)
    return out


TRACKERS: dict[tuple[str, str], TrackerEntry] = scan_trackers()


def parse_spec(spec: str) -> tuple[str, str]:
    body, sep, version = spec.partition(":")
    if not sep or not body or not version:
        raise ValueError(f"tracker spec must be '<body>:<version>', got {spec!r}")
    return body, version


def get_entry(spec: str) -> TrackerEntry:
    key = parse_spec(spec)
    if key not in TRACKERS:
        raise KeyError(f"unknown tracker {spec!r}; registered: {sorted(':'.join(k) for k in TRACKERS)}")
    return TRACKERS[key]


def tracker_path(body_key: str, contact: str | None = "v1") -> Path:
    """Actor file of the body's tracker for a contact model: v1 at trackers/<body>/actor.pt, vN at trackers/<body>/contact_vN/."""
    from rrp.bodies.contact import resolve
    c = resolve(contact)
    return Path(TRACKER_DIR) / body_key / ("actor.pt" if c == "v1" else f"contact_{c}/actor.pt")


def load_tracker(body_key: str, binding: LeggedBinding, meta: dict, kind: str = "auto", tracker: str | None = None):
    """kind: learned | cpg | auto (learned if a frozen, validated actor exists, else cpg for procedural).
    Default actor: the one matching the physics the scene was built with (meta['contact_model'], default v1).
    `tracker="<body>:<version>"` picks a registered actor by id (sha pin checked); it must match the scene's contact model."""
    contact = meta.get("contact_model", "contact_v1")
    if tracker is not None:
        e = get_entry(tracker)
        if not e.actor.exists():
            raise FileNotFoundError(e.actor)
        if e.sha256 is not None:
            got = file_digest(e.actor, length=None)
            if got != e.sha256:
                raise TrackerMismatch(f"tracker {tracker} sha256 {got[:12]} does not match its registry pin {e.sha256[:12]}")
        t = LearnedTracker(e.actor, binding, body_key)
        t.spec = e.spec
        if t.contact_model != contact:
            raise TrackerMismatch(f"tracker {tracker} trained in {t.contact_model}, scene uses {contact}")
        return t
    p = tracker_path(body_key, contact)
    if kind in ("learned", "auto") and p.exists():
        t = LearnedTracker(p, binding, body_key)
        if t.contact_model != contact:
            raise TrackerMismatch(f"tracker trained in {t.contact_model}, scene uses {contact}")
        return t
    if kind == "learned":
        raise FileNotFoundError(p)
    if meta.get("synthetic"):
        return CPGTracker(binding, meta)
    raise FileNotFoundError(f"no tracker for {body_key}")


def eligibility(body_key: str) -> dict | None:
    f = TRACKER_DIR / body_key / "eligibility.json"
    return json.loads(f.read_text()) if f.exists() else None
