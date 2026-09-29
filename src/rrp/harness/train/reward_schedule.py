"""Performance-gated reward schedule for gait_v2 tracker training.

alpha in [0, 1] moves the reward from gait-shaping PRIORS (air time, swing clearance, contact phase; they decay
to a floor) to NATURAL objectives (power / cost of transport, torque, action rate, jerk, touchdown impact; they
ramp up). PERMANENT terms (tracking, falls, orientation/height, stance slip, joint limits, clearance floor, and the
zero-command standing terms stand_contact / stand_still / stand_vel) never change. See PRIOR_TERMS / NATURAL_TERMS /
PERMANENT_STANDING_TERMS and RewardCfg.effective in rrp.envs.legged_core.
Revision 2026-09-27: stand_contact moved PRIOR -> PERMANENT (standing still on command is a task requirement, W8 halt
failures), and every command sampler issues at least MIN_STOP_SHARE (10%) zero commands.

alpha advances by `step` only when a whole evaluation window (`every` iterations) is within the ADVANCE
thresholds. It backs off one step when any metric crosses the BACK-OFF thresholds. It never moves on
wall time. Window metrics come from the envs' gate accumulators (steps with a translational command):
    track_rel_err = sum|c_xy - v_xy| / sum|c_xy|,  slip_ratio = sum(mean stance slip) / sum|v_xy|,
    fall_rate = falls / finished episodes,  cot = sum power / sum(m g |v_xy|).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


def window_metrics(stats: list) -> dict:
    gm = dict(steps=0, track_err=0.0, cmd=0.0, slip=0.0, speed=0.0, power=0.0, cot_den=0.0)
    eps = [s for s in stats if "fell" in s]
    for s in stats:
        if "gm" in s:
            for k in gm:
                gm[k] += s["gm"][k]
    out = dict(steps=gm["steps"], episodes=len(eps),
               fall_rate=(sum(e["fell"] for e in eps) / len(eps)) if eps else 0.0)   # no episode ended: no falls
    tc = sum(s["gm"].get("turn_cmd", 0.0) for s in stats if "gm" in s)
    if tc > 0:
        out["turn_ratio"] = sum(s["gm"].get("turn_w", 0.0) for s in stats if "gm" in s) / tc
    if gm["steps"]:
        out.update(track_rel_err=gm["track_err"] / max(gm["cmd"], 1e-9),
                   slip_ratio=gm["slip"] / max(gm["speed"], 1e-9),
                   cot=gm["power"] / max(gm["cot_den"], 1e-9))
    return out


@dataclass
class AlphaGate:
    step: float = 0.1
    every: int = 25
    warmup: int = 300                      # iterations at alpha = 0 before the first gate check
    advance: dict = field(default_factory=lambda: dict(track_rel_err=0.35, fall_rate=0.10, slip_ratio=0.20))
    backoff: dict = field(default_factory=lambda: dict(track_rel_err=0.50, fall_rate=0.25, slip_ratio=0.30))
    alpha: float = 0.0
    history: list = field(default_factory=list)

    def update(self, it: int, m: dict) -> str:
        """Returns 'advance' | 'backoff' | 'hold' | 'skip' and updates alpha."""
        if it < self.warmup or m.get("track_rel_err") is None or m.get("fall_rate") is None:
            return "skip"
        keys = ("track_rel_err", "fall_rate", "slip_ratio")
        if any(m[k] > self.backoff[k] for k in keys) and self.alpha > 0:
            self.alpha = round(max(0.0, self.alpha - self.step), 6)
            act = "backoff"
        elif all(m[k] <= self.advance[k] for k in keys) and self.alpha < 1.0:
            self.alpha = round(min(1.0, self.alpha + self.step), 6)
            act = "advance"
        else:
            act = "hold"
        self.history.append(dict(iter=it, action=act, alpha=self.alpha, **{k: m[k] for k in keys}))
        return act

    def state(self) -> dict:
        return asdict(self)


TERRAIN_CURRICULUM_VERSION = "terrain_curriculum_v1"


@dataclass
class TerrainCurriculum:
    """D-126 #15: rough-terrain curriculum for tracker training, gated like AlphaGate (never on wall time).

    `level` in [0, 1] scales the envs/perturb.py `bumps_v1` heightfield: worker w of W trains at level x (w + 1) / W x amp_max,
    so the pool always covers [0, level x amp_max]. Every `every` iterations after `warmup`, when alpha >= after_alpha: +step when the
    window's metrics are all within ADVANCE, -step when any crosses BACK-OFF. track_rel_err is used only when the window has it
    (gait_v2); fall_rate always. Default OFF (the trainer only builds terrain worlds with --terrain-curriculum gated)."""
    amp_max: float = 0.10
    step: float = 0.1
    every: int = 25
    warmup: int = 300
    after_alpha: float = 0.0
    advance: dict = field(default_factory=lambda: dict(fall_rate=0.10, track_rel_err=0.40))
    backoff: dict = field(default_factory=lambda: dict(fall_rate=0.25, track_rel_err=0.60))
    level: float = 0.0
    history: list = field(default_factory=list)

    @property
    def amp_m(self) -> float:
        return self.level * self.amp_max

    def update(self, it: int, m: dict, alpha: float = 1.0) -> str:
        """Returns 'advance' | 'backoff' | 'hold' | 'skip' and updates level."""
        if it < self.warmup or m.get("fall_rate") is None:
            return "skip"
        keys = [k for k in ("fall_rate", "track_rel_err") if m.get(k) is not None]
        if any(m[k] > self.backoff[k] for k in keys) and self.level > 0:
            self.level = round(max(0.0, self.level - self.step), 6)
            act = "backoff"
        elif alpha + 1e-9 >= self.after_alpha and all(m[k] <= self.advance[k] for k in keys) and self.level < 1.0:
            self.level = round(min(1.0, self.level + self.step), 6)
            act = "advance"
        else:
            act = "hold"
        self.history.append(dict(iter=it, action=act, level=self.level, amp_max_m=self.amp_m, **{k: m[k] for k in keys}))
        return act

    def state(self) -> dict:
        return dict(asdict(self), version=TERRAIN_CURRICULUM_VERSION)
