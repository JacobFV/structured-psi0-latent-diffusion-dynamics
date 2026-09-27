"""Performance-gated reward schedule for gait_v2 tracker training.

alpha in [0, 1] moves the reward from gait-shaping PRIORS (air time, swing clearance, contact phase,
stand_contact; they decay to a floor) to NATURAL objectives (power / cost of transport, torque, action rate,
jerk, touchdown impact; they ramp up). PERMANENT terms (tracking, falls, orientation/height, stance slip,
joint limits) never change. See RewardCfg.effective in rrp.control.legged_core.

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
