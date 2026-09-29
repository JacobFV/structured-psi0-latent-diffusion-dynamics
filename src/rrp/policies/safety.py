"""Safety layer between a controller and the joint-position servo (D-126 roadmap #30). numpy only (plus an optional
legged-body constructor that reads the compiled model).

Per control tick, for joint-position targets u of the policy joints, given the MEASURED joint positions q and
velocities qd (encoders) and the IMU gravity direction (public):
  1. position clamp:  u in [lo + margin, hi - margin]        (joint range intersected with the actuator ctrlrange)
  2. rate limit:      |u - u_prev| <= rate_max * dt           (per joint; default rate_max = qd_max, the joint's speed limit)
  3. torque clamp:    the PD torque kp (u - q) - kd qd stays within +-effort (sourced peak torques) by clamping u to
                      q + (kd qd +- effort) / kp  (exact for the affine PD servo used by the legged bodies)
  4. velocity guard:  a joint whose |qd| exceeds qd_max gets u = q (no further drive in that direction this tick)
  5. fall detection:  tilt (from the IMU gravity direction) > fall_tilt_frac * tilt_limit for fall_ticks ticks ->
                      `fallen`; the recovery hook is called once (default: start the safe stop)
  6. safe stop:       `safe_stop(now)` ramps the target from its current value to the hold posture (default stance)
                      over stop_s seconds (rate-limited), then holds it; used by the fall hook, the OOD fallback, or
                      a caller.
Mode `safety`: off (default; the layer is not constructed) | monitor (everything is computed and every would-be
intervention is counted, the controller's target passes UNCHANGED) | enforce (the filtered target is sent).
Limits are SOURCED where a source exists (rrp.physics.actuator.SOURCED peak torques and URDF speeds); otherwise the
model's own ranges and the labelled VMAX estimates; `limits.source` records which.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

SAFETY_VERSION = "safety-1"          # rate_max default = qd_max (fixed before any result)
SAFETY_MODES = ("off", "monitor", "enforce")


@dataclass
class SafetyLimits:
    lo: np.ndarray
    hi: np.ndarray
    effort: np.ndarray
    qd_max: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    hold: np.ndarray                   # hold posture for the safe stop (default stance)
    tilt_limit: float = 1.0            # rad; the body's fall tilt
    source: str = "model"

    def to_dict(self) -> dict:
        return {k: (v.round(4).tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}


@dataclass
class SafetyConfig:
    margin: float = 0.02               # rad inside the position limits
    rate_max: float | None = None      # rad/s max target change; None = the joint's (sourced) speed limit qd_max
    qd_factor: float = 1.0             # velocity guard at qd_factor * qd_max
    fall_tilt_frac: float = 0.8        # fall when tilt > frac * tilt_limit ...
    fall_ticks: int = 5                # ... for this many consecutive ticks
    stop_s: float = 1.0                # safe-stop ramp duration
    version: str = SAFETY_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SafetyStats:
    ticks: int = 0
    clamp_pos: int = 0                 # ticks with at least one joint clamped (per stage)
    clamp_rate: int = 0
    clamp_torque: int = 0
    vel_guard: int = 0
    modified: int = 0                  # ticks where the enforced target differs from the controller's
    max_abs_change: float = 0.0        # largest |filtered - raw| (rad)
    fall_detected_t: float | None = None
    safe_stop_t: float | None = None
    safe_stop_reason: str | None = None
    events: list = field(default_factory=list)


class SafetyLayer:
    def __init__(self, limits: SafetyLimits, cfg: SafetyConfig | None = None, mode: str = "enforce",
                 on_fall: Callable[["SafetyLayer", float], None] | None = None):
        if mode not in SAFETY_MODES:
            raise ValueError(f"safety mode {mode!r} not in {SAFETY_MODES}")
        self.L, self.cfg, self.mode = limits, cfg or SafetyConfig(), mode
        self.on_fall = on_fall if on_fall is not None else (lambda layer, t: layer.safe_stop(t, reason="fall"))
        self.reset()

    def reset(self, u0=None):
        self.prev = None if u0 is None else np.asarray(u0, float).copy()
        self.stats = SafetyStats()
        self._tilt_ticks = 0
        self.fallen = False
        self._stop = None              # (t0, u_start)

    # ------------------------------------------------------------------ safe stop
    def safe_stop(self, now: float, reason: str = "request"):
        if self._stop is None:
            start = self.prev if self.prev is not None else self.L.hold
            self._stop = (float(now), np.asarray(start, float).copy())
            self.stats.safe_stop_t, self.stats.safe_stop_reason = float(now), reason
            self.stats.events.append(dict(t=float(now), event="safe_stop", reason=reason))

    @property
    def stopping(self) -> bool:
        return self._stop is not None

    def _stop_target(self, now: float) -> np.ndarray:
        t0, u0 = self._stop
        a = 1.0 if self.cfg.stop_s <= 0 else min(1.0, max(0.0, (now - t0) / self.cfg.stop_s))
        return u0 + a * (self.L.hold - u0)

    # ------------------------------------------------------------------ fall detection
    def observe_tilt(self, tilt: float, now: float):
        if tilt > self.cfg.fall_tilt_frac * self.L.tilt_limit:
            self._tilt_ticks += 1
        else:
            self._tilt_ticks = 0
        if not self.fallen and self._tilt_ticks >= self.cfg.fall_ticks:
            self.fallen = True
            self.stats.fall_detected_t = float(now)
            self.stats.events.append(dict(t=float(now), event="fall_detected", tilt=float(tilt)))
            self.on_fall(self, now)

    # ------------------------------------------------------------------ filter
    def filter(self, u, q, qd, now: float, dt: float, tilt: float | None = None) -> np.ndarray:
        """Returns the target to send: filtered (enforce) or the input unchanged (monitor). Stats count both."""
        if self.mode == "off":
            return u
        raw = np.asarray(u, float)
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        L, c, st = self.L, self.cfg, self.stats
        if tilt is not None:
            self.observe_tilt(float(tilt), now)
        x = self._stop_target(now) if self._stop is not None else raw.copy()
        # 1. position
        lo, hi = L.lo + c.margin, L.hi - c.margin
        y = np.clip(x, lo, hi)
        st.clamp_pos += bool(np.any(y != x))
        # 2. rate
        if self.prev is not None:
            dmax = (L.qd_max if c.rate_max is None else c.rate_max) * dt
            z = np.clip(y, self.prev - dmax, self.prev + dmax)
            st.clamp_rate += bool(np.any(z != y))
            y = z
        # 3. torque (affine PD servo)
        kp = np.maximum(L.kp, 1e-9)
        tlo, thi = q + (L.kd * qd - L.effort) / kp, q + (L.kd * qd + L.effort) / kp
        z = np.clip(y, tlo, thi)
        st.clamp_torque += bool(np.any(z != y))
        y = z
        # 4. velocity guard
        fast = np.abs(qd) > c.qd_factor * L.qd_max
        if fast.any():
            st.vel_guard += 1
            y = np.where(fast, q, y)
        st.ticks += 1
        diff = float(np.max(np.abs(y - raw))) if len(raw) else 0.0
        st.modified += bool(diff > 1e-12)
        st.max_abs_change = max(st.max_abs_change, diff)
        out = y if self.mode == "enforce" else raw
        self.prev = np.asarray(out, float).copy()
        return out

    def summary(self) -> dict:
        d = asdict(self.stats)
        d["events"] = d["events"][:20]
        return dict(version=SAFETY_VERSION, mode=self.mode, cfg=self.cfg.to_dict(), limit_source=self.L.source, **d)


def legged_limits(binding, body: str | None = None) -> SafetyLimits:
    """Limits for a legged body's policy joints from the compiled model: joint range intersected with the actuator
    ctrlrange, PD gains, peak torque (SOURCED where rrp.physics.actuator has a source, else the model forcerange),
    max joint speed (SOURCED URDF velocity, else the labelled VMAX estimate)."""
    import re
    from rrp.bodies.actuator import SOURCED, vmax_for
    b, m = binding, binding.model
    j = m.actuator_trnid[b.pol_act, 0]
    lo, hi = np.maximum(b.jlo, b.lo), np.minimum(b.jhi, b.hi)
    eff = b.effort.copy().astype(float)
    qd_max = np.full(b.n, vmax_for(b.kind, body or ""), float)
    src = "model+vmax_estimate"
    if body in SOURCED:
        jn = [m.joint(int(x)).name[len(b.prefix):] for x in j]
        for k, n in enumerate(jn):
            hit = next(((e, v) for pat, e, v in SOURCED[body] if re.search(pat, n)), None)
            if hit is not None:
                eff[k] = min(eff[k], hit[0])
                qd_max[k] = hit[1]
        src = "sourced_urdf"
    kp = m.actuator_gainprm[b.pol_act, 0].astype(float).copy()
    kd = -m.actuator_biasprm[b.pol_act, 2].astype(float).copy()
    return SafetyLimits(lo=lo.astype(float), hi=hi.astype(float), effort=eff, qd_max=qd_max, kp=kp, kd=kd,
                        hold=np.clip(b.q0, lo, hi).astype(float), tilt_limit=float(b.tilt_limit), source=src)
