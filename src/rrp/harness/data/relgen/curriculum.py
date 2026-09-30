"""The one relgen scheduler (D-144, docs/relations.md 5.5; fanout unit R11): progressive composition
(1 -> 2 -> k -> full world), responsive shares, concise steering, exact replay.

Foundation skeleton (F4) contract, kept unchanged by this unit: the `SchedulerConfig` / `ScheduleState` / `SteerOp`
dataclass shapes, the steer grammar (`parse_steer`, `STEER_OPS`), `Scheduler.steer` / `.validate` / `._apply_steers`
and the replay contract (decisions depend only on `(SchedulerConfig, seed, steer log, metrics log)`;
`Scheduler.replay` reproduces the exact sampling sequence).

R11 fills in the decision policy that was a uniform-share placeholder:
- **signals** (`Scheduler._level_and_reasons`): per factor, EMA-smoothed (`cfg.ema`) competence / plateau /
  interference / attributed-failure rates, recomputed fresh from `self.metrics_log` at every `decide()` call (never
  cached on `self`) so the policy stays a pure function of its inputs, per the replay contract above. A factor with
  no observations yet reports unseen and falls back to the F4 placeholder's neutral baseline.
- **struggle score** `s_f = (1 - c_f) + 0.5*p_f + i_f + e_f` (docs/relations.md 5.5) drives both the promotion /
  demotion / drop-back walk and the responsive share weight `share_min + s_f`.
- **promotion**: level k -> k+1 after 2 consecutive intervals with competence >= its threshold (`cfg.promote`,
  default `_DEFAULT_PROMOTE_THRESHOLD`); **demotion**: k -> k-1 the first interval competence drops below
  threshold - hysteresis; **drop-back**: `_DROPBACK_INTERVALS` consecutive intervals of high struggle + positive
  interference drop the level by one and CAP it there -- promotion is skipped entirely, not just delayed -- until
  BOTH `_DROPBACK_COOLDOWN_INTERVALS` have elapsed AND the struggle score has actually recovered under threshold.
  A bare timer was tried first and produces genuine oscillation when interference stays high while competence stays
  high too (see `archived research/tracks/rel-r11.md`); recovery-gating is what makes "hysteresis prevents oscillation"
  (5.5) literally true.
- **responsive mixing**: shares proportional to `share_min + s_f` (unseen factors default to `s_f = 1.0`, the old
  placeholder's flat weight, so a scheduler with no `observe()` calls behaves exactly as F4's tests expect), clipped
  to `[share_min, share_max]`, renormalized to the non-full-world budget, then rate-limited to at most
  `cfg.max_step_change` absolute change from the previous decision's share.
- **interference** (`interference`, `max_interference`, `interfering_pairs`, module-level, not on `Scheduler`):
  `interference(f, g) = metric_f(sets with f, without g) - metric_f(sets with f and g)` over caller-supplied eval
  records at matched composition depth; a caller feeds `max_interference(...)` per factor into `observe()`'s
  `"interference"` key so it participates in the struggle score above.
- **`compose`** (in `rrp.harness.data.relgen.__init__`, not this module) is the sibling half of this unit: minimal
  part cover of a desired active set, transitive requires-closure, conflict rejection.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

_DEFAULT_PROMOTE_THRESHOLD = 0.8      # docs/relations.md 5.5 example: "competence 0.61 < 0.8"
_DROPBACK_STRUGGLE = 1.2              # struggle score above which a factor is considered to be struggling
_DROPBACK_INTERVALS = 3               # consecutive struggling intervals with rising interference before drop-back
_DROPBACK_COOLDOWN_INTERVALS = 3      # minimum hold after a drop-back; it also requires struggle to have recovered


@dataclass(frozen=True)
class SchedulerConfig:
    factors: tuple = ()
    parts: tuple = ()
    interval: int = 500
    promote: tuple = ()                     # ((factor, threshold), ...)
    ema: float = 0.9
    hysteresis: float = 0.05
    share_min: float = 0.02
    share_max: float = 0.35
    full_world_start: float = 0.1           # full-world floor ramps linearly start -> end over ramp_steps,
    full_world_end: float = 0.8             # and never decreases once reached
    ramp_steps: int = 100_000
    replay_min: float = 0.1
    boost_gain: float = 1.5
    max_step_change: float = 0.25

    def full_world_floor(self, step: int) -> float:
        f = min(1.0, step / max(1, self.ramp_steps))
        return self.full_world_start + f * (self.full_world_end - self.full_world_start)


@dataclass
class ScheduleState:
    step: int = 0
    level: dict = field(default_factory=dict)
    share: dict = field(default_factory=dict)
    full_world: float = 0.0
    replay: dict = field(default_factory=dict)
    frozen: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    pins: dict = field(default_factory=dict)
    boosts: list = field(default_factory=list)          # [factor, gain, until_step]
    parts: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


# ------------------------------------------------------------------ steer grammar
STEER_OPS = ("boost", "pin", "unpin", "freeze", "unfreeze", "drop", "restore", "set", "add_part", "revert")


@dataclass(frozen=True)
class SteerOp:
    op: str
    factor: str | None = None
    value: float | None = None              # boost gain / pin level / set value / revert count
    steps: int | None = None                # boost duration / revert-to step
    key: str | None = None                  # set: config field (or "full_world")
    cmp: str = "="                          # set: "=" | ">=" | "<="
    part: str | None = None
    author: str = ""
    reason: str = ""
    at: int | None = None                   # step at which it was applied (filled by Scheduler.steer)
    wall: float | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v not in (None, "")}, sort_keys=True)


_PAT = [
    (r"boost (\S+) x([0-9.]+) for (\d+)$", lambda m: SteerOp("boost", m[1], float(m[2]), int(m[3]))),
    (r"pin (\S+) k=(\d+)$", lambda m: SteerOp("pin", m[1], float(m[2]))),
    (r"(unpin|freeze|unfreeze|drop|restore) (\S+)$", lambda m: SteerOp(m[1], m[2])),
    (r"set (\w+)\s*(>=|<=|=)\s*([0-9.eE+-]+)$", lambda m: SteerOp("set", key=m[1], cmp=m[2], value=float(m[3]))),
    (r"add part (\S+)$", lambda m: SteerOp("add_part", part=m[1])),
    (r"revert (\d+)$", lambda m: SteerOp("revert", value=float(m[1]))),
    (r"revert to (\d+)$", lambda m: SteerOp("revert", steps=int(m[1]))),
]


def parse_steer(line: str | dict, author: str = "", reason: str = "") -> SteerOp:
    """One steer from its CLI form ("boost ix.support x2 for 5000") or JSON form ({"op": "boost", ...})."""
    if isinstance(line, dict):
        unknown = set(line) - {f.name for f in fields(SteerOp)}
        if unknown or line.get("op") not in STEER_OPS:
            raise ValueError(f"invalid steer {line}")
        return SteerOp(**line)
    s = line.strip()
    if s.startswith("{"):
        return parse_steer(json.loads(s))
    for pat, mk in _PAT:
        m = re.match(pat, s)
        if m:
            return replace(mk(m), author=author, reason=reason)
    raise ValueError(f"cannot parse steer {line!r}; grammar: docs/relations.md 5.5")


# ------------------------------------------------------------------ scheduler
class Scheduler:
    def __init__(self, cfg: SchedulerConfig, seed: int = 0):
        self.cfg, self.seed = cfg, int(seed)
        self.state = ScheduleState(level={f: 1 for f in cfg.factors}, parts=list(cfg.parts))
        self.steer_log: list[SteerOp] = []
        self.metrics_log: list[tuple[int, dict]] = []
        self.history: list[ScheduleState] = []

    # -- inputs
    def observe(self, step: int, metrics: dict) -> None:
        self.metrics_log.append((int(step), metrics))

    def validate(self, op: SteerOp) -> str | None:
        known = set(self.cfg.factors)
        if op.op not in STEER_OPS:
            return f"unknown op {op.op}"
        if op.op in ("boost", "pin", "unpin", "freeze", "unfreeze", "drop", "restore") and op.factor not in known:
            return f"unknown factor {op.factor!r}"
        if op.op == "boost" and not (op.value and op.value > 0 and op.steps and op.steps > 0):
            return "boost needs a positive gain and duration"
        if op.op == "set" and op.key != "full_world" and op.key not in {f.name for f in fields(SchedulerConfig)}:
            return f"unknown schedule field {op.key!r}"
        if op.op == "set" and op.key in ("share_min", "share_max", "full_world") and not 0 <= (op.value or 0) <= 1:
            return "shares must be in [0, 1]"
        if op.op == "add_part":
            from rrp.harness.data.relgen import PARTS
            if op.part not in PARTS:
                return f"unknown part {op.part!r} (a new part is code: register it in PARTS first)"
        return None

    def steer(self, op: SteerOp, step: int) -> SteerOp:
        """Validate, stamp and apply at the next decision; invalid ops are logged (with the error) and not applied."""
        op = replace(op, at=int(step), wall=op.wall if op.wall is not None else time.time())
        err = self.validate(op)
        if err:
            op = replace(op, reason=f"REJECTED: {err}; {op.reason}")
        self.steer_log.append(op)
        return op

    # -- decisions (competence-driven policy; R11)
    def _raw_signal(self, factor: str, key: str, lo: int, hi: int) -> float | None:
        """Mean of `metrics[factor][key]` over `self.metrics_log` entries with `lo <= step < hi`, or None if the
        factor was not observed for `key` in that decision window."""
        vals = [m[factor][key] for s, m in self.metrics_log
                if lo <= s < hi and isinstance(m, dict) and factor in m and key in m[factor]]
        return float(np.mean(vals)) if vals else None

    def _level_and_reasons(self, factor: str, step: int) -> tuple[int, dict, list[str], bool]:
        """Walk every decision interval boundary up to `step`, EMA-smoothing this factor's signals and applying
        promotion / demotion / drop-back as it goes (pure function of `self.metrics_log` + `self.cfg`, so replay
        reproduces it exactly). Only the LAST processed window's reasons are returned -- earlier windows' reasons
        were already returned by the `decide()` calls that walked up to them, so replaying the whole trajectory on
        every call (needed for purity: nothing is cached on `self`) must not re-announce old history every time.
        Returns (level, smoothed signals, this window's reasons, ever observed).

        Drop-back (`docs/relations.md` 5.5) caps the level -- promotion is skipped, not just delayed -- until BOTH
        `_DROPBACK_COOLDOWN_INTERVALS` have elapsed AND the struggle score has actually fallen back to/under
        `_DROPBACK_STRUGGLE`; a fixed timer alone would let a still-genuinely-interfering factor promote right back
        up and immediately drop again, oscillating forever. Recovery-gating instead of a bare timer is what makes
        "hysteresis prevents oscillation" (5.5) literally true rather than just slower."""
        cfg = self.cfg
        thr = dict(cfg.promote).get(factor, _DEFAULT_PROMOTE_THRESHOLD)
        max_level = max(1, len(cfg.factors))
        level, consec_hi, consec_struggle = 1, 0, 0
        capped, cap_release_step = False, -1
        ema_c = ema_p = ema_i = ema_e = 0.0
        reasons: list[str] = []
        any_obs = False
        for b in range(0, int(step) + 1, max(1, cfg.interval)):
            lo, hi = b, b + cfg.interval
            c, p, ifr, e = (self._raw_signal(factor, k, lo, hi) for k in ("competence", "plateau", "interference", "failures"))
            if not any(v is not None for v in (c, p, ifr, e)):
                continue          # a gap window with no observations holds the EMA (never decays toward 0)
            reasons = []           # only the LAST observed window's reasons survive to the return
            c, p, ifr, e = (v if v is not None else 0.0 for v in (c, p, ifr, e))
            if not any_obs:
                ema_c, ema_p, ema_i, ema_e = c, p, ifr, e     # first observation initializes the EMA directly
            else:
                ema_c = cfg.ema * ema_c + (1 - cfg.ema) * c
                ema_p = cfg.ema * ema_p + (1 - cfg.ema) * p
                ema_i = cfg.ema * ema_i + (1 - cfg.ema) * ifr
                ema_e = cfg.ema * ema_e + (1 - cfg.ema) * e
            any_obs = True
            struggle = (1 - ema_c) + 0.5 * ema_p + ema_i + ema_e
            if capped and b >= cap_release_step and struggle <= _DROPBACK_STRUGGLE:
                capped = False                    # released: cooldown elapsed AND it actually recovered
            if not capped:
                if ema_c >= thr:
                    consec_hi += 1
                    if consec_hi >= 2 and level < max_level:
                        level += 1
                        consec_hi = 0
                        reasons.append(f"promote {factor}: competence {ema_c:.3f} >= {thr} for 2 intervals -> k={level} (step {b})")
                else:
                    consec_hi = 0
                    if ema_c < thr - cfg.hysteresis and level > 1:
                        level -= 1
                        reasons.append(f"demote {factor}: competence {ema_c:.3f} < {thr - cfg.hysteresis:.3f} -> k={level} (step {b})")
            if not capped and struggle > _DROPBACK_STRUGGLE and ema_i > 0:
                consec_struggle += 1
                if consec_struggle >= _DROPBACK_INTERVALS and level > 1:
                    level -= 1
                    capped, cap_release_step = True, b + _DROPBACK_COOLDOWN_INTERVALS * cfg.interval
                    consec_struggle = 0
                    reasons.append(f"drop-back {factor}: struggle {struggle:.3f}, interference {ema_i:.3f} for "
                                   f"{_DROPBACK_INTERVALS} intervals -> k={level}, held until step {cap_release_step} "
                                   f"or recovery (step {b})")
            else:
                consec_struggle = 0
        struggle = (1 - ema_c) + 0.5 * ema_p + ema_i + ema_e
        signals = {"competence": ema_c, "plateau": ema_p, "interference": ema_i, "failures": ema_e, "struggle": struggle}
        return level, signals, reasons, any_obs

    def _apply_steers(self, st: ScheduleState, cfg: SchedulerConfig, step: int) -> SchedulerConfig:
        for op in self.steer_log:
            if op.at is None or op.at > step or op.reason.startswith("REJECTED"):
                continue
            if op.op == "boost":
                if step < op.at + op.steps:
                    st.boosts.append([op.factor, op.value, op.at + op.steps])
            elif op.op == "pin":
                st.pins[op.factor] = int(op.value)
            elif op.op == "unpin":
                st.pins.pop(op.factor, None)
            elif op.op in ("freeze", "unfreeze"):
                st.frozen = sorted(set(st.frozen) | {op.factor}) if op.op == "freeze" else [f for f in st.frozen if f != op.factor]
            elif op.op in ("drop", "restore"):
                st.dropped = sorted(set(st.dropped) | {op.factor}) if op.op == "drop" else [f for f in st.dropped if f != op.factor]
            elif op.op == "set" and op.key != "full_world":
                cfg = replace(cfg, **{op.key: type(getattr(cfg, op.key))(op.value)})
            elif op.op == "add_part" and op.part not in st.parts:
                st.parts.append(op.part)
        return cfg

    def decide(self, step: int) -> ScheduleState:
        prev = self.history[-1] if self.history else None
        reverts = [op for op in self.steer_log if op.op == "revert" and op.at is not None and op.at <= step
                   and not op.reason.startswith("REJECTED")]
        st = ScheduleState(step=int(step), level=dict(prev.level if prev else {f: 1 for f in self.cfg.factors}),
                           parts=list(self.cfg.parts))
        cfg = self._apply_steers(st, self.cfg, step)
        signals: dict[str, dict] = {}
        reasons: list[str] = []
        for f in cfg.factors:
            if f in st.dropped:
                continue
            level, sig, why, seen = self._level_and_reasons(f, step)
            if seen:
                st.level[f] = level
                signals[f] = sig
                reasons.extend(why)
        st.signals = signals
        st.reasons.extend(reasons)
        for f, k in st.pins.items():
            st.level[f] = k
        fw = cfg.full_world_floor(step)
        if prev is not None:
            fw = max(fw, prev.full_world)                        # the floor never decreases once reached
        for op in self.steer_log:
            if op.op == "set" and op.key == "full_world" and op.at is not None and op.at <= step \
                    and not op.reason.startswith("REJECTED"):
                fw = max(fw, op.value) if op.cmp == ">=" else (min(fw, op.value) if op.cmp == "<=" else op.value)
        st.full_world = float(min(1.0, fw))
        live = [f for f in self.cfg.factors if f not in st.dropped]
        # responsive mixing (docs/relations.md 5.5): weight proportional to share_min + struggle score; a factor with
        # no observations yet uses the F4 placeholder's flat weight (1.0) so an un-observed scheduler is unchanged.
        w = {f: cfg.share_min + signals.get(f, {}).get("struggle", 1.0) for f in live}
        for f, g, until in st.boosts:
            if f in w and step < until:
                w[f] *= g
                st.reasons.append(f"boost {f} x{g} until {until} (steer)")
        tot = sum(w.values()) or 1.0
        budget = 1.0 - st.full_world
        st.share = {f: min(cfg.share_max, max(cfg.share_min, budget * v / tot)) for f, v in w.items()}
        if prev is not None:
            for f in st.frozen:
                if f in prev.share:
                    st.share[f] = prev.share[f]
        s = sum(st.share.values())
        if s > budget and s > 0:
            st.share = {f: v * budget / s for f, v in st.share.items()}
        if prev is not None:
            # rate limit: at most cfg.max_step_change absolute change per decision interval from the previous share
            # (frozen factors already carried their previous share forward above and are left alone here).
            limited = {}
            for f, v in st.share.items():
                if f in st.frozen:
                    limited[f] = v
                    continue
                p0 = prev.share.get(f, v)
                limited[f] = max(p0 - cfg.max_step_change, min(p0 + cfg.max_step_change, v))
            st.share = limited
        if reverts:
            last = reverts[-1]
            target = last.steps if last.steps is not None else None
            if target is not None:
                cand = [h for h in self.history if h.step <= target]
                if cand:
                    keep = replace(cand[-1], step=int(step), reasons=[f"revert to {target}"])
                    st = keep
        self.history.append(st)
        return st

    def sample(self, step: int, n: int) -> list[tuple[frozenset, int]]:
        """(active set, count) for a batch of n relgen rows at `step` (deterministic in (seed, step))."""
        from rrp.harness.data.mix import allocate
        st = self.history[-1] if self.history else self.decide(step)
        rng = np.random.default_rng([self.seed, int(step)])
        alloc = allocate(n, {f: v for f, v in st.share.items()})
        out = []
        for f, c in sorted(alloc.items()):
            if f == "main" or c == 0:
                continue
            k = max(1, min(st.level.get(f, 1), len(self.cfg.factors)))
            others = [g for g in self.cfg.factors if g != f and g not in st.dropped]
            extra = tuple(rng.choice(others, size=min(k - 1, len(others)), replace=False)) if k > 1 and others else ()
            out.append((frozenset((f,) + tuple(extra)), c))
        return out

    def export(self, path, extra: dict | None = None) -> None:
        """Append the last decision to `path` (schedule.jsonl); `extra` keys (steers applied, metrics observed since the
        previous decision) ride on the record for `replay_records`."""
        with open(path, "a") as fh:
            fh.write(json.dumps({**asdict(self.history[-1]), **(extra or {})}, sort_keys=True, default=list) + "\n")

    @classmethod
    def replay_records(cls, cfg: SchedulerConfig, seed: int, records, steps=(), n: int = 0):
        """Rebuild the scheduler from a run's `schedule.jsonl` records (each: the decision `step`, the `steers` applied
        at it, the `metrics` observed before it), feeding each decision only what was known when it was taken, so the
        decisions come out identical to the live run's. With `steps` (and `n`), also returns the `sample(step, n)`
        composition at each of those steps (state = the last decision at or before it): `(scheduler, [sample, ...])`."""
        s = cls(cfg, seed)
        todo = sorted(records, key=lambda r: r["step"])

        def apply_until(t):
            while todo and todo[0]["step"] <= t:
                r = todo.pop(0)
                s.steer_log.extend(parse_steer(o) for o in r.get("steers", []))
                s.metrics_log.extend((int(st), m) for st, m in r.get("metrics", []))
                s.decide(r["step"])
        seq = []
        for t in steps:
            apply_until(t)
            seq.append(s.sample(t, n))
        apply_until(float("inf"))
        return s, seq

    @classmethod
    def replay(cls, cfg: SchedulerConfig, seed: int, steer_log, metrics_log, steps, n: int):
        """Rebuild the scheduler from its inputs and return the sampling sequence at `steps` (list of step ints)."""
        s = cls(cfg, seed)
        s.steer_log = list(steer_log)
        s.metrics_log = list(metrics_log)
        seq = []
        for t in steps:
            if t % cfg.interval == 0 or not s.history:
                s.decide(t)
            seq.append(s.sample(t, n))
        return s, seq


# ------------------------------------------------------------------ interference (docs/relations.md 5.5)
def interference(records: list[dict], f: str, g: str, metric: str = "competence") -> float | None:
    """`interference(f, g) = metric_f(sets with f, without g) - metric_f(sets with f and g)` (docs/relations.md 5.5),
    over caller-supplied eval `records` at matched composition depth / step window. Each record is
    `{"active": <iterable of factor names in that batch's composed active set>, "metric": {factor: value, ...}}`.
    A positive value means g hurts f's metric when co-active; None means one side has no matching records (not
    enough evidence either way). This is a plain function, not a `Scheduler` method: a caller (the eval loop) feeds
    its result into `Scheduler.observe(step, {f: {"interference": ...}})`, which is what feeds `i_f` above."""
    with_g = [r["metric"][f] for r in records if f in r["active"] and g in r["active"] and f in r.get("metric", {})]
    without_g = [r["metric"][f] for r in records if f in r["active"] and g not in r["active"] and f in r.get("metric", {})]
    if not with_g or not without_g:
        return None
    return float(np.mean(without_g) - np.mean(with_g))


def max_interference(records: list[dict], factors, metric: str = "competence") -> dict[str, float]:
    """`i_f = max_g interference(f, g)` per factor (0.0 for a factor with no comparable pairs), ready to feed
    `Scheduler.observe`."""
    out = {}
    for f in factors:
        vals = [v for g in factors if g != f for v in [interference(records, f, g, metric)] if v is not None]
        out[f] = max(vals) if vals else 0.0
    return out


def interfering_pairs(records: list[dict], factors, metric: str = "competence", threshold: float = 0.1) -> list[tuple[str, str, float]]:
    """Every `(f, g, interference(f, g))` above `threshold`, most-interfering first (`rrp suite
    relations-curriculum`'s "interfering pairs" report)."""
    out = []
    for f in factors:
        for g in factors:
            if f == g:
                continue
            v = interference(records, f, g, metric)
            if v is not None and v > threshold:
                out.append((f, g, v))
    return sorted(out, key=lambda t: -t[2])


def estimate_competence(metrics: dict, mae_ref: dict, factors) -> dict:
    """`estimates_loss` metrics -> `{factor: {"competence": c}}` for the `factors` the scheduler tracks, every readout
    kind giving one number in [0, 1]: a `<f>_acc` (hits, n) pair is the hit rate; a `<f>_mae` (abs-error sum, n) pair of a
    field factor (no hit rate exists) is the fraction of its initial mean absolute error removed,
    `clip(1 - mae / mae_ref[f], 0, 1)` (`mae_ref`: the factor's MAE at its first observation; a factor without one is
    not reported yet). Pairs with n == 0 (the label was absent from the batch) are skipped."""
    out = {}
    for kind in ("_mae", "_acc"):                                   # a factor that reports both: the hit rate wins
        for k, (v, n) in metrics.items():
            f = k[:-len(kind)]
            if n <= 0 or not k.endswith(kind) or f not in factors:
                continue
            if kind == "_acc":
                out[f] = {"competence": float(v) / n}
            elif mae_ref.get(f):
                out[f] = {"competence": float(min(1.0, max(0.0, 1.0 - (float(v) / n) / mae_ref[f])))}
    return out
