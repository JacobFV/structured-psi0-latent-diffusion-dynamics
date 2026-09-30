"""The one relgen scheduler (D-144, docs/relations.md 5.5): progressive composition (1 -> 2 -> k -> full world),
responsive shares, concise steering, exact replay.

Foundation skeleton (F4): the state / config / steer grammar / replay contract are final; the decision policy here is
a placeholder (uniform shares over non-dropped factors at level 1, full-world floor, boosts, pins, freezes, drops),
which fanout unit R11 replaces with the competence-driven policy (signals, hysteresis, promotion, drop-back,
interference) without changing these interfaces.

Replay contract: decisions depend only on (SchedulerConfig, seed, steer log, metrics log); `Scheduler.replay`
reproduces the exact sampling sequence.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np


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

    # -- decisions (placeholder policy; R11)
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
        w = {f: 1.0 for f in live}
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

    def export(self, path) -> None:
        with open(path, "a") as fh:
            fh.write(json.dumps(asdict(self.history[-1]), sort_keys=True, default=list) + "\n")

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
