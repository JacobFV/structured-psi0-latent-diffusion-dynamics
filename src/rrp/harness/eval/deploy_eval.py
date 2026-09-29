"""Deployment-credibility evaluation options for the legged closed loop (D-126 roadmap #27, #29, #30, #32).

`DeployOptions` bundles every option; the default instance is exactly the pre-D-126 evaluation (nothing is
constructed, rows are unchanged). Non-default options are written into each row under `deploy` (with versions), so a
result always states how it was produced.

  base_state_source   truth_noise (default) | estimator            -> rrp.envs.state_estimator (#27)
  packet_ood          off (default) | monitor | enforce            -> rrp.controllers.packet_ood (#29)
    ood_model         path of a fitted detector (<name>.json/.npz); required unless off
    ood_fallback      hold_default (default) | hold_measured | safe_stop (needs safety=enforce)
  safety              off (default) | monitor | enforce            -> rrp.controllers.safety (#30)
  eval_mode           standard (default) | long                    -> minutes-long runs with drift metrics (#32)
    long_s, window_s  duration (s; replaces max_s) and the time-series window (s)
  measure_latency     False (default) | True                       -> per-call timings of system i / system 0 (#32)
  record_packets      False (default) | True                       -> the packets' z kept for OOD fitting / scoring

Everything here is sensing / filtering / bookkeeping around the existing controllers: no model is changed.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

DEPLOY_OPTIONS_VERSION = "deploy-1"
EVAL_MODES = ("standard", "long")


@dataclass
class DeployOptions:
    base_state_source: str = "truth_noise"
    packet_ood: str = "off"
    ood_model: str | None = None
    ood_fallback: str = "hold_default"
    safety: str = "off"
    safety_cfg: dict = field(default_factory=dict)
    eval_mode: str = "standard"
    long_s: float = 180.0
    window_s: float = 10.0
    measure_latency: bool = False
    record_packets: bool = False

    def __post_init__(self):
        from rrp.policies.packet_ood import OOD_FALLBACKS, OOD_MODES
        from rrp.policies.safety import SAFETY_MODES
        from rrp.envs.mujoco.state_estimator import BASE_STATE_SOURCES
        for name, val, ok in (("base_state_source", self.base_state_source, BASE_STATE_SOURCES),
                              ("packet_ood", self.packet_ood, OOD_MODES), ("ood_fallback", self.ood_fallback, OOD_FALLBACKS),
                              ("safety", self.safety, SAFETY_MODES), ("eval_mode", self.eval_mode, EVAL_MODES)):
            if val not in ok:
                raise ValueError(f"{name}={val!r} not in {ok}")
        if self.packet_ood != "off" and not self.ood_model:
            raise ValueError("packet_ood monitor/enforce needs ood_model")
        if self.ood_fallback == "safe_stop" and self.safety != "enforce":
            raise ValueError("ood_fallback=safe_stop needs safety=enforce (the safe stop is the safety layer's)")

    def is_default(self) -> bool:
        return self == DeployOptions()

    def record(self) -> dict:
        """What goes into each row (only when non-default)."""
        from rrp.policies.packet_ood import OOD_VERSION
        from rrp.policies.safety import SAFETY_VERSION
        from rrp.envs.mujoco.state_estimator import ESTIMATOR_VERSION
        d = asdict(self)
        d["version"] = DEPLOY_OPTIONS_VERSION
        d["component_versions"] = dict(estimator=ESTIMATOR_VERSION if self.base_state_source == "estimator" else None,
                                       packet_ood=OOD_VERSION if self.packet_ood != "off" else None,
                                       safety=SAFETY_VERSION if self.safety != "off" else None)
        return d

    @classmethod
    def from_args(cls, a) -> "DeployOptions":
        import json
        return cls(base_state_source=a.base_state_source, packet_ood=a.packet_ood, ood_model=a.ood_model,
                   ood_fallback=a.ood_fallback, safety=a.safety,
                   safety_cfg=json.loads(a.safety_cfg) if a.safety_cfg else {}, eval_mode=a.eval_mode,
                   long_s=a.long_s, window_s=a.window_s, measure_latency=a.measure_latency,
                   record_packets=bool(a.record_packets))


def add_deploy_args(ap):
    """argparse flags (all default to the pre-D-126 behaviour)."""
    g = ap.add_argument_group("deployment credibility (D-126; defaults = unchanged evaluation)")
    g.add_argument("--base-state-source", default="truth_noise", choices=["truth_noise", "estimator"])
    g.add_argument("--packet-ood", default="off", choices=["off", "monitor", "enforce"])
    g.add_argument("--ood-model", default=None, help="fitted detector path (<name>.json + .npz)")
    g.add_argument("--ood-fallback", default="hold_default", choices=["hold_default", "hold_measured", "safe_stop"])
    g.add_argument("--safety", default="off", choices=["off", "monitor", "enforce"])
    g.add_argument("--safety-cfg", default=None, help='JSON overrides of SafetyConfig, e.g. {"rate_max": 8}')
    g.add_argument("--eval-mode", default="standard", choices=["standard", "long"])
    g.add_argument("--long-s", type=float, default=180.0, help="long mode: episode duration (s)")
    g.add_argument("--window-s", type=float, default=10.0, help="long mode: time-series window (s)")
    g.add_argument("--measure-latency", action="store_true")
    g.add_argument("--record-packets", default=None, help="directory: save each episode's packets z (.npz)")
    return ap


# ---------------------------------------------------------------------------------------------- safety wrapper
class SafeTracker:
    """Wraps whatever drives the joints inside LeggedSession (body tracker, BC adapter or system-0 adapter): the
    inner target passes through the safety layer, from MEASURED joint state and the IMU tilt (public)."""

    def __init__(self, inner, session, layer, dt: float = 0.02):
        self.inner, self.s, self.layer, self.dt = inner, session, layer, dt

    def __getattr__(self, k):                      # version, sha256, stats, log, ... of the inner controller
        return getattr(self.inner, k)

    def reset(self, *a, **k):
        self.layer.reset()
        return self.inner.reset(*a, **k)

    def state(self):
        return self.inner.state()

    def load(self, st):
        return self.inner.load(st)

    def act(self, data, cmd):
        from rrp.envs.mujoco.legged_core import quat_rotate_inv
        b = self.s.binding
        u = self.inner.act(data, cmd)
        quat, _ = b.imu(data)                      # == the IMU framequat (tests/unit/test_legged.py)
        g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
        tilt = math.acos(max(-1.0, min(1.0, -g[2])))
        return self.layer.filter(u, data.qpos[b.pol_qadr], data.qvel[b.pol_dadr], float(data.time), self.dt, tilt=tilt)


def make_safety(session, opts: DeployOptions):
    from rrp.policies.safety import SafetyConfig, SafetyLayer, legged_limits
    if opts.safety == "off":
        return None
    return SafetyLayer(legged_limits(session.binding, session.body_key), SafetyConfig(**opts.safety_cfg), opts.safety)


# ---------------------------------------------------------------------------------------------- latency
class Timer:
    def __init__(self):
        self.t: list[float] = []

    def wrap(self, fn):
        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t.append(time.perf_counter() - t0)
        return w

    def summary(self, deadline_s: float | None = None) -> dict | None:
        if not self.t:
            return None
        a = np.array(self.t) * 1000
        d = dict(n=len(a), p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)),
                 p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()), mean_ms=float(a.mean()))
        if deadline_s is not None:
            d.update(deadline_ms=deadline_s * 1000, misses=int((a > deadline_s * 1000).sum()))
        return d


def instrument_latency(ctl, adapter) -> dict:
    """Wrap system i (ctl.generate, once per 0.4 s replan) and the system-0 tick (adapter.act, every 20 ms; it
    includes the replan call on replan ticks). Timings are wall clock in the evaluation process (CPU lease,
    physics paused during inference): compute latencies, not a real-time claim."""
    timers = dict(system_i=Timer(), tick=Timer())
    if ctl is not None and hasattr(ctl, "generate"):
        ctl.generate = timers["system_i"].wrap(ctl.generate)
    adapter.act = timers["tick"].wrap(adapter.act)
    return timers


def latency_summary(timers: dict) -> dict:
    import os
    return dict(system_i=timers["system_i"].summary(deadline_s=0.4), tick=timers["tick"].summary(deadline_s=0.02),
                note="wall-clock compute latency in the eval process (physics paused); tick includes the replan "
                     "call on replan ticks", threads=os.environ.get("OMP_NUM_THREADS"))


# ---------------------------------------------------------------------------------------------- long runs
class LongRunRecorder:
    """Per-window time series and drift metrics for minutes-long runs. Reads privileged truth for SCORING only
    (base pose / velocity); nothing here feeds a controller."""

    def __init__(self, session, window_s: float = 10.0):
        self.s, self.w = session, float(window_s)
        self.samples: list[dict] = []
        self.done_t = None             # first time the task completed (for post-task drift)

    def on_step(self):
        s = self.s
        b = s.binding
        q, v = s.data.qpos, s.data.qvel
        from rrp.envs.mujoco.legged_core import yaw_of
        rec = dict(t=float(s.data.time), x=float(q[b.qa]), y=float(q[b.qa + 1]), z=float(q[b.qa + 2]),
                   yaw=yaw_of(q[b.qa + 3:b.qa + 7]), vxy=float(np.hypot(v[b.da], v[b.da + 1])),
                   tilt=float(b.tilt(s.data)), speed_est=float(s.speed_est), fell=bool(s.fell))
        if getattr(s, "base_estimator", None) is not None:
            e = s.base_estimator
            rec.update(est_vx=float(e.v_w[0]), est_vy=float(e.v_w[1]), est_px=float(e.p_w[0]), est_py=float(e.p_w[1]),
                       true_vx=float(v[b.da]), true_vy=float(v[b.da + 1]))
        if self.done_t is None and s.runtime.succeeded():
            self.done_t = rec["t"]
        self.samples.append(rec)

    def summary(self) -> dict:
        S = self.samples
        if not S:
            return dict(windows=[])
        t = np.array([r["t"] for r in S])
        wins = []
        for k in range(int(math.ceil((t[-1] - t[0] + 1e-9) / self.w))):
            sel = [r for r in S if t[0] + k * self.w <= r["t"] < t[0] + (k + 1) * self.w]
            if len(sel) < 2:
                continue
            a, z = sel[0], sel[-1]
            w = dict(t0=a["t"], t1=z["t"], disp_m=float(np.hypot(z["x"] - a["x"], z["y"] - a["y"])),
                     path_m=float(sum(np.hypot(q["x"] - p["x"], q["y"] - p["y"]) for p, q in zip(sel, sel[1:]))),
                     yaw_change=float((z["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi),
                     mean_speed_true=float(np.mean([r["vxy"] for r in sel])),
                     mean_speed_est=float(np.nanmean([r["speed_est"] for r in sel])) if any(
                         np.isfinite(r["speed_est"]) for r in sel) else None,
                     height_mean=float(np.mean([r["z"] for r in sel])), tilt_max=float(max(r["tilt"] for r in sel)),
                     fell=any(r["fell"] for r in sel))
            if "est_vx" in a:
                ev = np.array([[r["est_vx"] - r["true_vx"], r["est_vy"] - r["true_vy"]] for r in sel])
                w["est_vel_rmse"] = float(np.sqrt((ev ** 2).sum(1).mean()))
                w["est_pos_err_end"] = float(np.hypot(z["est_px"] - (z["x"] - S[0]["x"]),
                                                      z["est_py"] - (z["y"] - S[0]["y"])))
            wins.append(w)
        out = dict(window_s=self.w, windows=wins, duration_s=float(t[-1] - t[0]), task_done_t=self.done_t)
        if self.done_t is not None:                 # post-task drift: how far the robot wanders after the halt
            post = [r for r in S if r["t"] >= self.done_t]
            if len(post) >= 2:
                out["post_task_drift_m"] = float(np.hypot(post[-1]["x"] - post[0]["x"], post[-1]["y"] - post[0]["y"]))
                out["post_task_yaw_drift"] = float((post[-1]["yaw"] - post[0]["yaw"] + math.pi) % (2 * math.pi) - math.pi)
                out["post_task_s"] = float(post[-1]["t"] - post[0]["t"])
        if "est_vx" in S[0]:
            out["est_pos_err_final"] = wins[-1].get("est_pos_err_end") if wins else None
        return out
