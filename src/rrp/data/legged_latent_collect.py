"""Tick-level (50 Hz) legged teacher data for the latent-packet path (source label: scripted_teacher driving the
frozen body tracker; the tracker is the native-level expert whose joint targets system 0 learns to realize).

Per tick we store PUBLIC inputs (joint encoders, IMU, touch, system-i global context from localization/detector/
task view at the last 10 Hz observation, osc-v1 phase), the expert NATIVE label (clean tracker joint target in
action units (target - q0) / action_scale), and PRIVILEGED truth used only as probe labels / evaluation truth
(true base pose, true foot contacts, fall, true waypoint positions).

DART: with per-episode sigma, EXECUTED targets = expert target + N(0, sigma * action_scale) (clean label kept), so
system 0 sees off-nominal states with corrective labels. The teacher's speed/turn gains are randomized per episode
(declared diversity; still the scripted teacher).

usage: python -m rrp.data.legged_latent_collect --body go2 --seeds 0-399 --out artifacts/datasets/legged_latent_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rrp.control.legged_latent import LeggedMorph, public_context, local_state, active_event, TICK_DT
from rrp.control.legged_teachers import WaypointTeacher
from rrp.sim.legged import LeggedSession, build_waypoint_contact


class RecordingTracker:
    """Wraps the frozen body tracker: records (public local state, clean expert target) every native tick and
    executes the target with optional DART noise."""

    def __init__(self, session, morph, sigma: float, rng):
        self.s, self.inner, self.m = session, session.tracker, morph
        self.sigma, self.rng = sigma, rng
        self.rec = None
        self.ticks = 0
        self.ctx = None
        for k in ("source", "version"):
            setattr(self, k, getattr(self.inner, k))

    def reset(self, phase=0.0):
        self.ticks = 0
        self.inner.reset(phase)

    def state(self):
        return self.inner.state()

    def load(self, st):
        self.inner.load(st)

    def act(self, data, cmd):
        tgt = self.inner.act(data, cmd)
        b = self.s.binding
        osc = (self.ticks * TICK_DT / self.m.gait_period) % 1.0
        if self.rec is not None:
            q, qd, imu, touch = local_state(self.s, self.m)
            fc, bad = b.contacts(data)
            self.rec["q"].append(q); self.rec["qd"].append(qd); self.rec["imu"].append(imu)
            self.rec["touch"].append(touch); self.rec["osc"].append(osc)
            self.rec["a"].append(((tgt - b.q0) / b.action_scale).astype(np.float32))
            self.rec["ctx"].append(self.ctx)
            self.rec["ev"].append(self.ev)
            self.rec["pose"].append(self.s.base_pose_truth().astype(np.float32))
            self.rec["contact"].append(fc.copy())
            self.rec["bad"].append(bool(bad))
            self.rec["height"].append(float(data.qpos[b.qa + 2]))
            self.rec["cmd"].append(np.asarray(cmd, np.float32))
        self.ticks += 1
        if self.sigma > 0:
            tgt = np.clip(tgt + self.rng.normal(0, self.sigma * b.action_scale, len(tgt)), b.lo, b.hi)
        return tgt


def collect_episode(body: str, seed: int, sigma: float, tracker_kind="auto", max_steps=1300, arc_only=False):
    rng = np.random.default_rng([seed, 77])
    sc = build_waypoint_contact(body, seed)
    s = LeggedSession(sc, tracker_kind=tracker_kind, seed=seed)
    morph = LeggedMorph(s.model, s.binding, sc.robots[0].robot_spec.spec_hash)
    rt = RecordingTracker(s, morph, sigma, rng)
    s.tracker = rt
    s.reset()
    te = WaypointTeacher(s, speed_frac=float(rng.uniform(0.4, 0.9)), turn_gain=float(rng.uniform(1.0, 2.2)),
                         arc_only=arc_only)
    keys = ("q", "qd", "imu", "touch", "osc", "a", "ctx", "ev", "pose", "contact", "bad", "height", "cmd")
    rt.rec = {k: [] for k in keys}
    t0 = time.time()
    steps = 0
    for _ in range(max_steps):
        cmd = te.act()
        rt.ctx = public_context(s, (rt.ticks * TICK_DT / morph.gait_period) % 1.0)
        rt.ev = active_event(s.runtime)
        s.step(cmd)
        steps += 1
        if te.done or s.fell:
            break
    # 1 s of post-halt standing (teacher keeps zero command) so the halt/stand regime is represented
    if not s.fell:
        for _ in range(10):
            rt.ctx = public_context(s, (rt.ticks * TICK_DT / morph.gait_period) % 1.0)
            rt.ev = active_event(s.runtime)
            s.step(te.act())
            if s.fell:
                break
    status = "success" if s.privileged_success() and not s.fell else ("fell" if s.fell else "failure")
    arr = {k: np.asarray(v) for k, v in rt.rec.items()}
    meta = dict(body=body, seed=seed, sigma=sigma, status=status, steps=steps, ticks=len(arr["a"]),
                tracker_source=rt.source, tracker_version=rt.version, source="scripted_teacher",
                privileged_teacher=True, teacher_variant="arc_only" if arc_only else "default",
                speed_frac=te.vmax / te.r["vx"][1], turn_gain=te.k, waypoints=sc.meta["waypoints"],
                spec_hash=morph.spec_hash, wall_s=time.time() - t0)
    return arr, meta, morph


def _seeds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--tracker", default="auto")
    ap.add_argument("--sigmas", default="0,0.1,0.2,0.3", help="DART sigma cycled over seeds (action units)")
    ap.add_argument("--arc-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    sig = [float(x) for x in a.sigmas.split(",")]
    out = Path(a.out) / a.body
    out.mkdir(parents=True, exist_ok=True)
    seeds = _seeds(a.seeds)
    shard = a.shard or f"s{seeds[0]}-{seeds[-1]}"
    f = out / f"{shard}.npz"
    if f.exists():
        print(f"exists {f}")
        return
    eps, metas, morph = [], [], None
    for i, sd in enumerate(seeds):
        arr, meta, morph = collect_episode(a.body, sd, sig[i % len(sig)], a.tracker, arc_only=a.arc_only)
        eps.append(arr)
        metas.append(meta)
        print(json.dumps({k: meta[k] for k in ("seed", "sigma", "status", "steps", "ticks")}), flush=True)
    cat = {k: np.concatenate([e[k] for e in eps]) for k in eps[0]}
    cat["ep"] = np.concatenate([np.full(len(e["a"]), i) for i, e in enumerate(eps)])
    cat["t"] = np.concatenate([np.arange(len(e["a"])) for e in eps])
    np.savez_compressed(f, **cat, node_static=morph.node_static, node_asm=morph.node_asm,
                        asm_static=morph.asm_static, q0_all=morph.q0_all, n_policy=morph.n_policy, nf=morph.nf)
    f.with_suffix(".json").write_text(json.dumps(dict(body=a.body, episodes=metas, handles=morph.handles,
                                                      asm_kind=morph.asm_kind, body_kind=morph.body_kind,
                                                      gait_period=morph.gait_period), indent=1))
    n_ok = sum(m["status"] == "success" for m in metas)
    print(json.dumps(dict(body=a.body, shard=shard, n=len(metas), success=n_ok, ticks=int(len(cat["a"])))))


if __name__ == "__main__":
    main()
