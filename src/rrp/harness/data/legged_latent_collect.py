"""Tick-level (50 Hz) legged teacher data for the latent-packet path (source label: scripted_teacher driving the
frozen body tracker; the tracker is the native-level expert whose joint targets system 0 learns to realize).
`--task` (default waypoint_contact) is any registered mujoco/legged task with a scripted teacher; its env, teacher,
event slots and target entities come from the task registry (`legged_collect.open_episode`, features.legged.TaskView).
One output directory holds one task; the manifest records task, teacher and tracker sha; the sealed-split guard runs
before collection.

Per tick we store PUBLIC inputs (joint encoders, IMU, touch, system-i global context from localization/detector/
task view at the last 10 Hz observation, osc-v1 phase), the expert NATIVE label (clean tracker joint target in
action units (target - q0) / action_scale), and PRIVILEGED truth used only as probe labels / evaluation truth
(true base pose, true foot contacts, fall, true waypoint positions).

DART: with per-episode sigma, EXECUTED targets = expert target + N(0, sigma * action_scale) (clean label kept), so
system 0 sees off-nominal states with corrective labels. The teacher's speed/turn gains are randomized per episode
(declared diversity; still the scripted teacher).

usage: python -m rrp.cli data legged-latent-collect --body go2 --seeds 0-399 --out artifacts/datasets/legged_latent_v1 [--task T]
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from rrp.policies.features.legged import LeggedMorph, public_context, local_state, active_event, task_view_of, TICK_DT
from rrp.harness.data.legged_collect import (DEFAULT_MAX_STEPS, assert_one_task, guard_sealed, make_teacher, open_episode,
                                             outcome, teacher_done)
from rrp.core.provenance import CONTACT_VERSION_DEFAULT, parse_source, physics_provenance
from rrp.harness.data.manifest import dataset_provenance, write_manifest
from rrp.core.runs import parse_seed_spec


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


def collect_episode(body: str, seed: int, sigma: float, tracker_kind="auto", max_steps=None, arc_only=False,
                    task: str = "waypoint_contact", tracker_id: str | None = None):
    rng = np.random.default_rng([seed, 77])
    spec, s = open_episode(task, body, seed, tracker_kind=tracker_kind, tracker_id=tracker_id)
    sc = s.scenario
    max_steps = max_steps or spec.max_steps or DEFAULT_MAX_STEPS
    # W8/D-112: the W6 motion-quality recorder (read-only; install_legged with the nominal perturbation performs the
    # original tick operations in the same order), so every episode meta carries slip_ratio/cot/... for the dataset gate
    from rrp.envs.mujoco.perturb import PhysicsPerturbation, install_legged
    from rrp.envs.mujoco.motion_quality import LeggedMotionRecorder
    mrec = LeggedMotionRecorder(s)
    install_legged(s, PhysicsPerturbation(), seed, on_substep=mrec.on_substep, on_tick=mrec.on_tick,
                   on_reset=mrec.on_reset)
    morph = LeggedMorph(s.model, s.binding, sc.robots[0].robot_spec.spec_hash)
    rt = RecordingTracker(s, morph, sigma, rng)
    s.tracker = rt
    s.reset()
    # declared teacher diversity (drawn after the reset, as always; a teacher without these options ignores them)
    pol, te = make_teacher(spec, s, dict(speed_frac=float(rng.uniform(0.4, 0.9)), turn_gain=float(rng.uniform(1.0, 2.2))),
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
        if teacher_done(te, s) or s.fell:
            break
    # 1 s of post-halt standing (teacher keeps zero command) so the halt/stand regime is represented
    if not s.fell:
        for _ in range(10):
            rt.ctx = public_context(s, (rt.ticks * TICK_DT / morph.gait_period) % 1.0)
            rt.ev = active_event(s.runtime)
            s.step(te.act())
            if s.fell:
                break
    status, reason = outcome(spec, s)
    arr = {k: np.asarray(v) for k, v in rt.rec.items()}
    meta = dict(body=body, seed=seed, sigma=sigma, task=task, status=status, failure_reason=reason, steps=steps,
                max_steps=max_steps, ticks=len(arr["a"]),
                tracker_source=rt.source, tracker_version=rt.version, source=pol.info.source,
                teacher=pol.info.name, teacher_version=pol.info.version,
                tracker_sha256=getattr(rt.inner, "sha256", None), tracker_run=getattr(rt.inner, "run", None),
                actuator=("ideal_pd_servo (legacy; the realistic actuator model is not applied)" if s.actuator_model is None
                          else f"{s.actuator_mode} (rrp.bodies.actuator, nominal params, latency {s.actuator_latency_ms:.1f} ms)"),
                **({"actuator_mode": s.actuator_record()} if s.actuator_model is not None else {}),
                privileged_teacher=pol.info.requires.privileged, teacher_variant="arc_only" if arc_only else "default",
                **({"speed_frac": te.vmax / te.r["vx"][1], "turn_gain": te.k} if hasattr(te, "vmax") else {}),
                waypoints=sc.meta.get("waypoints"),
                spec_hash=morph.spec_hash, wall_s=time.time() - t0, tracker_source_label=str(parse_source(rt.source)),
                physics=physics_provenance(s.model, sc.meta.get("contact_model", CONTACT_VERSION_DEFAULT)).to_dict(),
                motion=mrec.summary())
    return arr, meta, morph




def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--task", default="waypoint_contact", help="a registered mujoco/legged task with a scripted teacher")
    ap.add_argument("--tracker", default="auto")
    ap.add_argument("--tracker-id", default=None, help="registered actor <body>:<version> (default: the body's actor)")
    ap.add_argument("--max-steps", type=int, default=None, help="control steps; default TaskSpec.max_steps, else 1300")
    ap.add_argument("--sigmas", default="0,0.1,0.2,0.3", help="DART sigma cycled over seeds (action units)")
    ap.add_argument("--arc-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    sig = [float(x) for x in a.sigmas.split(",")]
    out = Path(a.out) / a.body
    seeds = parse_seed_spec(a.seeds)
    guard_sealed(a.body, seeds)
    out.mkdir(parents=True, exist_ok=True)
    assert_one_task(out, a.task)
    shard = a.shard or f"s{seeds[0]}-{seeds[-1]}"
    f = out / f"{shard}.npz"
    if f.exists():
        print(f"exists {f}")
        return
    eps, metas, morph = [], [], None
    for i, sd in enumerate(seeds):
        arr, meta, morph = collect_episode(a.body, sd, sig[i % len(sig)], a.tracker, a.max_steps, arc_only=a.arc_only,
                                        task=a.task, tracker_id=a.tracker_id)
        eps.append(arr)
        metas.append(meta)
        gc.collect()      # W8: each episode's session (~0.45 GB in contact_v2) sits in reference cycles; without a full
        #                   collection a 100-episode shard grew to ~13+ GB RSS and the peer watchdog shed the job
        print(json.dumps({k: meta[k] for k in ("seed", "sigma", "status", "steps", "ticks")}), flush=True)
    cat = {k: np.concatenate([e[k] for e in eps]) for k in eps[0]}
    cat["ep"] = np.concatenate([np.full(len(e["a"]), i) for i, e in enumerate(eps)])
    cat["t"] = np.concatenate([np.arange(len(e["a"])) for e in eps])
    np.savez_compressed(f, **cat, node_static=morph.node_static, node_asm=morph.node_asm,
                        asm_static=morph.asm_static, q0_all=morph.q0_all, n_policy=morph.n_policy, nf=morph.nf)
    prov = dataset_provenance(metas, source="scripted_teacher",
                              flags=dict(privileged_teacher=True, arc_only=a.arc_only, tracker=a.tracker, task=a.task,
                                         dart_sigmas=sig, prev_action_input=False),
                              notes="tick-level labels = clean body-tracker joint targets (native expert)")
    prov.versions.update(tracker_source="|".join(sorted({m["tracker_source_label"] for m in metas})),
                         tracker_version="|".join(sorted({str(m["tracker_version"]) for m in metas})),
                         tracker_sha256="|".join(sorted({str(m["tracker_sha256"]) for m in metas})),
                         actuator="|".join(sorted({m["actuator"] for m in metas})),
                         teacher="|".join(sorted({f"{m['teacher']}@{m['teacher_version']}" for m in metas})))
    f.with_suffix(".json").write_text(json.dumps(dict(body=a.body, task=a.task, task_view=task_view_of(a.task).as_dict(), episodes=metas,
                                                      handles=morph.handles,
                                                      asm_kind=morph.asm_kind, body_kind=morph.body_kind,
                                                      gait_period=morph.gait_period, provenance=prov.to_dict()),
                                                 indent=1))
    write_manifest(out, f"legged_latent_{a.body}_{shard}", metas, extra=dict(body=a.body, task=a.task, shard=shard, source=prov.source,
                                                                         npz=f.name), provenance=prov,
                   filename=f"{shard}.manifest.json")
    n_ok = sum(m["status"] == "success" for m in metas)
    print(json.dumps(dict(body=a.body, shard=shard, n=len(metas), success=n_ok, ticks=int(len(cat["a"])))))
