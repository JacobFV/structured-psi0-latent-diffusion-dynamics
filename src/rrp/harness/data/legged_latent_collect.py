"""Tick-level (50 Hz) legged teacher data for the latent-packet path (source label: scripted_teacher driving the
frozen body tracker; the tracker is the native-level expert whose joint targets system 0 learns to realize).
`--task` (default waypoint_contact) is any registered mujoco/legged task with a scripted teacher; its env, teacher,
event slots and target entities come from the task registry (`legged_collect.open_episode`, features.legged.TaskView).
One output directory (`--out`) holds one task (bodies are subdirectories); the manifest records task, teacher and tracker
sha; the sealed-split guard runs before collection. The episode runs on `harness.rollout` (`legged_collect.run_teacher_episode`)
with recorder hooks; the session's control regime is the task's: "base_velocity" (10 Hz steps, 5 tracker ticks each) or
"wholebody" (50 Hz steps, the teacher's `legs` targets through the direct slot, its `upper` targets recorded per tick).

Per tick we store PUBLIC inputs (joint encoders, IMU, touch, system-i global context from localization/detector/
task view at the last 10 Hz observation, osc-v1 phase), the expert NATIVE label (clean tracker joint target in
action units (target - q0) / action_scale), and PRIVILEGED truth used only as probe labels / evaluation truth
(true base pose, true foot contacts, fall, true waypoint positions). Also per tick: the teacher's `upper` targets (held
joints, action units (target - q0) / action_scale, `upper_valid` only under wholebody control), the public terrain scan
(`terrain` [T,77] elevation + `terrain_valid`: the session's own scan when a scan-input tracker makes it, else the collector's
`TerrainScan` on the same geometry, meta `terrain_source`), and the training-only relation labels `com_support` (+ `_valid`;
legged_collect.com_support) and `foothold_cell` [T,M] (hindsight: -2 absent / no touchdown in the episode / outside the scan,
-1 planted, >= 0 the scan cell, in the yaw frame of that tick, where the foot next touches down).

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

from rrp.envs.mujoco.legged import DIRECT_CONTROLS
from rrp.envs.mujoco.legged_core import SCAN_DIM, SCAN_DX, SCAN_NX, SCAN_NY, SCAN_X0, SCAN_Y0, TerrainScan
from rrp.policies.features.legged import LeggedMorph, public_context, local_state, active_event, task_view_of, TICK_DT
from rrp.harness.data.legged_collect import (DEFAULT_MAX_STEPS, assert_one_task, com_support, guard_sealed, make_teacher,
                                             open_episode, outcome, run_teacher_episode, tracker_of)
from rrp.core.provenance import CONTACT_VERSION_DEFAULT, parse_source, physics_provenance
from rrp.harness.data.manifest import dataset_provenance, write_manifest
from rrp.core.runs import parse_seed_spec


POST_SECONDS = 1.0          # post-halt standing after the teacher's end (the halt / stand regime), in seconds

PER_TICK = ("q", "qd", "imu", "touch", "osc", "a", "ctx", "ev", "pose", "contact", "bad", "height", "cmd",
            "upper", "upper_valid", "terrain", "terrain_valid", "com_support", "com_support_valid")


def foothold_cells(contact: np.ndarray, foot_xy: np.ndarray, pose: np.ndarray, M: int) -> np.ndarray:
    """Hindsight `foothold_cell` [T, M] (assembly order, the feet first, M = LeggedMorph.M): for a foot planted at tick t -1;
    for a foot in swing the terrain-scan cell (index ix * SCAN_NY + iy of the 11 x 7 grid in the base yaw frame at tick t)
    containing the foot's planar position at its next touchdown (the first later tick with the foot down); -2 where there is
    none (no later touchdown in the episode, the landing outside the scan) and on every non-foot assembly.
    contact [T, nf] bool, foot_xy [T, nf, 2] world, pose [T, >=3] (x, y, yaw)."""
    T, nf = contact.shape
    out = np.full((T, M), -2, np.int64)
    for f in range(nf):
        nxt = None
        for t in range(T - 1, -1, -1):
            if contact[t, f]:
                out[t, f], nxt = -1, t
            elif nxt is not None:
                x, y, yaw = (float(v) for v in pose[t, :3])
                dx, dy = foot_xy[nxt, f, 0] - x, foot_xy[nxt, f, 1] - y
                c, s = np.cos(yaw), np.sin(yaw)
                ix = int(round((c * dx + s * dy - SCAN_X0) / SCAN_DX))
                iy = int(round((-s * dx + c * dy - SCAN_Y0) / SCAN_DX))
                if 0 <= ix < SCAN_NX and 0 <= iy < SCAN_NY:
                    out[t, f] = ix * SCAN_NY + iy
    return out


class RecordingTracker:
    """Wraps the session's tracker slot: records (public local state, clean expert target, terrain scan, labels) every native
    tick and executes the target with optional DART noise. base_velocity control: the slot is the frozen body tracker (the
    clean target is its output). "wholebody": the slot is `DirectTargets`; the clean target is the teacher's `legs` group and
    the teacher's `upper` group is read from `session.upper_target` (what the session applies this tick)."""

    def __init__(self, session, morph, sigma: float, rng):
        self.s, self.inner, self.m = session, session.tracker, morph
        self.sigma, self.rng = sigma, rng
        self.rec = None
        self.ticks = 0
        self.ctx = self.ev = self.cmd_truth = None
        self.direct = session.control in DIRECT_CONTROLS
        self.scan_own = TerrainScan(session.binding) if session.terrain is None else None
        self.scan = self.scan_own or session.terrain
        bt = tracker_of(session)
        for k in ("source", "version"):
            setattr(self, k, getattr(bt, k))
        self.body = bt

    @property
    def terrain_source(self) -> str:
        return "collector_scan" if self.scan_own is not None else "session_scan"

    def reset(self, phase=0.0):
        self.ticks = 0
        self.inner.reset(phase)
        if self.scan_own is not None:
            self.scan_own.reset(self.s.data, self.s.seed)

    def state(self):
        return self.inner.state()

    def load(self, st):
        self.inner.load(st)

    def act(self, data, cmd):
        if self.scan_own is not None:
            self.scan_own.tick(data)
        tgt = self.inner.act(data, cmd)
        s, b = self.s, self.s.binding
        osc = (self.ticks * TICK_DT / self.m.gait_period) % 1.0
        if self.rec is not None:
            q, qd, imu, touch = local_state(s, self.m)
            fc, bad = b.contacts(data)
            r = self.rec
            r["q"].append(q); r["qd"].append(qd); r["imu"].append(imu)
            r["touch"].append(touch); r["osc"].append(osc)
            r["a"].append(((tgt - b.q0) / b.action_scale).astype(np.float32))
            r["ctx"].append(self.ctx)
            r["ev"].append(self.ev)
            r["pose"].append(s.base_pose_truth().astype(np.float32))
            r["contact"].append(fc.copy())
            r["bad"].append(bool(bad))
            r["height"].append(float(data.qpos[b.qa + 2]))
            r["cmd"].append(np.asarray(cmd if self.cmd_truth is None else self.cmd_truth, np.float32))
            whole = s.control == "wholebody"
            r["upper"].append(((np.asarray(s.upper_target, float) - b.q0_held) / b.action_scale).astype(np.float32)
                              if whole else np.zeros(len(b.held_act), np.float32))
            r["upper_valid"].append(whole)
            r["terrain"].append(np.asarray(self.scan.values, np.float32).copy())
            r["terrain_valid"].append(np.asarray(self.scan.valid, bool).copy())
            cs = com_support(s)
            r["com_support"].append(cs)
            r["com_support_valid"].append(True)
            self.foot_xy.append(np.asarray([data.xpos[f][:2] for f in b.foot_bids], np.float64))
        self.ticks += 1
        if self.sigma > 0:
            tgt = np.clip(tgt + self.rng.normal(0, self.sigma * b.action_scale, len(tgt)), b.lo, b.hi)
        return tgt

    def start(self):
        self.rec = {k: [] for k in PER_TICK}
        self.foot_xy = []


class TickContext:
    """Rollout hook: the public system-i context (`public_context`, `active_event`) handed to every tick of a control step,
    taken at the last 10 Hz observation (base_velocity: every step; direct controls: steps after a boundary), and, for direct
    controls, the teacher's base-velocity command (`command_values`, the tracker input the `legs` targets realise)."""

    def __init__(self, s, morph, rt, te):
        self.s, self.m, self.rt, self.te = s, morph, rt, te

    def on_act(self, i, obs, act):
        s, rt = self.s, self.rt
        if not rt.direct or rt.ctx is None or s.boundary:
            rt.ctx = public_context(s, (rt.ticks * TICK_DT / self.m.gait_period) % 1.0)
            rt.ev = active_event(s.runtime)
        if rt.direct and hasattr(self.te, "command_values"):
            rt.cmd_truth = np.asarray(self.te.command_values(), np.float32)


def collect_episode(body: str, seed: int, sigma: float, tracker_kind="auto", max_steps=None, arc_only=False,
                    task: str = "waypoint_contact", tracker_id: str | None = None):
    rng = np.random.default_rng([seed, 77])
    spec, s = open_episode(task, body, seed, tracker_kind=tracker_kind, tracker_id=tracker_id)
    sc = s.scenario
    max_steps = max_steps or spec.max_steps or DEFAULT_MAX_STEPS
    direct = s.control in DIRECT_CONTROLS
    mrec = None
    if s.control != "wholebody":
        # W8/D-112: the W6 motion-quality recorder (read-only; install_legged with the nominal perturbation performs the
        # original tick operations in the same order), so every episode meta carries slip_ratio/cot/... for the dataset gate.
        # Not available under wholebody control (install_legged holds the upper joints at the default pose): motion is None.
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
    if direct:
        rt.ticks = 0                # the teacher's body tracker starts its phase at 0 here (after the reset settle)
    rt.start()
    t0 = time.time()
    step_s = TICK_DT if direct else s.dt                   # seconds per rollout step (wholebody: one tracker tick)
    ticks_per_step = int(round(step_s / TICK_DT))
    post = int(round(POST_SECONDS / step_s))
    steps = run_teacher_episode(spec, s, pol, te, [TickContext(s, morph, rt, te)], max_steps=max_steps, post_steps=post)
    status, reason = outcome(spec, s)
    arr = {k: np.asarray(v) for k, v in rt.rec.items()}
    arr["foothold_cell"] = foothold_cells(arr["contact"], np.asarray(rt.foot_xy), arr["pose"], morph.M)
    meta = dict(body=body, seed=seed, sigma=sigma, task=task, status=status, failure_reason=reason, steps=steps,
                max_steps=max_steps, ticks=len(arr["a"]), control=s.control, step_ticks=ticks_per_step,
                terrain_source=rt.terrain_source,
                **({"legs_source": getattr(te, "legs", None)} if direct else {}),
                tracker_source=rt.source, tracker_version=rt.version, source=pol.info.source,
                teacher=pol.info.name, teacher_version=pol.info.version,
                tracker_sha256=getattr(rt.body, "sha256", None), tracker_run=getattr(rt.body, "run", None),
                actuator=("ideal_pd_servo (legacy; the realistic actuator model is not applied)" if s.actuator_model is None
                          else f"{s.actuator_mode} (rrp.bodies.actuator, nominal params, latency {s.actuator_latency_ms:.1f} ms)"),
                **({"actuator_mode": s.actuator_record()} if s.actuator_model is not None else {}),
                privileged_teacher=pol.info.requires.privileged, teacher_variant="arc_only" if arc_only else "default",
                **({"speed_frac": te.vmax / te.r["vx"][1], "turn_gain": te.k} if hasattr(te, "vmax") else {}),
                waypoints=sc.meta.get("waypoints"),
                spec_hash=morph.spec_hash, wall_s=time.time() - t0, tracker_source_label=str(parse_source(rt.source)),
                physics=physics_provenance(s.model, sc.meta.get("contact_model", CONTACT_VERSION_DEFAULT)).to_dict(),
                motion=mrec.summary() if mrec is not None else None)
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
    assert_one_task(Path(a.out), a.task)
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
                        asm_static=morph.asm_static, q0_all=morph.q0_all, n_policy=morph.n_policy, nf=morph.nf,
                        node_parent=morph.node_parent, asm_trunk=morph.asm_trunk)
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
