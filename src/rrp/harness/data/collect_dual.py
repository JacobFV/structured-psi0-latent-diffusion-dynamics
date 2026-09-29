"""Dual-arm teacher data collection + generation (same on-disk format as rrp.data.collect).

Public file: MultiFeaturizer inputs, flat namespaced actions (`r<i>:<group>`), q0, public
runtime statuses, flat action space. Private file: privileged labels (per manipulator), teacher
phases, true insertion geometry. Failed and infeasible attempts are recorded as episodes too.
The episodes load with rrp.learning.data.load_episodes/ChunkDataset unchanged.

Usage (peer, under a broker lease):
  python -m rrp.cli data collect-dual --config configs/data/support_insert_primary_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rrp.harness.data.collect import EpisodeRecord, privileged_labels, write_episode, read_episode
from rrp.core.provenance import FEATURIZER_VERSION as BASE_FEATURIZER_VERSION
from rrp.policies.features.multi import MultiFeaturizer
from rrp.harness.data.manifest import write_manifest, dataset_provenance
from rrp.core.provenance import physics_provenance

FEATURIZER_VERSION = f"feat-multi-v1+{BASE_FEATURIZER_VERSION}"
# optional config keys (W12 / D-126 #18), all default off: absent -> the v2 collection, byte-identical files
EXTRA_KEYS = ("contact_labels", "teacher_version", "teacher_options", "noise_phase_gate", "record_quality")


def collect_dual_episode(session, teacher, max_steps: int = 1200, episode_id: str = "",
                         split_lineage: dict | None = None, pair_key: str = "", exec_noise: float = 0.0,
                         noise_seed: int = 0, stop_after_success: int | None = None, noise_period: int = 1,
                         noise_on: int = 1, contact_labels: bool = False, noise_phase_gate: bool = False,
                         record_quality: bool = False) -> EpisodeRecord:
    """exec_noise > 0 (DART): every robot's executed ARM command = teacher command + N(0, exec_noise) (resampled
    every 5 steps, clipped to joint limits) during bursts of `noise_on` of every `noise_period` control steps (the
    precise dual teachers never converge under continuous noise); the recorded LABEL is always the clean command.
    stop_after_success: end the episode this many control steps after the PUBLIC runtime reports success (the
    handover teacher otherwise idles until max_steps).
    contact_labels (W12, default off = byte-identical files): also record the privileged per-tick contact frames
    (rrp.data.contact_labels) into the PRIVATE file under `contact_frames` (labels only).
    noise_phase_gate (D-126 #18, default off): DART noise is applied to an arm only while its teacher phase is a
    free-space phase (rrp.teachers.dual_smooth.dart_phase_allowed); the noise stream is drawn identically either way.
    record_quality (default off): the episode meta gains `motion` (rrp.data.dual_quality.DualQualityRecorder: per-arm
    jerk / phase-switch steps / joint margin, penetration, W12 contact metrics) for rrp.evaluation.gates.check_dual_dataset.
    The teacher version is recorded in the meta only when it is not v2 (`teacher_version`, `teacher_options`,
    `teacher_limits`)."""
    feat = MultiFeaturizer(session.model, session.scenario.robots)
    nrng = np.random.default_rng([noise_seed, 7])
    nz: dict = {}
    lim = {}
    for n, (g, c) in enumerate(zip(feat.aspace.node_group, feat.aspace.node_col)):
        lim.setdefault(g, {})[c] = (feat.aspace.lower[n], feat.aspace.upper[n])
    succ_at = None
    cfrec = None
    if contact_labels:
        from rrp.harness.data.contact_labels import ContactFrameRecorder
        cfrec = ContactFrameRecorder(session)
    qrec = None
    if record_quality:
        from rrp.harness.data.dual_quality import DualQualityRecorder
        qrec = DualQualityRecorder(session, teacher)
    arm_ent = {(h.robot, h.arm_group): e for e, h in session.handles.items()}
    f = teacher.feasibility()
    t0 = time.time()
    inputs, actions, labels, phases, statuses, q0s = [], [], [], [], [], []
    obs = session.observe()
    prev = None
    status = "infeasible" if not f["feasible"] else "running"
    steps = 0
    if f["feasible"]:
        for _ in range(max_steps):
            pi = feat(obs, prev)
            cmds = teacher.act()
            flat = MultiFeaturizer.flatten({i: c.groups for i, c in cmds.items()})
            a = feat.aspace.normalize([flat], pi.q0)[0]
            inputs.append(pi)
            actions.append(flat)
            q0s.append(pi.q0)
            lab = privileged_labels(session, feat)
            if session.scenario.name == "support_insert":
                lab["insertion_truth"] = session.insertion_truth()
            labels.append(lab)
            if cfrec is not None:
                cfrec.tick()
            phases.append(teacher.phase_label)
            statuses.append({e: v.status for e, v in session.runtime.instances.items()})
            clean_cmds = cmds
            if exec_noise > 0 and steps % noise_period < noise_on:
                noisy = {}
                for i, c in cmds.items():
                    g2 = dict(c.groups)
                    for g, v in c.groups.items():
                        if "arm" not in g:
                            continue
                        key = f"r{i}:{g}"
                        if steps % 5 == 0 or key not in nz:
                            nz[key] = nrng.normal(0, exec_noise, len(v))
                        lo = np.array([lim[key][j][0] for j in range(len(v))])
                        hi = np.array([lim[key][j][1] for j in range(len(v))])
                        if noise_phase_gate:
                            from rrp.policies.teachers.dual_smooth import dart_phase_allowed
                            ent = arm_ent.get((i, g))
                            if ent is None or not dart_phase_allowed(session.scenario.name, teacher.phase.get(ent, "")):
                                continue                      # clean command in contact phases (D-121 gating)
                        g2[g] = np.clip(np.asarray(v, float) + nz[key], lo, hi).tolist()
                    noisy[i] = c.model_copy(update={"groups": g2})
                cmds = noisy
            if qrec is not None:
                qrec.before_step(clean_cmds)
            res = session.step(cmds)
            if qrec is not None:
                qrec.after_step()
            obs = res.observation
            prev = a
            steps += 1
            if teacher.done:
                break
            if stop_after_success is not None:
                if succ_at is None and session.runtime.succeeded():
                    succ_at = steps
                if succ_at is not None and steps - succ_at >= stop_after_success:
                    break
        for _ in range(5):
            session.step(None)
        status = "success" if session.privileged_success() else "failure"
    rs = [mr.robot_spec for mr in session.scenario.robots]
    roles = {ent: dict(robot=h.robot, robot_name=rs[h.robot].name, assembly=h.assembly,
                       gripper_kind=h.gripper_kind) for ent, h in session.handles.items()}
    meta = dict(episode_id=episode_id, robot=" + ".join(r.name for r in rs), robot_key=pair_key,
                spec_hash=feat.spec_hash, robot_spec_hashes=[r.spec_hash for r in rs],
                lineage=sorted({l for r in rs for l in r.lineage}), roles=roles,
                controller_version=session.multi_controller_version, task=session.scenario.name,
                task_hash=hashlib.sha256(json.dumps(session.scenario.task, sort_keys=True).encode()).hexdigest()[:16],
                seed=session.seed, control_dt=session.dt, physics_dt=float(session.model.opt.timestep),
                steps=steps, status=status, feasibility=f, source="scripted_teacher", privileged_teacher=True,
                exec_noise=exec_noise, stop_after_success=stop_after_success,
                noise_burst=dict(period=noise_period, on=noise_on) if exec_noise > 0 else None,
                public_runtime_success=bool(session.runtime.succeeded()), featurizer=FEATURIZER_VERSION,
                frame=feat.frame, wall_s=time.time() - t0, split_lineage=split_lineage or {},
                n_distractors=0, retries={e: v.attempt for e, v in session.runtime.instances.items() if v.attempt},
                hole_frame_used=getattr(teacher, "hole_frame_used", None),
                insertion_truth=session.insertion_truth() if session.scenario.name == "support_insert"
                and f["feasible"] else None, physics=physics_provenance(session.model).to_dict())
    tv = getattr(teacher, "version", "v2")
    if tv != "v2":
        meta.update(teacher_version=getattr(teacher, "teacher_version", tv),
                    teacher_options=dict(vars(teacher.o)) if hasattr(teacher, "o") else None,
                    teacher_limits=dict(getattr(teacher, "limits", {})))
    if noise_phase_gate:
        meta["dart_variant"] = "phase_gated_v1"
    if qrec is not None:
        meta["motion"] = qrec.summary(session.scenario.name)
    public = dict(meta=meta, inputs=inputs, actions=actions, q0=q0s, statuses=statuses,
                  action_space=dict(node_group=feat.aspace.node_group, node_col=feat.aspace.node_col,
                                    lower=feat.aspace.lower, upper=feat.aspace.upper,
                                    is_gripper=feat.aspace.is_gripper, open_value=feat.aspace.open_value,
                                    closed_value=feat.aspace.closed_value, delta_scale=feat.aspace.delta_scale))
    private = dict(meta=dict(episode_id=episode_id, kind="privileged_labels"), labels=labels, phases=phases,
                   manipulators=list(session.manip_map), slots=[o.sim_body for o in session.detectables],
                   layout=session.scenario.meta.get("privileged_layout"))
    if cfrec is not None:
        from rrp.harness.data.contact_labels import CONTACT_LABEL_VERSION
        private["contact_frames"] = cfrec.recording().to_dict()
        private["contact_label_version"] = CONTACT_LABEL_VERSION
    return EpisodeRecord(public, private)


def _job(args):
    task, pair, seed, out_dir, split, max_steps, noise, stop_after, burst, *rest = args
    ex = dict(rest[0]) if rest else {}              # optional extras (W12 / D-126); absent = the v2 behaviour
    eid = f"{task}_{pair}_s{seed}" + (f"_dart{int(noise * 1000)}" if noise else "")
    ep_dir = Path(out_dir) / "episodes"
    done = ep_dir / f"{eid}.public.pkl.gz"
    if done.exists() and (ep_dir / f"{eid}.private.pkl.gz").exists():
        try:   # resume: reuse a completed, readable episode
            meta = read_episode(done)["meta"]
            return dict(meta, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                                     for p in (done, ep_dir / f"{eid}.private.pkl.gz")}, resumed=True)
        except Exception:  # noqa: BLE001 - corrupt partial file: regenerate
            pass
    from rrp.policies.teachers.dual_validate import make_session
    from rrp.policies.teachers.dual import TEACHERS
    sess = make_session(task, pair, seed)
    from rrp.policies.teachers.dual_smooth import make_dual_teacher
    teacher = make_dual_teacher(task, sess, ex.get("teacher_version"), ex.get("teacher_options")) \
        if ex.get("teacher_version") not in (None, "v2") else TEACHERS[task](sess)
    rec = collect_dual_episode(sess, teacher, max_steps=max_steps, episode_id=eid,
                               split_lineage=dict(split=split, robot_key=pair), pair_key=pair,
                               exec_noise=noise, noise_seed=seed, stop_after_success=stop_after,
                               noise_period=burst[0], noise_on=burst[1], contact_labels=bool(ex.get("contact_labels")),
                               noise_phase_gate=bool(ex.get("noise_phase_gate")),
                               record_quality=bool(ex.get("record_quality")))
    meta = write_episode(rec, ep_dir)
    del rec, sess
    import gc
    gc.collect()      # sessions hold MuJoCo buffers in reference cycles; free them per episode
    return meta


def build_jobs(config: dict) -> tuple[list, dict]:
    """Job tuples for generate(). Without any EXTRA_KEYS in the config the tuples are exactly the v2 9-tuples."""
    out = Path(config["out_dir"])
    jobs = []
    extras = {k: config[k] for k in EXTRA_KEYS if config.get(k) not in (None, False, "v2")}
    for item in config["items"]:
        noises = item.get("noise_levels", config.get("noise_levels", [0.0]))    # 0.0 = clean; >0 = DART
        for k in range(item["episodes"]):
            for nl in noises:
                jobs.append((item.get("task", config.get("task", "support_insert")), item["pair"], item["seed_start"] + k,
                             str(out), item["split"], config.get("max_steps", 1200), float(nl),
                             config.get("stop_after_success"), tuple(config.get("noise_burst", (1, 1))))
                            + ((extras,) if extras else ()))
    return jobs, extras


def generate(config: dict) -> dict:
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    jobs, extras = build_jobs(config)
    t0 = time.time()
    metas = []
    # resume in the parent: completed, readable episodes are not resubmitted. (Submitting them made
    # workers hit max_tasks_per_child in seconds; CPython 3.12's worker replacement then hung the pool.)
    todo = []
    for j in jobs:
        eid = f"{j[0]}_{j[1]}_s{j[2]}" + (f"_dart{int(j[6] * 1000)}" if j[6] else "")
        pub, prv = out / "episodes" / f"{eid}.public.pkl.gz", out / "episodes" / f"{eid}.private.pkl.gz"
        if pub.exists() and prv.exists():
            try:
                meta = read_episode(pub)["meta"]
                metas.append(dict(meta, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                                              for p in (pub, prv)}, resumed=True))
                continue
            except Exception:  # noqa: BLE001 - corrupt partial file: regenerate
                pass
        todo.append(j)
    print(f"[collect_dual] resumed {len(metas)}, to generate {len(todo)}", flush=True)
    # no max_tasks_per_child: CPython 3.12 worker replacement hung this pool twice (workers gone,
    # parent blocked). Memory is bounded by one cached robot pair per worker + gc after each episode.
    with ProcessPoolExecutor(max_workers=config.get("workers", os.cpu_count())) as ex:
        futs = [ex.submit(_job, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            try:
                metas.append(f.result())
            except Exception as e:  # noqa: BLE001 - failures are data too
                metas.append(dict(status="generation_error", error=repr(e)[:300]))
            if i % 50 == 0:
                print(f"[collect_dual] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    metas = sorted(metas, key=lambda m: m.get("episode_id", ""))
    prov = dataset_provenance(metas, source="scripted_teacher", featurizer_version=FEATURIZER_VERSION,
                              flags=dict(privileged_teacher=True, noise_levels=config.get("noise_levels", [0.0]),
                                         noise_burst=config.get("noise_burst", (1, 1)),
                                         **({"collect_options": extras} if extras else {})))
    man = write_manifest(out, config["name"], metas,
                         extra=dict(config=config, wall_s=time.time() - t0, featurizer=FEATURIZER_VERSION,
                                    source=prov.source, privileged_teacher=True), provenance=prov)
    print(json.dumps({"name": config["name"], "status_counts": man["status_counts"], "wall_s": time.time() - t0}))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    if a.workers:
        cfg["workers"] = a.workers
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    generate(cfg)
