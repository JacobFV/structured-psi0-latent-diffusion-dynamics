"""Dual-arm teacher data collection + generation (same on-disk format as rrp.data.collect).

Public file: MultiFeaturizer inputs, flat namespaced actions (`r<i>:<group>`), q0, public
runtime statuses, flat action space. Private file: privileged labels (per manipulator), teacher
phases, true insertion geometry. Failed and infeasible attempts are recorded as episodes too.
The episodes load with rrp.learning.data.load_episodes/ChunkDataset unchanged.

Usage (peer, under a broker lease):
  python -m rrp.data.collect_dual --config configs/data/support_insert_primary_v1.json
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

from rrp.data.collect import EpisodeRecord, privileged_labels, write_episode, read_episode
from rrp.data.features_multi import MultiFeaturizer
from rrp.data.manifest import write_manifest

FEATURIZER_VERSION = "feat-multi-v1"


def collect_dual_episode(session, teacher, max_steps: int = 1200, episode_id: str = "",
                         split_lineage: dict | None = None, pair_key: str = "") -> EpisodeRecord:
    feat = MultiFeaturizer(session.model, session.scenario.robots)
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
            phases.append(teacher.phase_label)
            statuses.append({e: v.status for e, v in session.runtime.instances.items()})
            res = session.step(cmds)
            obs = res.observation
            prev = a
            steps += 1
            if teacher.done:
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
                public_runtime_success=bool(session.runtime.succeeded()), featurizer=FEATURIZER_VERSION,
                frame=feat.frame, wall_s=time.time() - t0, split_lineage=split_lineage or {},
                n_distractors=0, retries={e: v.attempt for e, v in session.runtime.instances.items() if v.attempt},
                hole_frame_used=getattr(teacher, "hole_frame_used", None),
                insertion_truth=session.insertion_truth() if session.scenario.name == "support_insert"
                and f["feasible"] else None)
    public = dict(meta=meta, inputs=inputs, actions=actions, q0=q0s, statuses=statuses,
                  action_space=dict(node_group=feat.aspace.node_group, node_col=feat.aspace.node_col,
                                    lower=feat.aspace.lower, upper=feat.aspace.upper,
                                    is_gripper=feat.aspace.is_gripper, open_value=feat.aspace.open_value,
                                    closed_value=feat.aspace.closed_value, delta_scale=feat.aspace.delta_scale))
    private = dict(meta=dict(episode_id=episode_id, kind="privileged_labels"), labels=labels, phases=phases,
                   manipulators=list(session.manip_map), slots=[o.sim_body for o in session.detectables],
                   layout=session.scenario.meta.get("privileged_layout"))
    return EpisodeRecord(public, private)


def _job(args):
    task, pair, seed, out_dir, split, max_steps = args
    eid = f"{task}_{pair}_s{seed}"
    ep_dir = Path(out_dir) / "episodes"
    done = ep_dir / f"{eid}.public.pkl.gz"
    if done.exists() and (ep_dir / f"{eid}.private.pkl.gz").exists():
        try:   # resume: reuse a completed, readable episode
            meta = read_episode(done)["meta"]
            return dict(meta, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                                     for p in (done, ep_dir / f"{eid}.private.pkl.gz")}, resumed=True)
        except Exception:  # noqa: BLE001 - corrupt partial file: regenerate
            pass
    from rrp.control.dual_validate import make_session
    from rrp.control.dual_teachers import TEACHERS
    sess = make_session(task, pair, seed)
    rec = collect_dual_episode(sess, TEACHERS[task](sess), max_steps=max_steps, episode_id=eid,
                               split_lineage=dict(split=split, robot_key=pair), pair_key=pair)
    return write_episode(rec, ep_dir)


def generate(config: dict) -> dict:
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item in config["items"]:
        for k in range(item["episodes"]):
            jobs.append((item.get("task", config.get("task", "support_insert")), item["pair"], item["seed_start"] + k,
                         str(out), item["split"], config.get("max_steps", 1200)))
    t0 = time.time()
    metas = []
    with ProcessPoolExecutor(max_workers=config.get("workers", os.cpu_count()), max_tasks_per_child=100) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                metas.append(f.result())
            except Exception as e:  # noqa: BLE001 - failures are data too
                metas.append(dict(status="generation_error", error=repr(e)[:300]))
            if i % 50 == 0:
                print(f"[collect_dual] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    man = write_manifest(out, config["name"], sorted(metas, key=lambda m: m.get("episode_id", "")),
                         extra=dict(config=config, wall_s=time.time() - t0, featurizer=FEATURIZER_VERSION,
                                    source="scripted_teacher", privileged_teacher=True))
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


if __name__ == "__main__":
    main()
