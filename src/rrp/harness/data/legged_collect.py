"""Legged teacher data collection for `waypoint_contact` (same public/private file split as
rrp.data.collect: *.public.pkl.gz holds policy-visible inputs + native base-velocity actions,
*.private.pkl.gz holds privileged labels). Failed attempts are kept.

Public per-step record: joint qpos/qvel (encoders), IMU (quat, gyro, acc), foot touch,
localization (noisy x, y, yaw), detector descriptors (waypoint slots), public predicate
estimates, public task view; action = base_velocity [vx, vy, wz] sent to the body tracker.
Private per-step record: true base pose/velocity, true foot contacts, truth predicates,
event completion truth, teacher phase.

usage: python -m rrp.harness.data.legged_collect --body hexapod6 --seeds 0-19 --out data/legged/waypoint_contact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rrp.policies.teachers.legged import WaypointTeacher
from rrp.core.paths import rrp_home
from rrp.core.provenance import CONTACT_VERSION_DEFAULT, parse_source, physics_provenance
from rrp.harness.data.collect import EpisodeRecord, write_episode
from rrp.harness.data.manifest import dataset_provenance, write_manifest
from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
from rrp.core.runs import parse_seed_spec


def public_record(obs) -> dict:
    ns = obs.measured_node_state
    ch = {c.name: np.asarray(c.values, np.float32) for c in obs.declared_sensor_channels}
    return dict(t=obs.sensor_time, qpos=np.asarray(ns.qpos, np.float32), qvel=np.asarray(ns.qvel, np.float32),
                channels=ch,
                objects=[dict(slot=o.slot, descriptor=o.descriptor, pos=o.position_estimate, cov=o.position_cov_diag,
                              visible=o.visible) for o in obs.object_descriptors],
                predicates=[(p.predicate, tuple(p.args), p.value, p.known) for p in obs.predicate_estimates],
                task=obs.task_input.model_dump() if obs.task_input is not None and hasattr(obs.task_input, "model_dump")
                else None)


def private_record(s: LeggedSession, teacher) -> dict:
    b = s.binding
    fc, bad = b.contacts(s.data)
    tr = s.truth()
    return dict(base_pose=s.base_pose_truth().astype(np.float32),
                base_vel_body=b.base_lin_vel_body(s.data).astype(np.float32),
                base_height=float(s.data.qpos[b.qa + 2]), foot_contact=fc, bad_contact=bool(bad),
                predicates_truth=dict(tr.predicates), event_completion_truth=dict(tr.event_completion_truth),
                teacher_phase=teacher.phase)


def collect_episode(body: str, seed: int, tracker_kind: str = "auto", max_steps: int = 1300,
                    split_lineage: dict | None = None, arc_only: bool = False) -> EpisodeRecord:
    sc = build_waypoint_contact(body, seed)
    s = LeggedSession(sc, tracker_kind=tracker_kind, seed=seed)
    te = WaypointTeacher(s, arc_only=arc_only)
    t0 = time.time()
    obs = s._last_obs
    inputs, actions, labels, phases = [], [], [], []
    steps = 0
    for k in range(max_steps):
        cmd = te.act()
        inputs.append(public_record(obs))
        labels.append(private_record(s, te))
        actions.append(dict(cmd.groups))
        phases.append(te.phase)
        res = s.step(cmd)
        obs = res.observation
        steps += 1
        if te.done or s.fell:
            break
    status = "success" if s.privileged_success() and not s.fell else ("fell" if s.fell else "failure")
    rs = sc.robots[0].robot_spec
    meta = dict(episode_id=f"{body}_wpc_s{seed}", robot=body, spec_hash=rs.spec_hash, lineage=rs.lineage,
                family=rs.family, synthetic=rs.synthetic, controller_version=s.controller_version(),
                tracker_source=s.tracker.source, task="waypoint_contact",
                task_hash=hashlib.sha256(json.dumps(sc.task, sort_keys=True).encode()).hexdigest()[:16],
                seed=seed, control_dt=s.dt, physics_dt=float(s.model.opt.timestep), tracker_hz=50.0, steps=steps,
                status=status, public_runtime_success=bool(s.runtime.succeeded()),
                event_status={e: i.status for e, i in s.runtime.instances.items()},
                source="scripted_teacher", privileged_teacher=True, teacher_variant="arc_only" if arc_only else "default", waypoints=sc.meta["waypoints"],
                wall_s=time.time() - t0, split_lineage=split_lineage or {"lineage": rs.lineage},
                tracker_source_label=str(parse_source(s.tracker.source)),
                physics=physics_provenance(s.model, sc.meta.get("contact_model", CONTACT_VERSION_DEFAULT)).to_dict())
    public = dict(meta=meta, inputs=inputs, actions=actions,
                  action_space=dict(group="base_velocity", units=["m/s", "m/s", "rad/s"],
                                    lower=s.tracker_contract.command_groups[0].lower,
                                    upper=s.tracker_contract.command_groups[0].upper))
    private = dict(meta=dict(episode_id=meta["episode_id"], kind="privileged_labels"), labels=labels, phases=phases)
    return EpisodeRecord(public, private)


_seeds = parse_seed_spec          # W4 dedup: one implementation in rrp.contracts.runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tracker", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arc-only", action="store_true", help="teacher keeps min forward speed while turning")
    ap.add_argument("--teacher-report", action="store_true",
                    help="also write artifacts/assets/legged_teacher/<body>.json (teacher validation gate)")
    a = ap.parse_args(argv)
    out = Path(a.out) / a.body
    rows = []
    from rrp.harness.data.collect import read_episode
    for sd in _seeds(a.seeds):
        pub = out / f"{a.body}_wpc_s{sd}.public.pkl.gz"
        if pub.exists() and (out / f"{a.body}_wpc_s{sd}.private.pkl.gz").exists() and \
                read_episode(pub)["meta"]["tracker_source"] == ("scripted_controller" if a.tracker == "cpg" else "learned_tracker"):
            m = dict(read_episode(pub)["meta"], files={})       # resumable: keep finished episodes
        else:
            rec = collect_episode(a.body, sd, a.tracker, arc_only=a.arc_only)
            m = write_episode(rec, out)
        rows.append(dict(episode_id=m["episode_id"], status=m["status"], steps=m["steps"], files=m["files"],
                         tracker_source=m["tracker_source"], event_status=m["event_status"],
                         source=m.get("source", "scripted_teacher"), physics=m.get("physics"),
                         tracker_version=m.get("controller_version")))
        print(json.dumps(rows[-1]), flush=True)
    summ = dict(body=a.body, n=len(rows), success=sum(r["status"] == "success" for r in rows),
                fell=sum(r["status"] == "fell" for r in rows), episodes=rows)
    trk = sorted({r["tracker_source"] for r in rows})
    prov = dataset_provenance(rows, source="scripted_teacher",
                              flags=dict(privileged_teacher=True, arc_only=a.arc_only, tracker=a.tracker,
                                         prev_action_input=False),
                              notes="native actions = base_velocity commands executed by the body tracker")
    prov.versions.update(tracker_source="|".join(str(parse_source(t)) for t in trk),
                         tracker_version="|".join(sorted({str(r.get("tracker_version")) for r in rows})))
    write_manifest(out, f"legged_waypoint_contact_{a.body}", rows,
                   extra={k: v for k, v in summ.items() if k != "episodes"} | dict(source=prov.source), provenance=prov)
    if a.teacher_report:
        rep = rrp_home() / "artifacts" / "assets" / "legged_teacher" / (
            f"{a.body}_arc_only.json" if a.arc_only else f"{a.body}.json")
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(dict(body=a.body, task="waypoint_contact", n=len(rows),
                                       success_rate=summ["success"] / max(1, len(rows)), fell=summ["fell"],
                                       tracker_source=rows[0]["tracker_source"] if rows else None,
                                       teacher="WaypointTeacher (scripted_teacher, privileged base pose)"
                                       + (" arc_only variant" if a.arc_only else ""),
                                       seeds=a.seeds, episodes=rows), indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
