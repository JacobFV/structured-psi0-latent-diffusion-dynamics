"""Legged teacher data collection for any registered `mujoco/legged` task with a scripted teacher (`--task`, default
`waypoint_contact`; h_steps / h_gap for humanoids). The env comes from `make_env` (the task's `build`), the teacher from
the policy registry key `TaskSpec.teacher`, the failure vocabulary from the task's judge; nothing here names a task.
Same public/private file split as rrp.harness.data.collect: *.public.pkl.gz holds policy-visible inputs + native
base-velocity actions, *.private.pkl.gz holds privileged labels. Failed attempts are kept, with their failure reason.
The manifest records task, teacher (source, version), tracker (source, sha256) and the sealed-split guard is called
before any episode is written (D-138: a sealed body / seed is refused).

Public per-step record: joint qpos/qvel (encoders), IMU (quat, gyro, acc), foot touch,
localization (noisy x, y, yaw), detector descriptors (waypoint slots), public predicate
estimates, public task view; action = base_velocity [vx, vy, wz] sent to the body tracker.
Private per-step record: true base pose/velocity, true foot contacts, truth predicates,
event completion truth, teacher phase.

usage: python -m rrp.cli data legged-collect --body hexapod6 --seeds 0-19 --out data/legged/waypoint_contact [--task T]
One output directory holds one task (the manifest names it; a directory of another task is refused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rrp.core.paths import rrp_home
from rrp.core.provenance import CONTACT_VERSION_DEFAULT, parse_source, physics_provenance
from rrp.harness.data.collect import EpisodeRecord, write_episode
from rrp.harness.data.manifest import dataset_provenance, write_manifest
from rrp.core.runs import parse_seed_spec
from rrp.core.sealed import SealedSplit
from rrp.envs.base import env_failure_reason, make_env
from rrp.policies.base import make_policy
from rrp.policies.features.legged import task_view, task_view_of
from rrp.tasks.spec import get_task

ENV_ID = "mujoco/legged"
DEFAULT_MAX_STEPS = 1300          # control steps (10 Hz) when neither --max-steps nor TaskSpec.max_steps says otherwise


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


def open_episode(task: str, body: str, seed: int, *, tracker_kind: str = "auto", tracker_id: str | None = None):
    """(TaskSpec, session) of one episode of `task`: the env through `make_env` (the task's `build`); `tracker_id` = a
    registered actor "<body>:<version>" (else the body's default)."""
    spec = get_task(task)
    if ENV_ID not in spec.envs:
        raise ValueError(f"task {task!r} does not exist in {ENV_ID} (envs: {sorted(spec.envs)})")
    if spec.teacher is None:
        raise ValueError(f"task {task!r} has no scripted teacher (TaskSpec.teacher is None): nothing to collect")
    s = make_env(ENV_ID, task=task, body=body, seed=seed, tracker_kind=tracker_kind,
                 **({"tracker": tracker_id} if tracker_id else {}))
    if task_view(s.scenario.task).as_dict() != task_view_of(task).as_dict():
        raise ValueError(f"{ENV_ID} built task graph {s.scenario.task['task_id']!r} for task {task!r} "
                         f"(TaskSpec.graph {spec.graph!r})")
    return spec, s


def make_teacher(spec, s, options: dict | None = None, *, arc_only: bool = False):
    """(policy, teacher) for the session: the policy registry key `TaskSpec.teacher`; `options` go to its factory (a
    teacher without such options ignores them); `arc_only` is the waypoint teacher's variant and an error elsewhere."""
    pol = make_policy(spec.teacher, options=dict(options or {}, **({"arc_only": True} if arc_only else {})))
    te = pol.make(s)
    if arc_only and not hasattr(te, "arc_only"):
        raise ValueError(f"--arc-only is a variant of the waypoint teacher; {spec.teacher} has none")
    return pol, te


def teacher_done(te, s) -> bool:
    """The teacher's own `done` when it has one (waypoint), else the public task graph's terminal state."""
    d = getattr(te, "done", None)
    return bool(d) if d is not None else bool(s.runtime.succeeded() or s.runtime.failed_terminal())


def outcome(spec, s) -> tuple[str, str | None]:
    """(status, failure_reason). status: success | fell | failure (privileged success without a fall decides it). The
    reason is the env's own public code, else "fell", else the task judge's reason (the failure vocabulary of the task)."""
    if s.privileged_success() and not s.fell:
        return "success", None
    status = "fell" if s.fell else "failure"
    reason = env_failure_reason(s) or ("fell" if s.fell else None)
    if reason is None:
        try:
            j = spec.judge(s, float(s.data.time), spec.max_seconds)
            reason = j.failure_reason if j.done and j.failure_reason else "incomplete"
        except RuntimeError as e:                 # the judge insists the env must name the failure (HJ): record, do not hide
            reason = f"judge_error: {e}"
    return status, reason


def guard_sealed(body: str, seeds, what: str = "collect") -> None:
    SealedSplit.load().assert_train_allowed([body], seeds, what=what)


def assert_one_task(out: Path, task: str) -> None:
    """One output directory holds one task: refuse a directory whose manifests name another (older ones: waypoint_contact)."""
    for f in sorted(Path(out).glob("*manifest.json")):
        got = json.loads(f.read_text()).get("task", "waypoint_contact")
        if got != task:
            raise SystemExit(f"{out} holds task {got!r} ({f.name}); collecting {task!r} needs its own directory")


def episode_tag(task: str) -> str:
    return "wpc" if task == "waypoint_contact" else task


def private_record(s, teacher) -> dict:
    b = s.binding
    fc, bad = b.contacts(s.data)
    tr = s.truth()
    return dict(base_pose=s.base_pose_truth().astype(np.float32),
                base_vel_body=b.base_lin_vel_body(s.data).astype(np.float32),
                base_height=float(s.data.qpos[b.qa + 2]), foot_contact=fc, bad_contact=bool(bad),
                predicates_truth=dict(tr.predicates), event_completion_truth=dict(tr.event_completion_truth),
                teacher_phase=getattr(teacher, "phase", None))


def collect_episode(body: str, seed: int, tracker_kind: str = "auto", max_steps: int | None = None,
                    split_lineage: dict | None = None, arc_only: bool = False, task: str = "waypoint_contact",
                    tracker_id: str | None = None) -> EpisodeRecord:
    spec, s = open_episode(task, body, seed, tracker_kind=tracker_kind, tracker_id=tracker_id)
    pol, te = make_teacher(spec, s, arc_only=arc_only)
    sc = s.scenario
    max_steps = max_steps or spec.max_steps or DEFAULT_MAX_STEPS
    t0 = time.time()
    obs = s._last_obs
    inputs, actions, labels, phases = [], [], [], []
    steps = 0
    for k in range(max_steps):
        cmd = te.act()
        inputs.append(public_record(obs))
        labels.append(private_record(s, te))
        actions.append(dict(cmd.groups))
        phases.append(getattr(te, "phase", None))
        res = s.step(cmd)
        obs = res.observation
        steps += 1
        if teacher_done(te, s) or s.fell:
            break
    status, reason = outcome(spec, s)
    rs = sc.robots[0].robot_spec
    meta = dict(episode_id=f"{body}_{episode_tag(task)}_s{seed}", robot=body, spec_hash=rs.spec_hash, lineage=rs.lineage,
                family=rs.family, synthetic=rs.synthetic, controller_version=s.controller_version(),
                tracker_source=s.tracker.source, tracker_sha256=getattr(s.tracker, "sha256", None), task=task,
                task_hash=hashlib.sha256(json.dumps(sc.task, sort_keys=True).encode()).hexdigest()[:16],
                seed=seed, control_dt=s.dt, physics_dt=float(s.model.opt.timestep), tracker_hz=50.0, steps=steps,
                max_steps=max_steps, status=status, failure_reason=reason, public_runtime_success=bool(s.runtime.succeeded()),
                event_status={e: i.status for e, i in s.runtime.instances.items()},
                source=pol.info.source, privileged_teacher=pol.info.requires.privileged, teacher=pol.info.name,
                teacher_version=pol.info.version, teacher_variant="arc_only" if arc_only else "default",
                waypoints=sc.meta.get("waypoints"), wall_s=time.time() - t0,
                split_lineage=split_lineage or {"lineage": rs.lineage},
                tracker_source_label=str(parse_source(s.tracker.source)),
                physics=physics_provenance(s.model, sc.meta.get("contact_model", CONTACT_VERSION_DEFAULT)).to_dict())
    public = dict(meta=meta, inputs=inputs, actions=actions,
                  action_space=dict(group="base_velocity", units=["m/s", "m/s", "rad/s"],
                                    lower=s.tracker_contract.command_groups[0].lower,
                                    upper=s.tracker_contract.command_groups[0].upper))
    private = dict(meta=dict(episode_id=meta["episode_id"], kind="privileged_labels"), labels=labels, phases=phases)
    return EpisodeRecord(public, private)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--task", default="waypoint_contact", help="a registered mujoco/legged task with a scripted teacher")
    ap.add_argument("--tracker", default="auto")
    ap.add_argument("--tracker-id", default=None, help="registered actor <body>:<version> (default: the body's actor)")
    ap.add_argument("--max-steps", type=int, default=None, help="control steps; default TaskSpec.max_steps, else 1300")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arc-only", action="store_true", help="teacher keeps min forward speed while turning")
    ap.add_argument("--teacher-report", action="store_true",
                    help="also write artifacts/assets/legged_teacher/<body>.json (teacher validation gate)")
    a = ap.parse_args(argv)
    out = Path(a.out) / a.body
    seeds = parse_seed_spec(a.seeds)
    guard_sealed(a.body, seeds)
    assert_one_task(out, a.task)
    rows = []
    from rrp.harness.data.collect import read_episode
    for sd in seeds:
        tag = episode_tag(a.task)
        pub = out / f"{a.body}_{tag}_s{sd}.public.pkl.gz"
        if pub.exists() and (out / f"{a.body}_{tag}_s{sd}.private.pkl.gz").exists() and \
                read_episode(pub)["meta"]["tracker_source"] == ("scripted_controller" if a.tracker == "cpg" else "learned_tracker"):
            m = dict(read_episode(pub)["meta"], files={})       # resumable: keep finished episodes
        else:
            rec = collect_episode(a.body, sd, a.tracker, a.max_steps, arc_only=a.arc_only, task=a.task,
                                  tracker_id=a.tracker_id)
            m = write_episode(rec, out)
        rows.append(dict(episode_id=m["episode_id"], status=m["status"], failure_reason=m.get("failure_reason"),
                         steps=m["steps"], files=m["files"],
                         tracker_source=m["tracker_source"], tracker_sha256=m.get("tracker_sha256"),
                         event_status=m["event_status"], source=m.get("source", "scripted_teacher"),
                         teacher=m.get("teacher"), teacher_version=m.get("teacher_version"), physics=m.get("physics"),
                         tracker_version=m.get("controller_version")))
        print(json.dumps(rows[-1]), flush=True)
    summ = dict(body=a.body, task=a.task, n=len(rows), success=sum(r["status"] == "success" for r in rows),
                fell=sum(r["status"] == "fell" for r in rows), episodes=rows)
    trk = sorted({r["tracker_source"] for r in rows})
    prov = dataset_provenance(rows, source="scripted_teacher",
                              flags=dict(privileged_teacher=True, arc_only=a.arc_only, tracker=a.tracker, task=a.task,
                                         prev_action_input=False),
                              notes="native actions = base_velocity commands executed by the body tracker")
    prov.versions.update(tracker_source="|".join(str(parse_source(t)) for t in trk),
                         tracker_version="|".join(sorted({str(r.get("tracker_version")) for r in rows})),
                         tracker_sha256="|".join(sorted({str(r.get("tracker_sha256")) for r in rows})),
                         teacher="|".join(sorted({f"{r.get('teacher')}@{r.get('teacher_version')}" for r in rows})))
    write_manifest(out, f"legged_{a.task}_{a.body}", rows,
                   extra={k: v for k, v in summ.items() if k != "episodes"} | dict(source=prov.source), provenance=prov)
    if a.teacher_report:
        stem = a.body if a.task == "waypoint_contact" else f"{a.body}_{a.task}"
        rep = rrp_home() / "artifacts" / "assets" / "legged_teacher" / (f"{stem}_arc_only.json" if a.arc_only else f"{stem}.json")
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(dict(body=a.body, task=a.task, n=len(rows),
                                       success_rate=summ["success"] / max(1, len(rows)), fell=summ["fell"],
                                       tracker_source=rows[0]["tracker_source"] if rows else None,
                                       teacher=f"{rows[0]['teacher']} (scripted_teacher, privileged)" + (" arc_only variant" if a.arc_only else "")
                                       if rows else None,
                                       seeds=a.seeds, episodes=rows), indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "episodes"}))
