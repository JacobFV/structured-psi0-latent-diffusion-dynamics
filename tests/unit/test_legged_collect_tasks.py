"""H4 (D-146): the legged collectors take `--task` through the task registry and the teacher factory. Events and
`public_context` come from the TaskSpec's graph; the manifest/episode meta record task, teacher source and tracker sha;
the sealed guard is called. 20-tick episode per task on a stub (random-weight) tracker; nothing here trains."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_trackers import _binding, _put, registry   # noqa: E402,F401  (registry: fixture)

import rrp.envs.mujoco.legged_tracker as LT   # noqa: E402
from rrp.harness.data import legged_collect as LCOL, legged_latent_collect as LLC   # noqa: E402
from rrp.policies.features import legged as F   # noqa: E402

BODY = "t1"
TASKS = ("waypoint_contact", "h_steps", "h_gap")
TICKS = 20


def _graph(n_events: int, n_targets: int) -> dict:
    ents = [f"e{i}" for i in range(n_targets)]
    evs = [dict(id=f"ev{k}", roles=[dict(role="target", binding=dict(entity=dict(id=ents[k % n_targets])))])
           for k in range(n_events)]
    for r, e in zip(F.ENTITY_ROLES[1:], ents[n_events:]):         # entities no event's target role reaches ride in other roles
        evs[0]["roles"].append(dict(role=r, binding=dict(entity=dict(id=e))))
    return dict(task_id="synthetic", graph_version="v0", events=evs)


def test_task_view_waypoint_contact_is_the_old_layout():
    v = F.task_view_of("waypoint_contact")
    assert v.events == F.EVENTS and len(v.targets) == 2 and F.EVENT_SLOTS == len(F.EVENTS)
    assert F.GLOBAL_DIM == 22 + 4 * (F.TARGET_SLOTS - F.LEAD_TARGET_SLOTS)      # the 22 legacy columns come first


@pytest.mark.parametrize("ne,nt", [(F.EVENT_SLOTS + 1, 1), (3, F.TARGET_SLOTS + 1)])
def test_task_view_refuses_a_graph_the_context_cannot_hold(ne, nt):
    with pytest.raises(ValueError, match="legged context holds"):
        F.TaskView(_graph(ne, nt))


def test_task_view_pads_short_graphs():
    v = F.TaskView(_graph(2, 1))
    assert v.events == ("ev0", "ev1") and v.targets == ("e0",) and v.as_dict()["event_slots"] == F.EVENT_SLOTS


def test_no_teacher_task_is_refused():
    with pytest.raises(ValueError, match="no scripted teacher"):
        LCOL.open_episode("foothold_steps", BODY, 0)


def test_unknown_task_is_refused():
    with pytest.raises(KeyError):
        LCOL.open_episode("no_such_task", BODY, 0)


def test_sealed_body_and_seed_are_refused():
    from rrp.core.sealed import SealedSplit, SealedSplitError
    sp = SealedSplit.load()
    with pytest.raises(SealedSplitError):
        LCOL.guard_sealed(BODY, [sp.ranges["evaluation"][0]])
    LCOL.guard_sealed(BODY, [sp.ranges["source"][0]])       # a development body on a source-train seed is fine


def test_one_directory_one_task(tmp_path):
    (tmp_path / "x_manifest.json").write_text(json.dumps(dict(task="h_gap")))
    LCOL.assert_one_task(tmp_path, "h_gap")
    with pytest.raises(SystemExit, match="own directory"):
        LCOL.assert_one_task(tmp_path, "h_steps")
    (tmp_path / "x_manifest.json").write_text("{}")          # older manifests: waypoint_contact
    LCOL.assert_one_task(tmp_path, "waypoint_contact")
    with pytest.raises(SystemExit):
        LCOL.assert_one_task(tmp_path, "h_gap")


def _sha(spec: str) -> str:
    import hashlib
    return hashlib.sha256(LT.get_entry(spec).actor.read_bytes()).hexdigest()


@pytest.fixture
def stub_tracker(tmp_path, registry):
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps
    b = _binding(build_h_steps(BODY, 0, h_frac=0.2))
    _put(tmp_path, BODY, "stub_v2", b, scan=False, seed=5)                       # the h_* scenes use contact_v2 ...
    _put(tmp_path, BODY, "stub_v1", b, scan=False, seed=6, contact="contact_v1")   # ... waypoint_contact contact_v1
    registry()
    return lambda task: f"{BODY}:stub_v1" if task == "waypoint_contact" else f"{BODY}:stub_v2"


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TASKS)
def test_20_tick_episode_per_task(task, stub_tracker):
    rec = LCOL.collect_episode(BODY, 0, task=task, max_steps=TICKS, tracker_id=stub_tracker(task))
    m = rec.public["meta"]
    view = F.task_view_of(task)
    assert m["task"] == task and m["steps"] <= TICKS and m["max_steps"] == TICKS
    assert m["tracker_sha256"] == _sha(stub_tracker(task)) and m["tracker_source"]
    assert m["teacher"].startswith("teacher:") and m["source"] and m["privileged_teacher"] is True
    assert list(m["event_status"]) == list(view.events)
    assert m["status"] in ("success", "fell", "failure") and (m["status"] == "success") == (m["failure_reason"] is None)
    assert len(rec.public["inputs"]) == m["steps"] == len(rec.private["labels"])
    assert task in m["episode_id"] or task == "waypoint_contact"


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TASKS)
def test_public_context_slots_follow_the_task(task, stub_tracker):
    spec, s = LCOL.open_episode(task, BODY, 0, tracker_id=stub_tracker(task))
    s.reset()
    view = F.task_view_of(task)
    ctx = F.public_context(s, 0.0)
    assert ctx.shape == (F.GLOBAL_DIM,) and np.isfinite(ctx).all()
    ev = ctx[16:16 + F.EVENT_SLOTS + 1]
    assert ev.sum() == 1.0 and ev.argmax() == 0                   # first event of that task's graph is active
    for k in range(F.TARGET_SLOTS):                                # valid flag: 1 for a slot the task fills, else 0
        col = ctx[F.target_slot_cols(k)]
        assert col[3] in (0.0, 1.0)
        if k >= len(view.targets):
            assert col.tolist() == [0, 0, 0, 0]


@pytest.mark.menagerie
@pytest.mark.parametrize("task", TASKS)
def test_latent_collect_20_ticks_records_task_and_teacher(task, stub_tracker):
    arr, meta, _ = LLC.collect_episode(BODY, 0, 0.0, task=task, max_steps=TICKS, tracker_id=stub_tracker(task))
    assert meta["task"] == task and meta["teacher"].startswith("teacher:") and meta["max_steps"] == TICKS
    assert meta["tracker_sha256"] == _sha(stub_tracker(task)) and meta["steps"] <= TICKS
    assert arr["ctx"].shape[1] == F.GLOBAL_DIM and arr["ev"].max() <= F.EVENT_SLOTS


def test_recording_tracker_forwards_the_direct_legs_target():
    """D-147 (2026-10-03): under legs / wholebody control the session hands the teacher's `legs` targets to `session.tracker.pending`; the
    collector's RecordingTracker wraps that slot and must forward them (before: the slot held the default stance and every wholebody
    collection fell at 1 s with all recorded actions 0)."""
    from types import SimpleNamespace as NS
    import numpy as np
    from rrp.envs.mujoco.legged import DirectTargets
    from rrp.harness.data.legged_latent_collect import RecordingTracker
    b = NS(q0=np.zeros(3), lo=-np.ones(3), hi=np.ones(3))
    slot = DirectTargets(b)
    s = NS(tracker=slot, control="wholebody", terrain=NS(values=None), binding=b, body_tracker=NS(source="learned_tracker", version="v"))
    rt = RecordingTracker(s, morph=NS(gait_period=0.7), sigma=0.0, rng=np.random.default_rng(0))
    s.tracker = rt
    s.tracker.pending = np.array([0.1, -0.2, 0.3])            # what LeggedSession._step_legs does
    assert np.allclose(rt.act(None, None), [0.1, -0.2, 0.3])  # the commanded target is executed, not the default stance
    assert np.allclose(rt.act(None, None), 0.0)               # one tick only, then the declared fallback
