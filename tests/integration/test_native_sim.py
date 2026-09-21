import numpy as np
import pytest

from rrp.sim.fixtures import make_arm_session, make_pick_place_session
from rrp.sim.snapshot_contract import Snapshot, SnapshotError
from rrp.contracts.action import NativeCommand, ActionChunk, GroupCommand
from rrp.contracts.errors import ControllerRejection, StaleActionError
from rrp.contracts.channels import serialize_public
from rrp.control.teachers import PickPlaceTeacher


def test_physics_steps_and_full_restore():
    sim = make_arm_session(seed=7)
    snap = sim.snapshot()
    command = sim.fixture_command()
    first = sim.step(command)
    sim.restore(snap)
    second = sim.step(command)
    assert np.allclose(first.qpos, second.qpos, atol=1e-8)
    assert np.isfinite(first.qpos).all()


def test_restore_is_a_full_continuation_including_runtime_tracker_and_rng():
    s = make_pick_place_session(seed=3)
    t = PickPlaceTeacher(s)
    for _ in range(25):
        s.step(t.act())
    snap = s.snapshot()
    t_state = t.state()
    run1 = []
    for _ in range(30):
        c = t.act()
        o = s.step(c)
        run1.append((o.qpos.copy(), [d.position_estimate for d in o.observation.object_descriptors],
                     {k: v.status for k, v in s.runtime.instances.items()}))
    s.restore(snap)
    t.load(t_state)
    for k in range(30):
        c = t.act()
        o = s.step(c)
        q, est, st = run1[k]
        assert np.allclose(o.qpos, q, atol=1e-9)
        assert [d.position_estimate for d in o.observation.object_descriptors] == est
        assert {kk: v.status for kk, v in s.runtime.instances.items()} == st


def test_snapshot_requires_every_component():
    s = make_arm_session(seed=1)
    snap = s.snapshot()
    comps = dict(snap.components)
    comps.pop("task_runtime")
    with pytest.raises(SnapshotError):
        Snapshot(comps).validate()
    comps = dict(snap.components)
    comps["physics"] = {k: v for k, v in comps["physics"].items() if k != "qacc_warmstart"}
    with pytest.raises(SnapshotError):
        Snapshot(comps).validate()


def test_controller_rejects_invalid_commands():
    s = make_arm_session(seed=2)
    v = s.controller_version()
    ok = s.fixture_command()
    for bad, code in [
        (NativeCommand(controller_version="other", groups=ok.groups, source="user"), "stale_action_chunk"),
        (NativeCommand(controller_version=v, groups={"arm": [0.0] * 4}, source="user"), "wrong_width"),
        (NativeCommand(controller_version=v, groups={"arm": [float("nan")] * 5}, source="user"), "nonfinite"),
        (NativeCommand(controller_version=v, groups={"arm": [9.0] * 5}, source="user"), "out_of_bounds"),
        (NativeCommand(controller_version=v, groups={"legs": [0.0]}, source="user"), "unknown_group"),
    ]:
        r = s.step(bad)
        assert r.rejected == code, (code, r.rejected)
    assert s.step(ok).rejected is None


def _chunk(s, graph_version, dt=None, h=4):
    tg = s.robots[0].controller.current_targets(s.data)
    return ActionChunk(observation_id="o", graph_version=graph_version, runtime_version=s.runtime.runtime_version,
                       robot_spec_hash=s.robots[0].spec.spec_hash, controller_version=s.controller_version(),
                       policy_version="fixture", codec_version=None, start_time=float(s.data.time),
                       dt=dt or s.dt, horizon=h,
                       command_groups=[GroupCommand(group="arm", values=np.tile(tg["arm"], (h, 1)),
                                                    mask=np.ones((h, 5), bool))],
                       sampling_seed=0, source="learned")


def test_stale_chunks_rate_mismatch_and_edit_invalidation():
    s = make_pick_place_session(seed=0)
    assert abs(s.dt - 0.05) < 1e-12
    with pytest.raises(StaleActionError):
        s.submit_chunk(_chunk(s, s.runtime.graph_version - 1))
    with pytest.raises(StaleActionError):
        s.submit_chunk(_chunk(s, s.runtime.graph_version, dt=0.02))
    s.submit_chunk(_chunk(s, s.runtime.graph_version, h=6))
    assert len(s.executor.queue) == 6
    s.step()
    assert len(s.executor.queue) == 5
    s.runtime.apply_edit([{"op": "set_priority", "event_id": "grasp", "priority": 1}],
                         expected_version=s.runtime.graph_version, request_id="edit-during-queue")
    assert len(s.executor.queue) == 0
    assert any(e["event"] == "queue_dropped" for e in s.executor.log)


def test_public_observation_is_serializable_and_privileged_is_separate():
    s = make_pick_place_session(seed=0, n_distractors=2)
    o = s.observe()
    serialize_public(o)
    t = s.truth(o.observation_id)
    assert "cube" in t.object_poses and t.object_entity_map["cube"] == "cube"
    assert len(o.object_descriptors) == 4 and {d.descriptor for d in o.object_descriptors} >= {"red cube", "blue cube"}


def test_teleport_is_logged_as_contaminating_intervention():
    s = make_pick_place_session(seed=0)
    s.teleport_object("cube", [0.4, 0.0, 0.1])
    assert s.intervention_log[-1]["contaminates_evaluation"] is True
    assert s.snapshot().components["interventions"]
