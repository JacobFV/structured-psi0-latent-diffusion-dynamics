import pytest
from rrp.control.teachers import run_fixture_pick_place, PickPlaceTeacher, run_teacher_episode
from rrp.sim.fixtures import make_pick_place_session


def test_real_teacher_trace_has_actions_and_success_evidence():
    result = run_fixture_pick_place(seed=4, max_control_steps=600)
    assert len(result.actions) > 0
    assert result.controller_source == "scripted_teacher"
    assert result.success == result.privileged_evaluator_success
    assert result.success


@pytest.mark.parametrize("gripper", ["parallel", "three_finger"])
def test_teacher_valid_for_both_compatible_modules(gripper):
    ok = 0
    for seed in range(8):
        r = run_fixture_pick_place(seed=seed, gripper=gripper)
        assert r.privileged_inputs and r.controller_source == "scripted_teacher"
        ok += int(r.success and r.privileged_evaluator_success)
    assert ok >= 7


def test_infeasible_episode_is_recorded_not_attempted():
    s = make_pick_place_session(seed=0)
    s.teleport_object("target_zone", [1.5, 0.0, 0.0005])
    r = run_teacher_episode(s, PickPlaceTeacher(s))
    assert r.failure_reason.startswith("infeasible") and r.steps == 0
    assert "transport" in r.feasibility["unreachable"]
